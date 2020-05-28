# -*- coding: utf-8 -*-

# pylint: disable=unused-import, superfluous-parens, line-too-long, missing-module-docstring, unused-variable, missing-function-docstring, invalid-name

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_mono as mc
import sksurgerycalibration.video.video_calibration_driver_stereo as sc
import sksurgerycalibration.video.video_calibration_utils as vu


def get_iterative_reference_data():
    number_of_points = 140
    x_size = 14
    y_size = 10
    pixels_per_square = 20
    reference_ids = np.zeros((number_of_points, 1))
    reference_points = np.zeros((number_of_points, 2))
    counter = 0
    for y_index in range(y_size):
        for x_index in range(x_size):
            reference_ids[counter][0] = counter
            reference_points[counter][0] = (x_index + 2) * pixels_per_square
            reference_points[counter][1] = (y_index + 2) * pixels_per_square
            counter = counter + 1
    reference_image_size = ((x_size + 4) * pixels_per_square, (y_size + 4) * pixels_per_square)
    return reference_ids, reference_points, reference_image_size


def test_chessboard_mono():

    images = []

    files = glob.glob('tests/data/laparoscope_calibration/left/*.png')
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    # This illustrates that the PointDetector sub-class holds the knowledge of the model.
    chessboard_detector = \
        pd.ChessboardPointDetector((14, 10),    # (Width, height), number of internal corners
                                   3,           # Actual square size in mm
                                   (1, 1)       # Scale factors. Here, images are 1920x1080 so no scaling needed.
                                   )            # If images were 1920x540, you'd pass in (1920, 540),
                                                # and PointDetector base class would scale it up.

    # Pass a PointDetector sub-class to the calibration driver.
    calibrator = \
        mc.MonoVideoCalibrationDriver(chessboard_detector, 140)

    # Repeatedly grab data, until you have enough.
    for image in images:
        successful = calibrator.grab_data(image, np.eye(4))
        assert successful > 0
    assert calibrator.is_device_tracked()
    assert not calibrator.is_calibration_target_tracked()

    # Extra checking, as its a unit test
    assert calibrator.get_number_of_views() == 9
    calibrator.pop()
    assert calibrator.get_number_of_views() == 8
    successful = calibrator.grab_data(images[-1])
    assert successful
    assert calibrator.get_number_of_views() == 9

    # Then do calibration
    reproj_err, recon_err, params = calibrator.calibrate()

    # Just for a regression test, checking reprojection error, and recon error.
    # We do appear to get different performance on Linux/Mac
    assert reproj_err < 0.6
    assert recon_err < 0.3

    # Test components of iterative calibration.
    original_image = calibrator.video_data.images_array[0]
    _, _, original_pts = chessboard_detector.get_points(original_image)

    # First ensure undistorting / redistorting points works.
    undistorted_points = cv2.undistortPoints(original_pts,
                                             calibrator.calibration_params.camera_matrix,
                                             calibrator.calibration_params.dist_coeffs,
                                             None,
                                             calibrator.calibration_params.camera_matrix
                                             )
    distorted_pts = vu.distort_points(undistorted_points.reshape((-1, 2)),
                                      calibrator.calibration_params.camera_matrix,
                                      calibrator.calibration_params.dist_coeffs)
    assert np.allclose(original_pts, distorted_pts, rtol=1e-4, atol=1e-4)

    rectify_map_1, rectify_map_2 = cv2.initUndistortRectifyMap(calibrator.calibration_params.camera_matrix,
                                                               calibrator.calibration_params.dist_coeffs,
                                                               None,
                                                               calibrator.calibration_params.camera_matrix,
                                                               (original_image.shape[1], original_image.shape[0]),
                                                               cv2.CV_32FC1
                                                               )
    #undistorted_image = cv2.undistort(original_image, calibrator.calibration_params.camera_matrix, calibrator.calibration_params.dist_coeffs, calibrator.calibration_params.camera_matrix)
    undistorted_image = cv2.remap(original_image, rectify_map_1, rectify_map_2, cv2.INTER_LANCZOS4)

    _, _, undistorted_pts = chessboard_detector.get_points(undistorted_image)
    distorted_pts = vu.distort_points(undistorted_pts, calibrator.calibration_params.camera_matrix, calibrator.calibration_params.dist_coeffs)
    assert np.allclose(original_pts, distorted_pts, rtol=1e-1, atol=1e-1)

    # Test iterative calibration.
    reference_ids, reference_points, reference_image_size = get_iterative_reference_data()

    reproj_err, recon_err, params = calibrator.iterative_calibration(3,
                                                                     reference_ids,
                                                                     reference_points,
                                                                     reference_image_size)
    assert reproj_err < 0.7
    assert recon_err < 0.4


def test_chessboard_stereo():

    left_images = []
    files = glob.glob('tests/data/laparoscope_calibration/left/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        left_images.append(image)
    assert(len(left_images) == 9)

    right_images = []
    files = glob.glob('tests/data/laparoscope_calibration/right/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        right_images.append(image)
    assert (len(right_images) == 9)

    chessboard_detector = \
        pd.ChessboardPointDetector((14, 10),
                                   3,
                                   (1, 1)
                                   )

    calibrator = \
        sc.StereoVideoCalibrationDriver(chessboard_detector, 140)

    # Repeatedly grab data, until you have enough.
    for i, _ in enumerate(left_images):
        successful = calibrator.grab_data(left_images[i], right_images[i])
        assert successful > 0
    assert not calibrator.is_device_tracked()
    assert not calibrator.is_calibration_target_tracked()

    # Then do calibration
    reproj_err, recon_err, params = calibrator.calibrate()

    # Just for a regression test, checking reprojection error, and recon error.
    assert reproj_err < 0.7
    assert recon_err < 1.7

    # Test iterative calibration.
    reference_ids, reference_points, reference_image_size = get_iterative_reference_data()

    reproj_err, recon_err, params = calibrator.iterative_calibration(3,
                                                                     reference_ids,
                                                                     reference_points,
                                                                     reference_image_size)
    assert reproj_err < 0.7
    assert recon_err < 1.5
