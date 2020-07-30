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
import tests.video.video_testing_utils as vtu


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

    left_images, right_images \
        = vtu.load_left_right_pngs('tests/data/laparoscope_calibration/', 9)

    chessboard_detector = \
        pd.ChessboardPointDetector((14, 10),
                                   3,
                                   (1, 1)
                                   )

    calibrator = \
        sc.StereoVideoCalibrationDriver(chessboard_detector, chessboard_detector, 140)

    # Repeatedly grab data, until you have enough.
    for i, _ in enumerate(left_images):
        number_left, number_right = calibrator.grab_data(left_images[i], right_images[i])
        assert number_left > 0
        assert number_right > 0
    assert not calibrator.is_device_tracked()
    assert not calibrator.is_calibration_target_tracked()

    # Then do calibration
    reproj_err, recon_err, params = calibrator.calibrate()

    # Just for a regression test, checking reprojection error, and recon error.
    print("\nStereo, default=" + str(reproj_err) + ", " + str(recon_err))
    assert reproj_err < 0.7
    assert recon_err < 1.7

    # Test running with fixed intrinsics and fixed stereo, using existing
    # calibration parameters, thereby re-optimising the camera poses.
    reproj_err, recon_err, params = \
        calibrator.calibrate(
            override_left_intrinsics=params.left_params.camera_matrix,
            override_left_distortion=params.left_params.dist_coeffs,
            override_right_intrinsics=params.right_params.camera_matrix,
            override_right_distortion=params.right_params.dist_coeffs,
            override_l2r_rmat=params.l2r_rmat,
            override_l2r_tvec=params.l2r_tvec
        )

    # The above re-optimisation shouldn't make things worse, as its using same intrinsics and stereo.
    print("Stereo, re-optimise=" + str(reproj_err) + ", " + str(recon_err))
    assert reproj_err < 0.7
    assert recon_err < 1.7

    # Test iterative calibration.
    reference_ids, reference_points, reference_image_size = get_iterative_reference_data()

    reproj_err, recon_err, params = calibrator.iterative_calibration(3,
                                                                     reference_ids,
                                                                     reference_points,
                                                                     reference_image_size)
    print("Stereo, iterative=" + str(reproj_err) + ", " + str(recon_err))
    assert reproj_err < 0.7
    assert recon_err < 1.6

    # Now test re-optimising extrinsics, using a completely different set of calibration params.
    ov_l_c = np.loadtxt('tests/data/laparoscope_calibration/cbh-viking/calib.left.intrinsics.txt')
    ov_l_d = np.loadtxt('tests/data/laparoscope_calibration/cbh-viking/calib.left.distortion.txt')
    ov_r_c = np.loadtxt('tests/data/laparoscope_calibration/cbh-viking/calib.right.intrinsics.txt')
    ov_r_d = np.loadtxt('tests/data/laparoscope_calibration/cbh-viking/calib.right.distortion.txt')

    ov_l2r_t = np.zeros((3, 1))
    ov_l2r_t[0][0] = -4.5

    reproj_err, recon_err, params = \
        calibrator.calibrate(
            override_left_intrinsics=ov_l_c,
            override_left_distortion=ov_l_d,
            override_right_intrinsics=ov_r_c,
            override_right_distortion=ov_r_d,
            override_l2r_rmat=np.eye(3),
            override_l2r_tvec=ov_l2r_t
        )

    # Must check that the overrides have actually been set on the output.
    assert np.allclose(params.left_params.camera_matrix, ov_l_c)
    assert np.allclose(params.left_params.dist_coeffs, ov_l_d)
    assert np.allclose(params.right_params.camera_matrix, ov_r_c)
    assert np.allclose(params.right_params.dist_coeffs, ov_r_d)
    assert np.allclose(params.l2r_rmat, np.eye(3))
    assert np.allclose(params.l2r_tvec, ov_l2r_t)

    # Not expecting good results, as the camera parameters are completely wrong.
    print("Stereo, override=" + str(reproj_err) + ", " + str(recon_err))
    assert reproj_err < 33
    assert recon_err < 109
