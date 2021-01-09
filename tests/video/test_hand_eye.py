# -*- coding: utf-8 -*-

"""
Tests for calibration_manager.
"""

import glob
import cv2
import pytest
import numpy as np
import sksurgeryimage.calibration.chessboard_point_detector as pd
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector as chpd
import sksurgerycalibration.video as vidcal


def test_set_model2hand_arrays():
    """Test that model2hand arrays are set without raising an error"""
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = \
        vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(
            chessboard_detector, chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    stereo_calib.tracking_data.set_model2hand_arrays()


def load_images_from_glob(glob_pattern):
    """ Load images from files based on a glob pattern. """
    images = []
    files = glob.glob(glob_pattern)
    files.sort()
    for file in files:
        image = cv2.imread(file)
        images.append(image)

    return images


def load_tracking_from_glob(glob_pattern):
    """ Load tracking data from files based on a glob pattern. """
    tracking = []
    files = glob.glob(glob_pattern)
    files.sort()
    for file in files:
        data = np.loadtxt(file)
        tracking.append(data)

    return tracking


def test_handeye_calibration_mono():
    """
    Load mono data (only using left channel) and tracking, do video and
    handeye calibration, compare results against expected values.
    """
    images = load_images_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.left.*.png')

    device_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.device_tracking.*.txt')

    obj_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.calib_obj_tracking.*.txt')

    ref_img = cv2.imread(
        'tests/data/2020_01_20_storz/pattern_4x4_19x26_5_4_with_inset_9x14.png')

    assert len(images) == 10

    assert len(device_tracking) == 10
    assert len(obj_tracking) == 10

    min_number_of_points_per_image = 50
    detector = \
        chpd.CharucoPlusChessboardPointDetector(ref_img,
                                                error_if_no_chessboard=False)

    calibrator = \
        vidcal.video_calibration_driver_mono.MonoVideoCalibrationDriver(
            detector, min_number_of_points_per_image)

    # Grab data from images/tracking arrays
    for image, device, calib_obj in zip(images, device_tracking, obj_tracking):
        successful = calibrator.grab_data(image, device, calib_obj)
        assert successful > 0

    reproj_err_1, recon_err_1, _ = calibrator.calibrate()

    one_pixel = 1
    assert reproj_err_1 < one_pixel
    assert recon_err_1 < one_pixel

    proj_err, recon_err, _ = calibrator.handeye_calibration()

    print(f'Reproj err {proj_err}')
    print(f'Recon err {recon_err}')

    # These values are taken from a previous successful run
    # Not objective measures of correctness
    expected_reproj_error = 14.97731
    expected_recon_error = 5.434362

    assert proj_err == pytest.approx(expected_reproj_error, rel=1e-4)
    assert recon_err == pytest.approx(expected_recon_error, rel=1e-4)


def test_handeye_calibration_stereo():
    """
    Load Stereo data and tracking, do video and
    handeye calibration, compare results against expected values.
    """
    left_images = load_images_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.left.*.png')

    right_images = load_images_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.right.*.png')

    device_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.device_tracking.*.txt')

    obj_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.calib_obj_tracking.*.txt')

    ref_img = cv2.imread(
        'tests/data/2020_01_20_storz/pattern_4x4_19x26_5_4_with_inset_9x14.png')

    assert len(left_images) == 10
    assert len(right_images) == 10
    assert len(device_tracking) == 10
    assert len(obj_tracking) == 10

    min_number_of_points_per_image = 50
    detector = \
        chpd.CharucoPlusChessboardPointDetector(ref_img,
                                                charuco_filtering=True,
                                                error_if_no_chessboard=False)

    calibrator = \
        vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(
            detector, detector, min_number_of_points_per_image)

    # Grab data from images/tracking arrays
    for left, right, device, calib_obj in \
        zip(left_images, right_images, device_tracking, obj_tracking):
        num_left, num_right = calibrator.grab_data(left,
                                                   right,
                                                   device,
                                                   calib_obj)
        assert num_left > 0
        assert num_right > 0

    reproj_err_1, recon_err_1, _ = calibrator.calibrate()

    one_pixel = 1
    assert reproj_err_1 < one_pixel
    assert recon_err_1 < one_pixel

    proj_err, recon_err, _ = calibrator.handeye_calibration()

    print(f'Reproj err {proj_err}')
    print(f'Recon err {recon_err}')

    # These values are taken from a previous successful run
    # Not objective measures of correctness
    # expected_reproj_error = 1.77138
    # expected_recon_error = 1.24267
    # Unfortunately the tolerances need to change depending on OS.

    assert proj_err < 2
    assert recon_err < 2

    # test save/load for hand-eye
    calibrator.save_params('tests/output/test_handeye_calibration_stereo', '')
    current_params = calibrator.get_params()

    calibrator.reinit()
    calibrator.load_params('tests/output/test_handeye_calibration_stereo', '')
    loaded_params = calibrator.get_params()
    assert np.allclose(current_params.left_params.handeye_matrix,
                       loaded_params.left_params.handeye_matrix)
    assert np.allclose(current_params.left_params.pattern2marker_matrix,
                       loaded_params.left_params.pattern2marker_matrix)
    assert np.allclose(current_params.right_params.handeye_matrix,
                       loaded_params.right_params.handeye_matrix)
    assert np.allclose(current_params.right_params.pattern2marker_matrix,
                       loaded_params.right_params.pattern2marker_matrix)


def test_load_data_stereo_calib():
    """ Load tracking and image data from test directory. """
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = \
        vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(
            chessboard_detector, chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    assert len(stereo_calib.tracking_data.device_tracking_array) == 10
    assert len(stereo_calib.tracking_data.calibration_tracking_array) == 10
    assert len(stereo_calib.video_data.left_data.images_array) == 10
    assert len(stereo_calib.video_data.right_data.images_array) == 10
