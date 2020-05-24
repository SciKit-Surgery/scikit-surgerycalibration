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
import sksurgerycalibration.video.video_calibration_utils as vu

import sksurgerycalibration.video as vidcal
import sksurgerycalibration.video.video_calibration_metrics as vcm

def test_set_model2hand_arrays():
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    stereo_calib.tracking_data.set_model2hand_arrays()
    print(stereo_calib.tracking_data.quat_model2hand_array)
    print(stereo_calib.tracking_data.trans_model2hand_array)


def load_images_from_glob(glob_pattern):
    images = []
    files = glob.glob(glob_pattern)
    files.sort()
    for f in files:
        image = cv2.imread(f)
        images.append(image)

    return images

def load_tracking_from_glob(glob_pattern):
    tracking = []
    files = glob.glob(glob_pattern)
    files.sort()
    for f in files:
        data = np.loadtxt(f)
        tracking.append(data)

    return tracking

def test_handeye_calibration_stereo():
    
    left_images = load_images_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.left.*.png')

    right_images = load_images_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.right.*.png')

    device_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.device_tracking.*.txt')

    obj_tracking = load_tracking_from_glob(
        'tests/data/2020_01_20_storz/12_50_30/calib.calib_obj_tracking.*.txt')   

    assert(len(left_images) == 10)
    assert(len(right_images) == 10)
    assert(len(device_tracking) == 10)
    assert(len(obj_tracking) == 10)

    min_number_of_points_per_image = 50
    detector = \
        chpd.CharucoPlusChessboardPointDetector(error_if_no_chessboard=False)
    
    calibrator = \
        vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(
            detector, min_number_of_points_per_image)

    # Grab data from images/tracking arrays
    for l, r, device, calib_obj in zip(left_images, right_images, device_tracking, obj_tracking):
        successful = calibrator.grab_data(l, r, device, calib_obj)
        assert successful > 0

    reproj_err_1, recon_err_1, params_1 = calibrator.calibrate()

    one_pixel = 1
    assert(reproj_err_1 < one_pixel)
    assert(recon_err_1 < one_pixel)

    proj_err, recon_err = calibrator.handeye_calibration()

    print(f'Reproj err {proj_err}')
    print(f'Recon err {recon_err}')

    # TODO: These aren't objective values
    # Reconstruction error will fail with these values.
    acceptable_reproj_error = 25
    acceptable_recon_error = 25

    assert(proj_err < acceptable_reproj_error)
    assert(recon_err < acceptable_recon_error)

def test_load_data_stereo_calib():
    """ Load tracking and image data from test directory. """
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    assert(len(stereo_calib.tracking_data.device_tracking_array) == 10)
    assert(len(stereo_calib.tracking_data.calibration_tracking_array) == 10)

    assert(len(stereo_calib.video_data.left_data.images_array) == 10)
    assert(len(stereo_calib.video_data.right_data.images_array) == 10)