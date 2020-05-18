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

def test_set_model2hand_arrays():
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    stereo_calib.tracking_data.set_model2hand_arrays()
    print(stereo_calib.tracking_data.quat_model2hand_array)
    print(stereo_calib.tracking_data.trans_model2hand_array)


def test_handeye_calibration_stereo():
    
    left_images = []
    files = glob.glob('tests/data/2020_01_20_storz/12_50_30/calib.left.*.png')
    files.sort()
    for f in files:
        image = cv2.imread(f)
        left_images.append(image)
    assert(len(left_images) == 10)

    right_images = []
    files = glob.glob('tests/data/2020_01_20_storz/12_50_30/calib.right.*.png')
    files.sort()
    for f in files:
        image = cv2.imread(f)
        right_images.append(image)
    assert(len(right_images) == 10)

    device_tracking = []
    files = glob.glob('tests/data/2020_01_20_storz/12_50_30/calib.device_tracking.*.txt')
    files.sort()
    for f in files:
        data = np.loadtxt(f)
        device_tracking.append(data)
    assert(len(device_tracking) == 10)

    obj_tracking = []
    files = glob.glob('tests/data/2020_01_20_storz/12_50_30/calib.calib_obj_tracking.*.txt')
    files.sort()
    for f in files:
        data = np.loadtxt(f)
        obj_tracking.append(data)
    assert(len(obj_tracking) == 10)

    minimum_number_of_points_per_image = 50
    detector = chpd.CharucoPlusChessboardPointDetector(error_if_no_chessboard=False)
    
    calibrator = \
        vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(detector, minimum_number_of_points_per_image)

    for l, r, device, calib_obj in zip(left_images, right_images, device_tracking, obj_tracking):
        successful = calibrator.grab_data( l, r, device, calib_obj)
        assert successful > 0

    # Calibrate & handeye
    reproj_err_1, recon_err_1, params_1 = calibrator.calibrate()
    calibrator.handeye_calibration()

    print(calibrator.handeye_matrix)
    print(calibrator.pattern2marker_matrix)
    print(calibrator.calibration_params.l2r_rmat)
    print(calibrator.calibration_params.l2r_tvec)

    # stereo_calib.load_data(tracking_data_dir, file_prefix)
    # print(stereo_calib.video_data.get_number_of_views())
    # print(stereo_calib.tracking_data.get_number_of_views())


    
    # num_frames = len(stereo_calib.tracking_data.calibration_tracking_array)
    # for i in range(num_frames):
    #     print(f'Frame {i} of {num_frames}')
    #     l_image = stereo_calib.video_data.left_data.images_array[i]
    #     r_image = stereo_calib.video_data.right_data.images_array[i]
    #     device_tracking = stereo_calib.tracking_data.device_tracking_array[i]
    #     calibration_tracking = stereo_calib.tracking_data.calibration_tracking_array[i]

    #     stereo_calib.grab_data(l_image, r_image, device_tracking, calibration_tracking)

    # print(stereo_calib.video_data.get_number_of_views())
    # print(stereo_calib.tracking_data.get_number_of_views())
    # stereo_calib.calibrate()
    # stereo_calib.handeye_calibration()

@pytest.mark.skip()
def test_calibration_regression_test():
    configuration_manager = config.ConfigurationManager('tests/data/config_offline_test_data.json')
    configuration_data = configuration_manager.get_copy()

    data_dir = "tests/data/2020_01_20_stortz/12_50_30"

    calibration_manager = cm.CalibrationManager(configuration_data,
                                                data_dir=data_dir)
    calibration_manager.calibrate()

    left_handeye_old = np.array([75.2,  189.8, -606.1])
    left_handeye_new = calibration_manager.left_params.t_handeye
    assert(np.linalg.norm(left_handeye_new - left_handeye_old) < 1)

# Lots of other tests.
