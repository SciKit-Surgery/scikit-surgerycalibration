# -*- coding: utf-8 -*-

"""
Tests for calibration_manager.
"""

import pytest
import numpy as np
import sksurgeryimage.calibration.chessboard_point_detector as pd
import sksurgerycalibration.video as vidcal

def test_load_tracking_data():
    
    tracking_data = vidcal.video_calibration_data.TrackingData()
    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib.tracking'

    tracking_data.load_data(tracking_data_dir, file_prefix)
    assert(len(tracking_data.device_tracking_array) == 10)
    assert(len(tracking_data.calibration_tracking_array) == 10)

def test_load_data_stereo_calib():
    
    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))

    stereo_calib = vidcal.video_calibration_driver_stereo.StereoVideoCalibrationDriver(chessboard_detector, 140)

    tracking_data_dir = 'tests/data/2020_01_20_storz/12_50_30'
    file_prefix = 'calib'

    stereo_calib.load_data(tracking_data_dir, file_prefix)

    assert(len(stereo_calib.tracking_data.device_tracking_array) == 10)
    assert(len(stereo_calib.tracking_data.calibration_tracking_array) == 10)

    assert(len(stereo_calib.video_data.left_data.images_array) == 10)
    assert(len(stereo_calib.video_data.right_data.images_array) == 10)

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
