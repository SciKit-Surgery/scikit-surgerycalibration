# -*- coding: utf-8 -*-

"""
Tests for calibration_manager.
"""

import numpy as np
import smartliver.calibration.calibration_manager as cm
import sksurgerycore.configuration.configuration_manager as config


def test_calibration_regression_test():
    configuration_manager = config.ConfigurationManager('tests/data/config_offline_test_data.json')
    configuration_data = configuration_manager.get_copy()

    data_dir = "tests/data/calibration/evaluation_data_viking"

    calibration_manager = cm.CalibrationManager(configuration_data,
                                                data_dir=data_dir)
    calibration_manager.calibrate()

    left_handeye_old = np.array([75.2,  189.8, -606.1])
    left_handeye_new = calibration_manager.left_params.t_handeye
    assert(np.linalg.norm(left_handeye_new - left_handeye_old) < 1)


# Lots of other tests.
