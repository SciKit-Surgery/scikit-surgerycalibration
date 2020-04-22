"""Tests for command line application """
from sksurgerycalibration.ui.pivot_calibration_command_line import main

def test_cl_no_config():
    """ Run command line app with no config file """
    main(['-i', 'tests/data/PivotCalibration/'])


def test_cl_ransac_config():
    """ Run command line app with ransac in config """
    main(['-i', 'tests/data/PivotCalibration/', '-c',
          'config/ransac_conf.json'])
