"""Tests for command line application """
import copy
import pytest
from sksurgerycalibration.ui.video_calibration_checker_app import \
                run_video_calibration_checker

config = { "method": "chessboard",
    "source": "tests/data/laparoscope_calibration/left/left.ogv",
    "corners": [14, 10],
    "square size in mm": 6,
    "minimum number of views": 5,
    "keypress delay": 0,
    "interactive" : False,
    "sample frequency" : 2
}

def test_with_no_config():
    """It shouldn't run with no configuration file"""
    with pytest.raises(ValueError):
        run_video_calibration_checker(None,
                calib_dir = 'tests/data/laparoscope_calibration/cbh-viking',
                prefix = "calib.right")


def test_with_prefix():
    """ Run command line app with an existing calibration"""
    run_video_calibration_checker(config,
                calib_dir = 'tests/data/laparoscope_calibration/cbh-viking',
                prefix = "calib.right")


def test_with_invalid_capture():
    """Should throw a runtime error if we can't open video capture"""
    duff_config = copy.deepcopy(config)
    duff_config['source'] = 'bad source'
    with pytest.raises(RuntimeError):
        run_video_calibration_checker(duff_config,
                calib_dir = 'tests/data/laparoscope_calibration/cbh-viking',
                prefix = "calib.right")


def test_with_custome_window_size():
    """We should be able to set the window size in config"""
    ok_config = copy.deepcopy(config)
    ok_config['window size'] = [640, 480]
    run_video_calibration_checker(ok_config,
                calib_dir = 'tests/data/laparoscope_calibration/cbh-viking',
                prefix = "calib.right")
