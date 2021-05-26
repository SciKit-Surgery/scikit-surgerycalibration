"""Tests for command line application """
import copy
import pytest
from sksurgerycalibration.ui.video_calibration_app import run_video_calibration

config = { "method": "chessboard",
    "source": "tests/data/laparoscope_calibration/left/left.ogv",
    "corners": [14, 10],
    "square size in mm": 6,
    "minimum number of views": 5,
    "keypress delay": 0,
    "interactive" : False,
    "sample frequency" : 2
}


def test_with_save_prefix():
    """ Run command line app with a save prefix"""
    run_video_calibration(config, prefix = "testjunk")

def test_with_save_directory():
    """ Run command line app with a save prefix"""
    run_video_calibration(config, save_dir = "testjunk")

def test_with_invalid_method():
    """Should throw a value error if method is not supported"""
    duff_config = copy.deepcopy(config)
    duff_config['method'] = 'not chessboard'
    with pytest.raises(ValueError):
        run_video_calibration(duff_config)

def test_with_invalid_capture():
    """Should throw a runtime error if we can't open video capture"""
    duff_config = copy.deepcopy(config)
    duff_config['source'] = 'bad source'
    with pytest.raises(RuntimeError):
        run_video_calibration(duff_config)

def test_with_custome_window_size():
    """We should be able to set the window size in config"""
    ok_config = copy.deepcopy(config)
    ok_config['window size'] = [640, 480]
    run_video_calibration(ok_config)
