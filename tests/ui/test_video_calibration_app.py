"""Tests for command line application """
import copy
import os
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

def _clean_up(prefix):
    """Helper to clean up calibration results"""
    for i in range(5):
        os.remove(prefix + ".extrinsics." + str(i) + ".txt")
        os.remove(prefix + ".ids." + str(i) + ".txt")
        os.remove(prefix + ".image_points." + str(i) + ".txt")
        os.remove(prefix + ".object_points." + str(i) + ".txt")
        os.remove(prefix + ".images." + str(i) + ".png")
    os.remove(prefix + ".distortion.txt")
    os.remove(prefix + ".handeye.txt")
    os.remove(prefix + ".intrinsics.txt")
    os.remove(prefix + ".pattern2marker.txt")


def test_with_save_prefix():
    """ Run command line app with a save prefix"""
    run_video_calibration(config, prefix = "testjunk")
    _clean_up("testjunk")

def test_with_save_directory():
    """ Run command line app with a save prefix"""
    run_video_calibration(config, save_dir = "testjunk")
    _clean_up("testjunk/calib")
    os.rmdir("testjunk")

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
