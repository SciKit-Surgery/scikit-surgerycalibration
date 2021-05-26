"""Tests for command line application """
import pytest
from sksurgerycalibration.ui.video_calibration_command_line import main

def test_cl_no_config():
    """ Run command line app with no config file. The parser should
    raise SystemExit due to missing required argument"""
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        main([])

    #I'm not sure how useful the next 2 asserts are. We already know it's
    #a SystemExit, if the code value specific to the parser?
    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 2


def test_cl_with_config():
    """ Run command line app with config """
    main(['-c', 'config/recorded_chessboard.json'])
