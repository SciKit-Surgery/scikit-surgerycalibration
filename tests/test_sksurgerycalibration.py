# coding=utf-8

"""scikit-surgerycalibration tests"""

from sksurgerycalibration.ui.sksurgerycalibration_demo import run_demo
from sksurgerycalibration.algorithms import addition, multiplication

# Pytest style

def test_sksurgerycalibration():
    """Test that the demo works"""
    var_x = 1
    var_y = 2
    verbose = False
    multiply = False

    expected_answer = 3
    assert run_demo(var_x, var_y, multiply, verbose) == expected_answer


def test_addition():
    """addition returns right value"""

    assert addition.add_two_numbers(1, 2) == 3


def test_multiplication():
    """multiplication returns right value"""

    assert multiplication.multiply_two_numbers(2, 2) == 4
