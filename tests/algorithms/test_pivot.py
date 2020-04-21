#  -*- coding: utf-8 -*-
"""Tests for sksrurgerycalibration pivot calibration"""
from glob import glob
import numpy as np
import pytest
import sksurgerycalibration.algorithms.pivot as p


def test_empty_matrices():
    """Throws a type error if empty matrices are None"""

    with pytest.raises(TypeError):
        p.pivot_calibration(None)


def test_rank_lt_six():
    """Throw a value error if matrix rank is less than 6?"""
    with pytest.raises(ValueError):
        file_names = glob('tests/data/PivotCalibration/1378476417807806000.txt')
        arrays = [np.loadtxt(f) for f in file_names]
        matrices = np.concatenate(arrays)
        number_of_matrices = int(matrices.size/16)
        matrices = matrices.reshape((number_of_matrices, 4, 4))
        p.pivot_calibration(matrices)


def test_four_columns_matrices4x4():
    """Throw a value error if matrix is not 4 column"""

    with pytest.raises(ValueError):
        p.pivot_calibration(np.arange(2, 11, dtype=float).reshape(3, 3))


def test_four_rows_matrices4x4():
    """Throw a value error if matrix is not 4 rows"""

    with pytest.raises(ValueError):
        p.pivot_calibration(np.arange(2, 11, dtype=float).reshape(3, 3))


def test_return_value():
    """A regression test using some recorded data"""

    file_names = glob('tests/data/PivotCalibration/*')
    arrays = [np.loadtxt(f) for f in file_names]
    matrices = np.concatenate(arrays)
    number_of_matrices = int(matrices.size/16)
    matrices = matrices.reshape((number_of_matrices, 4, 4))
    x_values, residual_error = p.pivot_calibration(matrices)
    assert round(residual_error, 3) == 1.761
    assert round(x_values[0, 0], 3) == -14.473
    assert round(x_values[1, 0], 3) == 394.634
    assert round(x_values[2, 0], 3) == -7.407
    assert round(x_values[3, 0], 3) == -804.742
    assert round(x_values[4, 0], 3) == -85.474
    assert round(x_values[5, 0], 3) == -2112.131


def test_rank_if_condition():
    """
    This test will be checking a specific if condition.
    But at the moment I dont know what data I need
    To get proper s_values to cover that if condition.
    """
    with pytest.raises(ValueError):
        file_names = glob('tests/data/test_case_data.txt')
        arrays = [np.loadtxt(f) for f in file_names]
        matrices = np.concatenate(arrays)
        number_of_matrices = int(matrices.size/16)
        matrices = matrices.reshape((number_of_matrices, 4, 4))
        p.pivot_calibration(matrices)


def test_pivot_with_ransac():
    """Tests that pivot with ransac runs"""
    file_names = glob('tests/data/PivotCalibration/*')
    arrays = [np.loadtxt(f) for f in file_names]
    matrices = np.concatenate(arrays)
    number_of_matrices = int(matrices.size/16)
    matrices = matrices.reshape((number_of_matrices, 4, 4))
    _, residual_1 = p.pivot_calibration(matrices)
    _, residual_2 = p.pivot_calibration_with_ransac(matrices, 10, 4, 0.25)
    assert residual_2 < residual_1
    _, _ = p.pivot_calibration_with_ransac(matrices,
                                           10, 4, 0.25,
                                           early_exit=True)
