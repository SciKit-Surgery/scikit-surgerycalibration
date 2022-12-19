#  -*- coding: utf-8 -*-
"""Tests for sksrurgerycalibration pivot calibration"""
from glob import glob
from random import seed
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
        p.pivot_calibration_aos(matrices)


def test_unkown_method():
    """Throw value error if config set for unknown method"""
    config = {"method" : "not implemented"}
    matrix = (np.arange(2, 18, dtype=float).reshape((1, 4, 4)))
    with pytest.raises(ValueError):
        p.pivot_calibration(matrix, config)

def test_four_columns_matrices4x4():
    """Throw a value error if matrix is not 4 column"""

    with pytest.raises(ValueError):
        p.pivot_calibration(np.arange(2, 14, dtype=float).reshape((1, 4, 3)))


def test_four_rows_matrices4x4():
    """Throw a value error if matrix is not 4 rows"""

    with pytest.raises(ValueError):
        p.pivot_calibration(np.arange(2, 14, dtype=float).reshape((1, 3, 4)))


def test_return_value():
    """A regression test using some recorded data"""

    file_names = glob('tests/data/PivotCalibration/*')
    arrays = [np.loadtxt(f) for f in file_names]
    matrices = np.concatenate(arrays)
    number_of_matrices = int(matrices.size/16)
    matrices = matrices.reshape((number_of_matrices, 4, 4))
    config = {"method" : "aos"}
    pointer_offset, pivot_point, residual_error = \
        p.pivot_calibration(matrices, config)
    assert round(residual_error, 3) == 1.761
    assert round(pointer_offset[0, 0], 3) == -14.473
    assert round(pointer_offset[1, 0], 3) == 394.634
    assert round(pointer_offset[2, 0], 3) == -7.407
    assert round(pivot_point[0, 0], 3) == -804.742
    assert round(pivot_point[1, 0], 3) == -85.474
    assert round(pivot_point[2, 0], 3) == -2112.131


def test_pivot_with_ransac():
    """Tests that pivot with ransac runs"""
    #seed the random number generator. Seeding
    #with 0 leads to one failed pivot calibration (rank < 6), so we
    #hit lines 127-129
    seed(0)

    file_names = glob('tests/data/PivotCalibration/*')
    arrays = [np.loadtxt(f) for f in file_names]
    matrices = np.concatenate(arrays)
    number_of_matrices = int(matrices.size/16)
    matrices = matrices.reshape((number_of_matrices, 4, 4))
    _, _, residual_1 = p.pivot_calibration(matrices)
    _, _, residual_2 = p.pivot_calibration_with_ransac(matrices, 10, 4, 0.25)
    assert residual_2 < residual_1
    _, _, _ = p.pivot_calibration_with_ransac(matrices,
                                              10, 4, 0.25,
                                              early_exit=True)
    #tests for the value checkers at the start of RANSAC
    with pytest.raises(ValueError):
        _, _, _ = p.pivot_calibration_with_ransac(None, 0, None, None)

    with pytest.raises(ValueError):
        _, _, _ = p.pivot_calibration_with_ransac(None, 2, -1.0, None)

    with pytest.raises(ValueError):
        _, _, _ = p.pivot_calibration_with_ransac(None, 2, 1.0, 1.1)

    with pytest.raises(TypeError):
        _, _, _ = p.pivot_calibration_with_ransac(None, 2, 1.0, 0.8)

    #with consensus threshold set to 1.0, we get a value error
    #as no best model is found.
    with pytest.raises(ValueError):
        _, _, _ = p.pivot_calibration_with_ransac(matrices,
                                                  10, 4, 1.0,
                                                  early_exit=True)


def test_pivot_with_sphere_fit():
    """Tests pivot calibration with sphere fitting"""
    config = {"method" : "sphere_fitting"}
    file_names = glob('tests/data/PivotCalibration/*')
    arrays = [np.loadtxt(f) for f in file_names]
    matrices = np.concatenate(arrays)
    number_of_matrices = int(matrices.size/16)
    matrices = matrices.reshape((number_of_matrices, 4, 4))
    _, _, residual_error = p.pivot_calibration(matrices, config)

    #do a regression test on the residual error
    assert round(residual_error, 3) == 2.346


def test_replace_small_values():
    """Tests for small values replacement"""
    list_in = [0.2, 0.6, 0.0099, 0.56]

    rank = p._replace_small_values(list_in) #pylint: disable=protected-access

    assert rank == 3
    assert list_in[2] == 0

    rank = p._replace_small_values( #pylint: disable=protected-access
        list_in,
        threshold=0.3,
        replacement_value=-1.0)

    assert rank == 2
    assert list_in[0] == -1.0
    assert list_in[2] == -1.0


def _rms(*values):
    """
    Works out RMS error, so we can test against it.
    """
    count = len(values)
    square_sum = 0.0
    for value in values:
        square_sum += value * value

    mean_square_sum = square_sum / count
    return np.sqrt(mean_square_sum)

def test_residual_error0():
    """
    Test that residual error returns a correct value
    """

    pivot_point = [0.0, 0.0, 0.0]
    pointer_offset = [-100.0, 0.0, 0.0]

    tracker_mat = np.array([[[1.0, 0.0, 0.0, 110.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]]
                           ])
    #error is 10.0 in one direction and zero in the others
    expected_value = _rms(10.0, 0.0, 0.0)
    residual_error = p._residual_error( #pylint: disable=protected-access
        tracker_mat,
        pointer_offset, pivot_point)
    assert residual_error == expected_value


def test_residual_error1():
    """
    Test that residual error returns a correct value
    """

    pivot_point = [0.0, 0.0, 0.0]
    pointer_offset = [-100.0, 0.0, 0.0]

    tracker_mat = np.array([[[1.0, 0.0, 0.0, 100.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]]
                           ])

    #error in all three directions is zero
    expected_value = _rms(0.0, 0.0, 0.0)
    residual_error = p._residual_error( #pylint: disable=protected-access
        tracker_mat,
        pointer_offset, pivot_point)
    assert residual_error == expected_value


def test_residual_error2():
    """
    Test that residual error returns a correct value
    """

    pivot_point = [0.0, 0.0, 0.0]
    pointer_offset = [-100.0, 0.0, 0.0]


    tracker_mat = np.array([[[1.0, 0.0, 0.0, 110.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]],

                            [[1.0, 0.0, 0.0, 90.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]],
                           ])

    #error in all 2 of 6 directions is 10.0
    expected_value = _rms(10.0, 0.0, 0.0, -10.0, 0.0, 0.0)
    residual_error = p._residual_error( #pylint: disable=protected-access
        tracker_mat,
        pointer_offset, pivot_point)
    assert residual_error == expected_value


def test_residual_error3():
    """
    Test that residual error returns a correct value
    """

    pivot_point = [0.0, 0.0, 0.0]
    pointer_offset = [-100.0, 0.0, 0.0]

    tracker_mat = np.array([[[1.0, 0.0, 0.0, 110.0],
                             [0.0, 1.0, 0.0, 10.0],
                             [0.0, 0.0, 1.0, -10.0],
                             [0.0, 0.0, 0.0, 1.0]]
                           ])

    #error in all three directions is 10.0
    expected_value = _rms(10.0, 10.0, -10.0)
    residual_error = p._residual_error( #pylint: disable=protected-access
        tracker_mat,
        pointer_offset, pivot_point)
    assert residual_error == expected_value
