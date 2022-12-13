#  -*- coding: utf-8 -*-
"""Tests for sksrurgerycalibration triangulate"""

import numpy as np
import pytest
import sksurgerycalibration.algorithms.triangulate as sat


def load_chessboard_arrays():
    """
    Load array of data for "Chessboard Test"
    """

    points_in_2d = np.zeros((4, 4), dtype=np.double)
    points_in_2d[0, 0] = 1100.16
    points_in_2d[0, 1] = 262.974
    points_in_2d[0, 2] = 1184.84
    points_in_2d[0, 3] = 241.915
    points_in_2d[1, 0] = 1757.74
    points_in_2d[1, 1] = 228.971
    points_in_2d[1, 2] = 1843.52
    points_in_2d[1, 3] = 204.083
    points_in_2d[2, 0] = 1065.44
    points_in_2d[2, 1] = 651.593
    points_in_2d[2, 2] = 1142.75
    points_in_2d[2, 3] = 632.817
    points_in_2d[3, 0] = 1788.22
    points_in_2d[3, 1] = 650.41
    points_in_2d[3, 2] = 1867.78
    points_in_2d[3, 3] = 632.59

    left_undistorted = np.zeros((4, 2), dtype=np.double)
    left_undistorted[0, 0] = points_in_2d[0, 0]
    left_undistorted[0, 1] = points_in_2d[0, 1]
    left_undistorted[1, 0] = points_in_2d[1, 0]
    left_undistorted[1, 1] = points_in_2d[1, 1]
    left_undistorted[2, 0] = points_in_2d[2, 0]
    left_undistorted[2, 1] = points_in_2d[2, 1]
    left_undistorted[3, 0] = points_in_2d[3, 0]
    left_undistorted[3, 1] = points_in_2d[3, 1]

    right_undistorted = np.zeros((4, 2), dtype=np.double)
    right_undistorted[0, 0] = points_in_2d[0, 2]
    right_undistorted[0, 1] = points_in_2d[0, 3]
    right_undistorted[1, 0] = points_in_2d[1, 2]
    right_undistorted[1, 1] = points_in_2d[1, 3]
    right_undistorted[2, 0] = points_in_2d[2, 2]
    right_undistorted[2, 1] = points_in_2d[2, 3]
    right_undistorted[3, 0] = points_in_2d[3, 2]
    right_undistorted[3, 1] = points_in_2d[3, 3]

    left_intrinsic = np.eye(3, dtype=np.double)
    left_intrinsic[0, 0] = 2012.186314
    left_intrinsic[1, 1] = 2017.966019
    left_intrinsic[0, 2] = 944.7173708
    left_intrinsic[1, 2] = 617.1093984

    right_intrinsic = np.eye(3, dtype=np.double)
    right_intrinsic[0, 0] = 2037.233928
    right_intrinsic[1, 1] = 2052.018948
    right_intrinsic[0, 2] = 1051.112809
    right_intrinsic[1, 2] = 548.0675962

    left_to_right_rotation = np.eye(3, dtype=np.double)
    left_to_right_rotation[0, 0] = 0.999678
    left_to_right_rotation[0, 1] = 0.000151
    left_to_right_rotation[0, 2] = 0.025398
    left_to_right_rotation[1, 0] = -0.000720
    left_to_right_rotation[1, 1] = 0.999749
    left_to_right_rotation[1, 2] = 0.022394
    left_to_right_rotation[2, 0] = -0.025388
    left_to_right_rotation[2, 1] = -0.022405
    left_to_right_rotation[2, 2] = 0.999426

    left_to_right_translation = np.zeros((3, 1), dtype=np.double)
    left_to_right_translation[0, 0] = -4.631472
    left_to_right_translation[1, 0] = 0.268695
    left_to_right_translation[2, 0] = 1.300256

    return points_in_2d, \
           left_undistorted, \
           right_undistorted, \
           left_intrinsic, \
           right_intrinsic, \
           left_to_right_rotation, \
           left_to_right_translation


def test_triangulate_points_with_hartley():
    """
    Test triangulate points with hartley using "Chessboard Test"
    """

    points_in_2d, left_undistorted, right_undistorted, left_intrinsic, right_intrinsic, \
    left_to_right_rotation, left_to_right_translation = load_chessboard_arrays()

    pointsFromHartley = sat.triangulate_points_using_hartley(points_in_2d,
                                                             left_intrinsic,
                                                             right_intrinsic,
                                                             left_to_right_rotation,
                                                             left_to_right_translation)


def test_triangulate_points_using_hartley_opencv():
    """
    Test triangulate points with hartley with cv2.triangulatePoints using "Chessboard Test"
    """

    points_in_2d, left_undistorted, right_undistorted, left_intrinsic, right_intrinsic, \
    left_to_right_rotation, left_to_right_translation = load_chessboard_arrays()

    pointsFromHartley_opencv = sat.triangulate_points_using_hartley_opencv(left_undistorted,
                                                                           right_undistorted,
                                                                           left_intrinsic,
                                                                           right_intrinsic,
                                                                           left_to_right_rotation,
                                                                           left_to_right_translation)
