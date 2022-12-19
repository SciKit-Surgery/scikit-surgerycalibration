#  -*- coding: utf-8 -*-

""" Functions for Triangulation using Harley method """

import cv2
import numpy as np
import sksurgerycore.transforms.matrix as stm


def _triangulate_point_using_svd(p1_array,
                                 p2_array,
                                 u1_array,
                                 u2_array,
                                 w1_const,
                                 w2_const):
    """
    Function for Internal Triangulate Point Using SVD
    :param p1_array: [3x4] ndarray
    :param p2_array: [3x4] ndarray
    :param u1_array: [3x1] ndarray
    :param u2_array: [3x1] ndarray
    :param w1_const: constant value
    :param w2_const: constant value

    :return X:
    """

    p1_array = np.zeros((3, 4, 1), dtype=np.double)
    p2_array = np.zeros((3, 4, 1), dtype=np.double)
    u1_array = np.zeros((3, 1, 1), dtype=np.double)
    u2_array = np.zeros((3, 1, 1), dtype=np.double)

    # Build matrix A for homogeneous equation system Ax = 0
    # Assume X = (x,y,z,1), for Linear-LS method
    # Which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1

    a_array = np.zeros((4, 3, 1), dtype=np.double)
    a_array[0, 0] = (u1_array[0] * p1_array[2, 0] - p1_array[0, 0]) / w1_const
    a_array[0, 1] = (u1_array[0] * p1_array[2, 1] - p1_array[0, 1]) / w1_const
    a_array[0, 2] = (u1_array[0] * p1_array[2, 2] - p1_array[0, 2]) / w1_const
    a_array[1, 0] = (u1_array[1] * p1_array[2, 0] - p1_array[1, 0]) / w1_const
    a_array[1, 1] = (u1_array[1] * p1_array[2, 1] - p1_array[1, 1]) / w1_const
    a_array[1, 2] = (u1_array[1] * p1_array[2, 2] - p1_array[1, 2]) / w1_const
    a_array[2, 0] = (u2_array[0] * p2_array[2, 0] - p2_array[0, 0]) / w2_const
    a_array[2, 1] = (u2_array[0] * p2_array[2, 1] - p2_array[0, 1]) / w2_const
    a_array[2, 2] = (u2_array[0] * p2_array[2, 2] - p2_array[0, 2]) / w2_const
    a_array[3, 0] = (u2_array[1] * p2_array[2, 0] - p2_array[1, 0]) / w2_const
    a_array[3, 1] = (u2_array[1] * p2_array[2, 1] - p2_array[1, 1]) / w2_const
    a_array[3, 2] = (u2_array[1] * p2_array[2, 2] - p2_array[1, 2]) / w2_const

    b_array = np.zeros((4, 1, 1), dtype=np.double)
    b_array[0] = -(u1_array[0] * p1_array[2, 3] - p1_array[0, 3]) / w1_const
    b_array[1] = -(u1_array[1] * p1_array[2, 3] - p1_array[1, 3]) / w1_const
    b_array[2] = -(u2_array[0] * p2_array[2, 3] - p2_array[0, 3]) / w2_const
    b_array[3] = -(u2_array[1] * p2_array[2, 3] - p2_array[1, 3]) / w2_const

    x_array = np.zeros((4, 1, 1), dtype=np.double)
    cv2.solve(a_array, b_array, x_array, flags=cv2.DECOMP_SVD)

    return x_array


def _iter_triangulate_point_w_svd(p1_array,
                                           p2_array,
                                           u1_array,
                                           u2_array):
    """
    Function for Iterative Triangulate Point Using SVD

    :param p1_array: [3x4] ndarray
    :param p2_array: [3x4] ndarray
    :param u1_array: [3x1] ndarray
    :param u2_array: [3x1] ndarray

    :return result:
    """

    p1_array = np.zeros((3, 4), dtype=np.double)
    p2_array = np.zeros((3, 4), dtype=np.double)
    u1_array = np.zeros((3, 1), dtype=np.double)
    u2_array = np.zeros((3, 1), dtype=np.double)

    epsilon = 0.00000000001
    w1_const = 1
    w2_const = 1
    x_array = np.zeros((4, 1), dtype=np.double)

    # Hartley suggests 10 iterations at most
    for dummy_index in range(0, 10):
        x_array_ = _triangulate_point_using_svd(p1_array, p2_array, u1_array, u2_array, w1_const, w2_const)
        x_array[0] = x_array_[0]
        x_array[1] = x_array_[1]
        x_array[2] = x_array_[2]
        x_array[3] = 1.0

        p2x1 = (p1_array[2][:].dot(x_array))[0]
        p2x2 = (p2_array[2][:].dot(x_array))[0]
        # p2x1 and p2x2 should not be zero to avoid RuntimeWarning: invalid value encountered in true_divide

        if (abs(w1_const - p2x1) <= epsilon and abs(w2_const - p2x2) <= epsilon):
            break

        w1_const = p2x1
        w2_const = p2x2

    result = np.zeros((3, 1), dtype=np.double)

    result[0] = x_array[0]
    result[1] = x_array[1]
    result[2] = x_array[2]

    return result


def triangulate_points_hartley(input_undistorted_points,
                                     left_camera_intrinsic_params,
                                     right_camera_intrinsic_params,
                                     left_to_right_rotation_matrix,
                                     left_to_right_trans_vector
                                     ):
    """
    Function to compute triangulation of points using Harley

    :param input_undistorted_points:
    :param left_camera_intrinsic_params:
    :param right_camera_intrinsic_params:
    :param left_to_right_rotation_matrix:
    :param left_to_right_trans_vector:

    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.


    :return outputPoints:
    """
    number_of_points = input_undistorted_points.shape[0]  # >input_undistorted_points.rows
    output_points = np.zeros((number_of_points, 3, 1), dtype=np.double)
    k1_array = np.eye(3, dtype=np.double)
    k2_array = np.eye(3, dtype=np.double)
    k1_array = left_camera_intrinsic_params
    k2_array = right_camera_intrinsic_params
    k1inv = np.zeros((3, 3, 1), dtype=np.double)
    k2inv = np.zeros((3, 3, 1), dtype=np.double)
    #r1_array = np.eye(3, dtype=np.double) #(unused-variable)
    r2_array = left_to_right_rotation_matrix
    e1_array = np.eye(4, dtype=np.double)
    e1inv = np.eye(4, dtype=np.double)
    e2_array = np.eye(4, dtype=np.double)
    l2r = np.zeros((4, 4, 1), dtype=np.double)
    p1d = np.zeros((3, 4, 1), dtype=np.double)  # cv::Matx34d P1d, P2d;?
    p2d = np.zeros((3, 4, 1), dtype=np.double)

    # Construct:
    # E1 = Object to Left Camera = Left Camera Extrinsics.
    # E2 = Object to Right Camera = Right Camera Extrinsics.
    # K1 = Copy of Left Camera intrinsics.
    # K2 = Copy of Right Camera intrinsics.
    # Copy data into cv::Mat data types.
    # Camera calibration routines are 32 bit, as some drawing functions require 32 bit data.
    # These triangulation routines need 64 bit data.
    for row_idx in range(0, 3):
        for col_idx in range(0, 3):
            # K1[r, c] = left_camera_intrinsic_params[r, c] #no-need for python
            # K2[r, c] = right_camera_intrinsic_params[r, c] #no-need for python
            e2_array[row_idx, col_idx] = r2_array[row_idx, col_idx]
        e2_array[row_idx, 3] = left_to_right_trans_vector[row_idx, 0]

    # We invert the intrinsic params, so we can convert from pixels to normalised image coordinates.
    k1inv = np.linalg.inv(k1_array)
    k2inv = np.linalg.inv(k2_array)

    # # We want output coordinates relative to left camera.
    e1inv = np.linalg.inv(e1_array)
    l2r = e2_array * e1inv

    # Reading Prince 2012 Computer Vision, the projection matrix, is just the extrinsic parameters,
    # as our coordinates will be in a normalised camera space. P1 should be identity, so that
    # reconstructed coordinates are in Left Camera Space, to P2 should reflect a right to left transform.
    p1d[0, 0] = 1
    p1d[0, 1] = 0
    p1d[0, 2] = 0
    p1d[0, 3] = 0
    p1d[1, 0] = 0
    p1d[1, 1] = 1
    p1d[1, 2] = 0
    p1d[1, 3] = 0
    p1d[2, 0] = 0
    p1d[2, 1] = 0
    p1d[2, 2] = 1
    p1d[2, 3] = 0

    for i in range(0, 3):
        for j in range(0, 4):
            p2d[i, j] = l2r[i, j]

    # `#pragma omp parallel` for its C++ counterpart
    u1_array = np.zeros((3, 1), dtype=np.double)
    u2_array = np.zeros((3, 1), dtype=np.double)
    u1t = np.zeros((3, 1), dtype=np.double)
    u2t = np.zeros((3, 1), dtype=np.double)

    u1p = np.zeros((3, 1), dtype=np.double)
    # Normalised image coordinates. (i.e. relative to a principal point of zero, and in millimetres not pixels).
    u2p = np.zeros((3, 1), dtype=np.double)
    # reconstructed_point = np.zeros((3, 1, 1), dtype=np.double)  # the output 3D point, in reference frame of left camera.

    # print(f'\n {u1p}')

    # # `pragma omp for` for its C++ counterpart
    for i in range(1, number_of_points):
        u1_array[0, 0] = input_undistorted_points[i, 0]
        u1_array[1, 0] = input_undistorted_points[i, 1]
        u1_array[2, 0] = 1

        u2_array[0, 0] = input_undistorted_points[i, 2]
        u2_array[1, 0] = input_undistorted_points[i, 3]
        u2_array[2, 0] = 1

        # Converting to normalised image points
        u1t = k1inv * u1_array
        u2t = k2inv * u2_array

        u1p[0] = u1t[0, 0]
        u1p[1] = u1t[1, 0]
        u1p[2] = u1t[2, 0]

        u2p[0] = u2t[0, 0]
        u2p[1] = u2t[1, 0]
        u2p[2] = u2t[2, 0]

        reconstructed_point = _iter_triangulate_point_w_svd(p1d, p2d, u1p, u2p)

        output_points[i, 0] = reconstructed_point[0]
        output_points[i, 1] = reconstructed_point[1]
        output_points[i, 2] = reconstructed_point[2]

    return output_points

def triangulate_points_opencv(left_undistorted,
                                            right_undistorted,
                                            left_camera_intrinsic_params,
                                            right_camera_intrinsic_params,
                                            left_to_right_rotation_matrix,
                                            left_to_right_trans_vector):
    """
    Function to compute triangulation of points using Harley with cv2.triangulatePoints
    :param left_undistorted:
    :param right_undistorted:
    :param input_undistorted_points:
    :param left_camera_intrinsic_params:
    :param right_camera_intrinsic_params:
    :param left_to_right_rotation_matrix:
    :param left_to_right_trans_vector:
    p_l,p_r: Left and Right camera projection matrixes
    left_undistorted, right_undistorted: point image positions in 2 cameras
    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.
    :return triangulated_cv:
    """

    l2r_mat = stm.construct_rigid_transformation(left_to_right_rotation_matrix, left_to_right_trans_vector)
    p_l = np.zeros((3, 4))
    p_l[:, :-1] = left_camera_intrinsic_params
    p_r = np.zeros((3, 4))
    p_r[:, :-1] = right_camera_intrinsic_params
    p_l = np.matmul(p_l, np.eye(4))
    p_r = np.matmul(p_r, l2r_mat)

    triangulated_cv = cv2.triangulatePoints(p_l,
                                            p_r,
                                            left_undistorted.T,
                                            right_undistorted.T)

    return triangulated_cv
