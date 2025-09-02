#  -*- coding: utf-8 -*-

""" Functions for Triangulation using Harley method """

import cv2
import numpy as np
import sksurgerycore.transforms.matrix as stm


# pylint:disable=too-many-positional-arguments
def _triangulate_point_using_svd(p1_array,
                                 p2_array,
                                 u1_array,
                                 u2_array,
                                 w1_const,
                                 w2_const):
    """
    Function for Internal Triangulate Point Using SVD

    (u1_array, p1_array) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2_array, p2_array) is the second pair.

    :param p1_array: [3x4] ndarray
    :param p2_array: [3x4] ndarray
    :param u1_array: [3x1] ndarray
    :param u2_array: [3x1] ndarray
    :param w1_const: constant value
    :param w2_const: constant value

    :return x_array:  [4x1] ndarray
    """

    # Build matrix A for homogeneous equation system Ax = 0
    # Assume X = (x,y,z,1), for Linear-LS method
    # Which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    a_array = np.zeros((4, 3), dtype=np.double)
    a_array[0, 0] = (u1_array[0][0] * p1_array[2, 0] - p1_array[0, 0]) / w1_const
    a_array[0, 1] = (u1_array[0][0] * p1_array[2, 1] - p1_array[0, 1]) / w1_const
    a_array[0, 2] = (u1_array[0][0] * p1_array[2, 2] - p1_array[0, 2]) / w1_const
    a_array[1, 0] = (u1_array[1][0] * p1_array[2, 0] - p1_array[1, 0]) / w1_const
    a_array[1, 1] = (u1_array[1][0] * p1_array[2, 1] - p1_array[1, 1]) / w1_const
    a_array[1, 2] = (u1_array[1][0] * p1_array[2, 2] - p1_array[1, 2]) / w1_const
    a_array[2, 0] = (u2_array[0][0] * p2_array[2, 0] - p2_array[0, 0]) / w2_const
    a_array[2, 1] = (u2_array[0][0] * p2_array[2, 1] - p2_array[0, 1]) / w2_const
    a_array[2, 2] = (u2_array[0][0] * p2_array[2, 2] - p2_array[0, 2]) / w2_const
    a_array[3, 0] = (u2_array[1][0] * p2_array[2, 0] - p2_array[1, 0]) / w2_const
    a_array[3, 1] = (u2_array[1][0] * p2_array[2, 1] - p2_array[1, 1]) / w2_const
    a_array[3, 2] = (u2_array[1][0] * p2_array[2, 2] - p2_array[1, 2]) / w2_const

    b_array = np.zeros((4, 1), dtype=np.double)
    b_array[0][0] = -(u1_array[0][0] * p1_array[2, 3] - p1_array[0, 3]) / w1_const
    b_array[1][0] = -(u1_array[1][0] * p1_array[2, 3] - p1_array[1, 3]) / w1_const
    b_array[2][0] = -(u2_array[0][0] * p2_array[2, 3] - p2_array[0, 3]) / w2_const
    b_array[3][0] = -(u2_array[1][0] * p2_array[2, 3] - p2_array[1, 3]) / w2_const

    # start = time.time_ns()
    x_array = cv2.solve(a_array, b_array, flags=cv2.DECOMP_SVD)
    # x_array, _, _, _ = np.linalg.lstsq(a_array, b_array, rcond=-1) #Alternatively
    # print(f'{ time.time_ns() - start } nsecs')

    return x_array[1]  # for cv2.solve #x_array#for np.linalg.lstsq


def _iter_triangulate_point_w_svd(p1_array,
                                  p2_array,
                                  u1_array,
                                  u2_array):
    """
    Function for Iterative Triangulate Point Using SVD

    (u1_array, p1_array) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2_array, p2_array) is the second pair.

    :param p1_array: [3x4] ndarray
    :param p2_array: [3x4] ndarray
    :param u1_array: [3x1] ndarray
    :param u2_array: [3x1] ndarray

    :return result:
    """

    epsilon = 0.00000000001
    w1_const = 1
    w2_const = 1
    x_array = np.zeros((4, 1), dtype=np.double)
    result = np.zeros((3, 1), dtype=np.double)

    # Hartley suggests 10 iterations at most
    for dummy_idx in range(10):
        x_array_ = _triangulate_point_using_svd(p1_array, p2_array, u1_array, u2_array, w1_const, w2_const)
        x_array[0][0] = x_array_[0][0]
        x_array[1][0] = x_array_[1][0]
        x_array[2][0] = x_array_[2][0]
        x_array[3][0] = 1.0

        p2x1 = p1_array[2, :].dot(x_array)
        p2x2 = p2_array[2, :].dot(x_array)
        # p2x1 and p2x2 should not be zero to avoid RuntimeWarning: invalid value encountered in true_divide
        if (abs(w1_const - p2x1) <= epsilon and abs(w2_const - p2x2) <= epsilon):
            break

        w1_const = p2x1[0]
        w2_const = p2x2[0]

    result[0][0] = x_array[0][0]
    result[1][0] = x_array[1][0]
    result[2][0] = x_array[2][0]

    return result


def new_e2_array(e2_array, r2_array, left_to_right_trans_vector):
    """

    Function to create a new_e2_array which concatenate left_to_right_trans_vector into the last column of e2_array.
    Notes. new_e2_array() is used in triangulate_points_hartley() to avoid too many variables in one method (see R0914)

    :param e2_array: [4x4] narray
    :param r2_array: = left_to_right_rotation_matrix: [3x3] narray
    :param left_to_right_trans_vector: [3x1] narray

    :return e2_array: [4x4] narray
    """
    # While the specification is [3x1] which implies ndarray, the users
    # may also pass in array (3,), ndarray (3, 1) or ndarray (1, 3).
    # So, this will flatten all of them to the same array-like shape.
    l_r = np.ravel(left_to_right_trans_vector)

    for row_idx in range(0, 3):
        for col_idx in range(0, 3):
            e2_array[row_idx, col_idx] = r2_array[row_idx, col_idx]
        e2_array[row_idx, 3] = l_r[row_idx]

    return e2_array


def l2r_to_p2d(p2d, l2r):
    """

    Function to convert l2r array to p2d array, which removes last row of l2r to create p2d.
    Notes. l2r_to_p2d() is used in triangulate_points_hartley() to avoid too many variables in one method (see R0914).

    :param p2d: [3x4] narray
    :param l2r: [4x4] narray

    :return p2d: [3x4] narray
    """

    for dummy_row_index in range(0, 3):
        for dummy_col_index in range(0, 4):
            p2d[dummy_row_index, dummy_col_index] = l2r[dummy_row_index, dummy_col_index]

    return p2d


def triangulate_points_hartley(input_undistorted_points,
                               left_camera_intrinsic_params,
                               right_camera_intrinsic_params,
                               left_to_right_rotation_matrix,
                               left_to_right_trans_vector
                               ):
    """
    Function to compute triangulation of points using Harley

    :param input_undistorted_points: [4x4] narray
    :param left_camera_intrinsic_params: [3x3] narray
    :param right_camera_intrinsic_params: [3x3] narray
    :param left_to_right_rotation_matrix: [3x3] narray
    :param left_to_right_trans_vector: [3x1] narray

    :return outputPoints: [4x3] narray

    References
    ----------
    * Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.
    * Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.

    :return outputPoints:
    """
    number_of_points = input_undistorted_points.shape[0]
    output_points = np.zeros((number_of_points, 3), dtype=np.double)
    k1_array = left_camera_intrinsic_params
    k2_array = right_camera_intrinsic_params
    _r1_array = np.eye(3, dtype=np.double)  # (unused-variable)
    r2_array = left_to_right_rotation_matrix
    e1_array = np.eye(4, dtype=np.double)
    e2_array = np.eye(4, dtype=np.double)
    p1d = np.zeros((3, 4), dtype=np.double)
    p2d = np.zeros((3, 4), dtype=np.double)

    e2_array = new_e2_array(e2_array, r2_array, left_to_right_trans_vector)
    # Inverting intrinsic params to convert from pixels to normalised image coordinates.
    k1inv = np.linalg.inv(k1_array)
    k2inv = np.linalg.inv(k2_array)

    # Computing coordinates relative to left camera.
    e1inv = np.linalg.inv(e1_array)
    l2r = np.matmul(e2_array, e1inv)

    # The projection matrix, is just the extrinsic parameters, as our coordinates will be in a normalised camera space.
    # P1 should be identity, so that reconstructed coordinates are in Left Camera Space, to P2 should reflect
    # a right to left transform.
    # Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.
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

    p2d = l2r_to_p2d(p2d, l2r)

    u1_array = np.zeros((3, 1), dtype=np.double)
    u2_array = np.zeros((3, 1), dtype=np.double)

    for dummy_index in range(0, number_of_points):
        u1_array[0, 0] = input_undistorted_points[dummy_index, 0]
        u1_array[1, 0] = input_undistorted_points[dummy_index, 1]
        u1_array[2, 0] = 1

        u2_array[0, 0] = input_undistorted_points[dummy_index, 2]
        u2_array[1, 0] = input_undistorted_points[dummy_index, 3]
        u2_array[2, 0] = 1

        # Converting to normalised image points
        u1t = np.matmul(k1inv, u1_array)
        u2t = np.matmul(k2inv, u2_array)

        # array shapes for input args _iter_triangulate_point_w_svd( [3, 4]; [3, 4]; [3, 1]; [3, 1] )
        reconstructed_point = _iter_triangulate_point_w_svd(p1d, p2d, u1t, u2t)

        output_points[dummy_index, 0] = reconstructed_point[0][0]
        output_points[dummy_index, 1] = reconstructed_point[1][0]
        output_points[dummy_index, 2] = reconstructed_point[2][0]

    return output_points


def triangulate_points_opencv(input_undistorted_points,
                              left_camera_intrinsic_params,
                              right_camera_intrinsic_params,
                              left_to_right_rotation_matrix,
                              left_to_right_trans_vector):
    """
    Function to compute triangulation of points using Harley with cv2.triangulatePoints

    :param input_undistorted_points: [4x4] narray

    :param left_camera_intrinsic_params: [3x3] narray
    :param right_camera_intrinsic_params: [3x3] narray
    :param left_to_right_rotation_matrix: [3x3] narray
    :param left_to_right_trans_vector: [3x1] narray

    :return output_points: [4x3] narray

    Other related variables:
        left_undistorted, right_undistorted: point image positions in 2 cameras
        left_undistorted[4x2], right_undistorted[4x2] from input_undistorted_points [4x4]

    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.

    """

    l2r_mat = stm.construct_rigid_transformation(left_to_right_rotation_matrix, left_to_right_trans_vector)

    # The projection matrix, is just the extrinsic parameters, as our coordinates will be in a normalised camera space.
    # P1 should be identity, so that reconstructed coordinates are in Left Camera Space, to P2 should reflect
    # a right to left transform.
    # Prince, Simon JD. Computer vision: models, learning, and inference. Cambridge University Press, 2012.
    p1mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.double)

    p2mat = np.zeros((3, 4), dtype=np.double)
    p2mat = l2r_to_p2d(p2mat, l2r_mat)

    number_of_points = input_undistorted_points.shape[0]
    output_points = np.zeros((number_of_points, 3), dtype=np.double)

    # Inverting intrinsic params to convert from pixels to normalised image coordinates.
    k1inv = np.linalg.inv(left_camera_intrinsic_params)
    k2inv = np.linalg.inv(right_camera_intrinsic_params)

    u1_array = np.zeros((3, 1), dtype=np.double)
    u2_array = np.zeros((3, 1), dtype=np.double)

    for dummy_index in range(0, number_of_points):
        u1_array[0, 0] = input_undistorted_points[dummy_index, 0]
        u1_array[1, 0] = input_undistorted_points[dummy_index, 1]
        u1_array[2, 0] = 1

        u2_array[0, 0] = input_undistorted_points[dummy_index, 2]
        u2_array[1, 0] = input_undistorted_points[dummy_index, 3]
        u2_array[2, 0] = 1

        # Converting to normalised image points
        u1t = np.matmul(k1inv, u1_array)
        u2t = np.matmul(k2inv, u2_array)

        # array shapes for input args cv2.triangulatePoints( [3, 4]; [3, 4]; [2, 1]; [2, 1] )
        reconstructed_point = cv2.triangulatePoints(p1mat, p2mat, u1t[:2], u2t[:2])
        reconstructed_point /= reconstructed_point[3][0]  # Homogenize

        output_points[dummy_index, 0] = reconstructed_point[0][0]
        output_points[dummy_index, 1] = reconstructed_point[1][0]
        output_points[dummy_index, 2] = reconstructed_point[2][0]

    return output_points
