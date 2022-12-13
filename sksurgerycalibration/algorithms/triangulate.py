#  -*- coding: utf-8 -*-

""" Functions for Triangulation using Harley method """

import cv2
import numpy as np
import sksurgerycore.transforms.matrix as stm


def InternalTriangulatePointUsingSVD(P1,
                                     P2,
                                     u1,
                                     u2,
                                     w1,
                                     w2):
    """
    Function for Internal Triangulate Point Using SVD
    :param P1: [3x4] ndarray
    :param P2: [3x4] ndarray
    :param u1: [3x1] ndarray
    :param u2: [3x1] ndarray
    :param w1: constant value
    :param w2: constant value

    :return X:
    """

    P1 = np.zeros((3, 4, 1), dtype=np.double)
    P2 = np.zeros((3, 4, 1), dtype=np.double)
    u1 = np.zeros((3, 1, 1), dtype=np.double)
    u2 = np.zeros((3, 1, 1), dtype=np.double)

    # Build matrix A for homogeneous equation system Ax = 0
    # Assume X = (x,y,z,1), for Linear-LS method
    # Which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1

    A = np.zeros((4, 3, 1), dtype=np.double)
    A[0, 0] = (u1[0] * P1[2, 0] - P1[0, 0]) / w1
    A[0, 1] = (u1[0] * P1[2, 1] - P1[0, 1]) / w1
    A[0, 2] = (u1[0] * P1[2, 2] - P1[0, 2]) / w1
    A[1, 0] = (u1[1] * P1[2, 0] - P1[1, 0]) / w1
    A[1, 1] = (u1[1] * P1[2, 1] - P1[1, 1]) / w1
    A[1, 2] = (u1[1] * P1[2, 2] - P1[1, 2]) / w1
    A[2, 0] = (u2[0] * P2[2, 0] - P2[0, 0]) / w2
    A[2, 1] = (u2[0] * P2[2, 1] - P2[0, 1]) / w2
    A[2, 2] = (u2[0] * P2[2, 2] - P2[0, 2]) / w2
    A[3, 0] = (u2[1] * P2[2, 0] - P2[1, 0]) / w2
    A[3, 1] = (u2[1] * P2[2, 1] - P2[1, 1]) / w2
    A[3, 2] = (u2[1] * P2[2, 2] - P2[1, 2]) / w2

    B = np.zeros((4, 1, 1), dtype=np.double)
    B[0] = -(u1[0] * P1[2, 3] - P1[0, 3]) / w1
    B[1] = -(u1[1] * P1[2, 3] - P1[1, 3]) / w1
    B[2] = -(u2[0] * P2[2, 3] - P2[0, 3]) / w2
    B[3] = -(u2[1] * P2[2, 3] - P2[1, 3]) / w2

    X = np.zeros((4, 1, 1), dtype=np.double)
    cv2.solve(A, B, X, flags=cv2.DECOMP_SVD)

    return X


def InternalIterativeTriangulatePointUsingSVD(P1,
                                              P2,
                                              u1,
                                              u2):
    """
    Function for Iterative Triangulate Point Using SVD

    :param P1: [3x4] ndarray
    :param P2: [3x4] ndarray
    :param u1: [3x1] ndarray
    :param u2: [3x1] ndarray

    :return result:
    """

    P1 = np.zeros((3, 4), dtype=np.double)
    P2 = np.zeros((3, 4), dtype=np.double)
    u1 = np.zeros((3, 1), dtype=np.double)
    u2 = np.zeros((3, 1), dtype=np.double)

    epsilon = 0.00000000001
    w1 = 1
    w2 = 1
    X = np.zeros((4, 1), dtype=np.double)

    # Hartley suggests 10 iterations at most
    for i in range(0, 10):
        X_ = InternalTriangulatePointUsingSVD(P1, P2, u1, u2, w1, w2)
        X[0] = X_[0]
        X[1] = X_[1]
        X[2] = X_[2]
        X[3] = 1.0

        p2x1 = (P1[2][:].dot(X))[0]
        p2x2 = (P2[2][:].dot(X))[0]
        # p2x1 and p2x2 should not be zero to avoid RuntimeWarning: invalid value encountered in true_divide

        if (abs(w1 - p2x1) <= epsilon and abs(w2 - p2x2) <= epsilon):
            break

        w1 = p2x1
        w2 = p2x2

    result = np.zeros((3, 1), dtype=np.double)

    result[0] = X[0]
    result[1] = X[1]
    result[2] = X[2]

    return result


def triangulate_points_using_hartley(inputUndistortedPoints,
                                     leftCameraIntrinsicParams,
                                     rightCameraIntrinsicParams,
                                     leftToRightRotationMatrix,
                                     leftToRightTranslationVector
                                     ):
    """
    Function to compute triangulation of points using Harley

    :param inputUndistortedPoints:
    :param leftCameraIntrinsicParams:
    :param rightCameraIntrinsicParams:
    :param leftToRightRotationMatrix:
    :param leftToRightTranslationVector:

    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.


    :return outputPoints:
    """
    numberOfPoints = inputUndistortedPoints.shape[0] #>inputUndistortedPoints.rows
    outputPoints = np.zeros((numberOfPoints, 3, 1), dtype=np.double)
    K1 = np.eye(3, dtype=np.double)
    K2 = np.eye(3, dtype=np.double)
    K1 = leftCameraIntrinsicParams
    K2 = rightCameraIntrinsicParams
    K1Inv = np.zeros((3, 3, 1), dtype=np.double)
    K2Inv = np.zeros((3, 3, 1), dtype=np.double)
    R1 = np.eye(3, dtype=np.double)
    R2 = leftToRightRotationMatrix
    E1 = np.eye(4, dtype=np.double)
    E1Inv = np.eye(4, dtype=np.double)
    E2 = np.eye(4, dtype=np.double)
    L2R = np.zeros((4, 4, 1), dtype=np.double)
    P1d = np.zeros((3, 4, 1), dtype=np.double)  # cv::Matx34d P1d, P2d;?
    P2d = np.zeros((3, 4, 1), dtype=np.double)

    # Construct:
    # E1 = Object to Left Camera = Left Camera Extrinsics.
    # E2 = Object to Right Camera = Right Camera Extrinsics.
    # K1 = Copy of Left Camera intrinsics.
    # K2 = Copy of Right Camera intrinsics.
    # Copy data into cv::Mat data types.
    # Camera calibration routines are 32 bit, as some drawing functions require 32 bit data.
    # These triangulation routines need 64 bit data.
    for r in range(0, 3):
        for c in range(0, 3):
            # K1[r, c] = leftCameraIntrinsicParams[r, c] #no-need for python
            # K2[r, c] = rightCameraIntrinsicParams[r, c] #no-need for python
            E2[r, c] = R2[r, c]
    E2[r, 3] = leftToRightTranslationVector[r, 0]

    # We invert the intrinsic params, so we can convert from pixels to normalised image coordinates.
    K1Inv = np.linalg.inv(K1)
    K2Inv = np.linalg.inv(K2)

    # # We want output coordinates relative to left camera.
    E1Inv = np.linalg.inv(E1)
    L2R = E2 * E1Inv

    # Reading Prince 2012 Computer Vision, the projection matrix, is just the extrinsic parameters,
    # as our coordinates will be in a normalised camera space. P1 should be identity, so that
    # reconstructed coordinates are in Left Camera Space, to P2 should reflect a right to left transform.
    P1d[0, 0] = 1
    P1d[0, 1] = 0
    P1d[0, 2] = 0
    P1d[0, 3] = 0
    P1d[1, 0] = 0
    P1d[1, 1] = 1
    P1d[1, 2] = 0
    P1d[1, 3] = 0
    P1d[2, 0] = 0
    P1d[2, 1] = 0
    P1d[2, 2] = 1
    P1d[2, 3] = 0

    for i in range(0, 3):
        for j in range(0, 4):
            P2d[i, j] = L2R[i, j]

    # `#pragma omp parallel` for its C++ counterpart
    u1 = np.zeros((3, 1), dtype=np.double)
    u2 = np.zeros((3, 1), dtype=np.double)
    u1t = np.zeros((3, 1), dtype=np.double)
    u2t = np.zeros((3, 1), dtype=np.double)

    u1p = np.zeros((3, 1), dtype=np.double)
    # Normalised image coordinates. (i.e. relative to a principal point of zero, and in millimetres not pixels).
    u2p = np.zeros((3, 1), dtype=np.double)
    # reconstructedPoint = np.zeros((3, 1, 1), dtype=np.double)  # the output 3D point, in reference frame of left camera.

    # print(f'\n {u1p}')

    # # `pragma omp for` for its C++ counterpart
    for i in range(1, numberOfPoints):
        u1[0, 0] = inputUndistortedPoints[i, 0]
        u1[1, 0] = inputUndistortedPoints[i, 1]
        u1[2, 0] = 1

        u2[0, 0] = inputUndistortedPoints[i, 2]
        u2[1, 0] = inputUndistortedPoints[i, 3]
        u2[2, 0] = 1

        # Converting to normalised image points
        u1t = K1Inv * u1
        u2t = K2Inv * u2

        u1p[0] = u1t[0, 0]
        u1p[1] = u1t[1, 0]
        u1p[2] = u1t[2, 0]

        u2p[0] = u1t[0, 0]
        u2p[1] = u1t[1, 0]
        u2p[2] = u1t[2, 0]

        reconstructedPoint = InternalIterativeTriangulatePointUsingSVD(P1d, P2d, u1p, u2p)

        outputPoints[i, 0] = reconstructedPoint[0]
        outputPoints[i, 1] = reconstructedPoint[1]
        outputPoints[i, 2] = reconstructedPoint[2]

    return outputPoints


def triangulate_points_using_hartley_opencv(left_undistorted,
                                            right_undistorted,
                                            leftCameraIntrinsicParams,
                                            rightCameraIntrinsicParams,
                                            leftToRightRotationMatrix,
                                            leftToRightTranslationVector):
    """
    Function to compute triangulation of points using Harley with cv2.triangulatePoints

    :param inputUndistortedPoints:
    :param leftCameraIntrinsicParams:
    :param rightCameraIntrinsicParams:
    :param leftToRightRotationMatrix:
    :param leftToRightTranslationVector:

    p_l,p_r: Left and Right camera projection matrixes
    left_undistorted, right_undistorted: point image positions in 2 cameras
    
    References
    ----------
    Hartley, Richard I., and Peter Sturm. "Triangulation." Computer vision and image understanding 68, no. 2 (1997): 146-157.

    :return triangulated_cv:
    """

    l2r_mat = stm.construct_rigid_transformation(leftToRightRotationMatrix, leftToRightTranslationVector)
    p_l = np.zeros((3, 4))
    p_l[:, :-1] = leftCameraIntrinsicParams
    p_r = np.zeros((3, 4))
    p_r[:, :-1] = rightCameraIntrinsicParams
    p_l = np.matmul(p_l, np.eye(4))
    p_r = np.matmul(p_r, l2r_mat)
    triangulated_cv = cv2.triangulatePoints(p_l,
                                            p_r,
                                            left_undistorted,
                                            right_undistorted)

    return triangulated_cv
