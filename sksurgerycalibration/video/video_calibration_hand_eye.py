# -*- coding: utf-8 -*-

""" SmartLiver calibration interface. """

import logging
import datetime
from fnmatch import filter as file_filter
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from sksurgeryimage.processing import charuco_point_detector as charuco_pd
from sksurgeryimage.processing import chessboard_point_detector as chessboard_pd
import smartliver.utils.smartliver_utils as slu

LOGGER = logging.getLogger(__name__)


# pylint: disable=invalid-name, no-member, unsubscriptable-object
# pylint: disable=too-many-locals, too-many-arguments, too-many-lines
# pylint: disable=too-many-branches

def quat_conjugate(q):
    """
    Obtains the conjugate of a quaternion.
    """
    assert len(q) == 4

    result = [q[0], -q[1], -q[2], -q[3]]

    return np.asarray(result)


def quat_multiply(q, r):
    """
    Calculates the quaternion product n, for two given quaternions q and r:
    n = q x r
    """
    assert len(q) == 4
    assert len(r) == 4
    n0 = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    n1 = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    n2 = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    n3 = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]

    result = [n0, n1, n2, n3]

    return np.asarray(result)


def quat_to_rotm(q):
    """
    Get the corresponding rotation matrix of a quaternion.
    """
    assert len(q) == 4

    qn = q / np.linalg.norm(q)
    s = qn[0]
    x = qn[1]
    y = qn[2]
    z = qn[3]
    M = [[s**2 + x**2 - y**2 - z**2, 2*(x*y - s*z), 2*(x*z + s*y)],
         [2*(x*y + s*z), s**2 - x**2 + y**2 - z**2, 2*(y*z - s*x)],
         [2*(x*z - s*y), 2*(y*z + s*x), s**2 - x**2 - y**2 + z**2]]

    return np.asarray(M)


def rotm_to_quat_precise(M):
    """
    Get the corresponding quaternion of a rotation matrix.
    Assuming the rotation matrix is orthonormal.
    """
    q = np.empty((4,))
    tr = np.trace(M)
    if tr > 0:
        q[0] = tr + 1.0
        q[3] = M[1, 0] - M[0, 1]
        q[2] = M[0, 2] - M[2, 0]
        q[1] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        tr = M[i, i] - (M[j, j] + M[k, k])
        q[i] = tr + 1.0
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]

    q *= 0.5 / np.sqrt(tr + 1.0)

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def rotm_to_quat(M):
    """
    Get the corresponding quaternion of a rotation matrix.
    Assuming the rotation matrix is not strictly orthonormal.
    """
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    # symmetric matrix K
    K = np.array([[m00-m11-m22, 0.0, 0.0, 0.0],
                  [m01+m10, m11-m00-m22, 0.0, 0.0],
                  [m02+m20, m12+m21, m22-m00-m11, 0.0],
                  [m21-m12, m02-m20, m10-m01, m00+m11+m22]])
    K /= 3.0

    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0.0:
        np.negative(q, q)

    return q


def rvec_to_quat(rvec):
    """
    Get the corresponding quaternion of a rotation vector.
    """
    rad = np.linalg.norm(rvec)
    axis = rvec/rad
    q = np.concatenate([[np.cos(rad/2)], np.sin(rad/2) * axis])

    return q


def to_one_hemisphere(quaternions):
    """
    Transform a group of quaternions to one hemisphere.
    """
    new_quaternions = np.zeros(np.shape(quaternions))

    sum_1 = np.sum(np.abs(quaternions[:, 0]))
    sum_2 = np.sum(np.abs(quaternions[:, 1]))
    sum_3 = np.sum(np.abs(quaternions[:, 2]))
    sum_4 = np.sum(np.abs(quaternions[:, 3]))

    max_sum = sum_1
    flag = 0

    if sum_2 > max_sum:
        max_sum = sum_2
        flag = 1

    if sum_3 > max_sum:
        max_sum = sum_3
        flag = 2

    if sum_4 > max_sum:
        flag = 3

    number = np.shape(quaternions)[0]

    for i in range(0, number):
        if quaternions[i, flag] > 0:
            new_quaternions[i] = quaternions[i]
        else:
            new_quaternions[i] = -quaternions[i]

    return new_quaternions


def solve_2quaternions(qx, q_A, q_B):
    """
    Provide cost function for least-square optimisation
    to solve quaternions qx1 and qx2 in: q_B = qx1 * q_A * qx2

    :param qx: 1x8 vector of 2 quaternions
    :param q_A: Nx4 matrices of N quaternions
    :param q_B: Nx4 matrices of N quaternions
    :return:
    """
    if np.shape(q_A) != np.shape(q_B):
        raise AssertionError("q_A and q_B must have the same shape.")

    number_of_frames = np.shape(q_A)[0]

    f = []

    qx_1 = qx[0:4]
    qx_2 = qx[4:8]

    qx_1 = qx_1/np.linalg.norm(qx_1)
    qx_2 = qx_2/np.linalg.norm(qx_2)

    for i in range(0, number_of_frames):
        f.append(quat_multiply(quat_multiply(qx_1, q_A[i]), qx_2) - q_B[i])

    f = np.ndarray.flatten(np.asarray(f))

    return f


def solve_2translations(q_handeye,
                        quat_model2hand_array,
                        trans_model2hand_array,
                        trans_extrinsics_array):
    """
    Solve translations t_handeye and t_pattern2marker.
    translations = [t_handeye t_pattern2marker]

    :param q_handeye: 1x4 matrix quaternion
    :param quat_model2hand_array: Nx4 matrix of N quaternions
    :param trans_model2hand_array: Nx3 matrices of N translations
    :param trans_extrinsics_array: Nx3 matrices of extrinsic translation vectors
    :return:
    """
    if np.shape(quat_model2hand_array)[0] \
            != np.shape(trans_model2hand_array)[0]:
        raise AssertionError("Wrong dimensions in solve_2translations().")
    if np.shape(trans_model2hand_array) != np.shape(trans_extrinsics_array):
        raise AssertionError("Wrong dimensions in solve_2translations().")

    number_of_frames = np.shape(quat_model2hand_array)[0]

    A = np.zeros((3 * number_of_frames, 6))
    B = np.zeros((3 * number_of_frames, 1))

    for i in range(0, number_of_frames):
        qHEqMH = quat_multiply(q_handeye, quat_model2hand_array[i])
        rHErMH = quat_to_rotm(qHEqMH)
        rHE = quat_to_rotm(q_handeye)

        A[i*3:i*3+3, :] = np.hstack([np.identity(3), rHErMH])
        b_i = np.transpose(trans_extrinsics_array[i, :])\
            - np.matmul(rHE, np.transpose(trans_model2hand_array[i, :]))
        B[i*3:i*3+3] = np.reshape(b_i, (3, 1))

    A_T = np.transpose(A)
    translations = np.linalg.inv(A_T @ A) @ A_T @ B
    translations = np.transpose(translations)

    return translations.flatten()
