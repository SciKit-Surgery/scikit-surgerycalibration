# -*- coding: utf-8 -*-

"""
Utilities for manipulating quaternions.
"""
import numpy as np


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

