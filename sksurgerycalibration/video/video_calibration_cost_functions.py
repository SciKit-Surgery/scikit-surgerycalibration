# -*- coding: utf-8 -*-

""" Cost functions for video calibration, used with scipy. """

import numpy as np
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm


def stereo_2d_error_for_extrinsics(x_0,
                                   common_object_points,
                                   common_left_image_points,
                                   common_right_image_points,
                                   left_intrinsics,
                                   left_distortion,
                                   right_intrinsics,
                                   right_distortion,
                                   l2r_rmat,
                                   l2r_tvec
                                   ):
    """
    Computes a vector of residuals between projected image points
    and actual image points, for left and right image. x_0 should
    contain left camera extrinsic parameters.
    """
    rvecs = []
    tvecs = []
    number_of_frames = len(common_object_points)
    for i in range(0, number_of_frames):
        rvec = np.zeros((3, 1))
        rvec[0][0] = x_0[6 * i + 0]
        rvec[1][0] = x_0[6 * i + 1]
        rvec[2][0] = x_0[6 * i + 2]
        tvec = np.zeros((3, 1))
        tvec[0][0] = x_0[6 * i + 3]
        tvec[1][0] = x_0[6 * i + 4]
        tvec[2][0] = x_0[6 * i + 5]
        rvecs.append(rvec)
        tvecs.append(tvec)

    residual = vm.compute_stereo_2d_err(l2r_rmat,
                                        l2r_tvec,
                                        common_object_points,
                                        common_left_image_points,
                                        left_intrinsics,
                                        left_distortion,
                                        common_object_points,
                                        common_right_image_points,
                                        right_intrinsics,
                                        right_distortion,
                                        rvecs,
                                        tvecs,
                                        return_residuals=True
                                        )
    return residual


def mono_proj_err_h2e(x_0,
                      object_points,
                      image_points,
                      intrinsics,
                      distortion,
                      pattern_tracking,
                      device_tracking,
                      pattern2marker_matrix
                      ):
    """
    Computes the SSE of projected
    image points and actual image points, for a single camera,
    where we have a tracked calibration pattern, and assume the
    pattern2marker transform should remain fixed. Therefore we
    only optimise hand-eye. So, x_0 should be of length 6.
    """
    assert len(x_0) == 6

    rvec = np.zeros((3, 1))
    rvec[0] = x_0[0]
    rvec[1] = x_0[1]
    rvec[2] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0] = x_0[3]
    tvec[1] = x_0[4]
    tvec[2] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    number_of_frames = len(object_points)
    rvecs = []
    tvecs = []

    # Computes pattern2camera for each pose
    for i in range(0, number_of_frames):

        p2c = h2e @ np.linalg.inv(device_tracking[i]) @ \
              pattern_tracking[i] @ pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_mono_2d_err(object_points,
                                     image_points,
                                     rvecs,
                                     tvecs,
                                     intrinsics,
                                     distortion,
                                     return_residuals=False)
    return proj


def mono_proj_err_p2m_h2e(x_0,
                          object_points,
                          image_points,
                          intrinsics,
                          distortion,
                          pattern_tracking,
                          device_tracking
                          ):
    """
    Computes the SSE between projected
    image points to actual image points, for a single camera,
    where we have a tracked pattern. Both the
    pattern2marker and hand2eye are optimised.
    So, x_0 should be of length 12.
    """
    assert len(x_0) == 12

    rvec = np.zeros((3, 1))
    rvec[0] = x_0[0]
    rvec[1] = x_0[1]
    rvec[2] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0] = x_0[3]
    tvec[1] = x_0[4]
    tvec[2] = x_0[5]

    p2m = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    rvec[0] = x_0[6]
    rvec[1] = x_0[7]
    rvec[2] = x_0[8]

    tvec[0] = x_0[9]
    tvec[1] = x_0[10]
    tvec[2] = x_0[11]

    h2e = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    number_of_frames = len(object_points)
    rvecs = []
    tvecs = []

    # Computes pattern2camera for each pose
    for i in range(0, number_of_frames):

        p2c = h2e @ np.linalg.inv(device_tracking[i])\
              @ pattern_tracking[i] @ p2m

        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_mono_2d_err(object_points,
                                     image_points,
                                     rvecs,
                                     tvecs,
                                     intrinsics,
                                     distortion,
                                     return_residuals=False)
    return proj


def mono_proj_err_h2e_g2w(x_0,
                          object_points,
                          image_points,
                          intrinsics,
                          distortion,
                          device_tracking
                          ):
    """
    Method to the SSE of projected
    image points to actual image points, for a single camera,
    where we have an untracked pattern. Both the
    hand2eye and grid2world are optimised.
    So, x_0 should be of length 12.
    """
    assert len(x_0) == 12

    rvec = np.zeros((3, 1))
    rvec[0] = x_0[0]
    rvec[1] = x_0[1]
    rvec[2] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0] = x_0[3]
    tvec[1] = x_0[4]
    tvec[2] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    rvec[0] = x_0[6]
    rvec[1] = x_0[7]
    rvec[2] = x_0[8]

    tvec[0] = x_0[9]
    tvec[1] = x_0[10]
    tvec[2] = x_0[11]

    g2w = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    number_of_frames = len(object_points)
    rvecs = []
    tvecs = []

    # Computes pattern2camera for each pose
    for i in range(0, number_of_frames):

        p2c = h2e @ np.linalg.inv(device_tracking[i]) @ g2w

        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_mono_2d_err(object_points,
                                     image_points,
                                     rvecs,
                                     tvecs,
                                     intrinsics,
                                     distortion,
                                     return_residuals=False)
    return proj


def mono_proj_err_h2e_int_dist(x_0,
                               object_points,
                               image_points,
                               device_tracking,
                               pattern_tracking,
                               pattern2marker_matrix
                               ):
    """
    Computes the SSE between projected
    image points to actual image points, for a single camera,
    where we have a tracked pattern. The handeye, intrinsics and
    distortion parameters are optimised.
    So, x_0 should be of length 6+4+5 = 15.
    """
    assert len(x_0) == 15

    rvec = np.zeros((3, 1))
    rvec[0] = x_0[0]
    rvec[1] = x_0[1]
    rvec[2] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0] = x_0[3]
    tvec[1] = x_0[4]
    tvec[2] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    intrinsics = np.zeros((3, 3))
    intrinsics[0][0] = x_0[6]
    intrinsics[1][1] = x_0[7]
    intrinsics[0][2] = x_0[8]
    intrinsics[1][2] = x_0[9]

    distortion = np.zeros((1, 5))
    distortion[0][0] = x_0[10]
    distortion[0][1] = x_0[11]
    distortion[0][2] = x_0[12]
    distortion[0][3] = x_0[13]
    distortion[0][4] = x_0[14]

    number_of_frames = len(object_points)
    rvecs = []
    tvecs = []

    # Computes pattern2camera for each pose
    for i in range(0, number_of_frames):

        p2c = h2e @ np.linalg.inv(device_tracking[i]) @ \
              pattern_tracking[i] @ pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_mono_2d_err(object_points,
                                     image_points,
                                     rvecs,
                                     tvecs,
                                     intrinsics,
                                     distortion)
    return proj


def stereo_handeye_proj_error(x_0,
                              common_object_points,
                              common_left_image_points,
                              common_right_image_points,
                              left_intrinsics,
                              left_distortion,
                              right_intrinsics,
                              right_distortion,
                              l2r_rmat,
                              l2r_tvec,
                              device_tracking_array,
                              pattern_tracking_array,
                              left_pattern2marker_matrix=None
                              ):
    """
    Computes the SSE of projected image points
    and actual image points for left and right cameras. x_0 should contain
    the 6DOF of hand-to-eye, and if left_pattern2marker_matrix is None,
    then an additional 6DOF of pattern-to-marker. So, x_0 can be either
    length 6 or length 12.

    :param x_0:
    :param common_object_points:
    :param common_left_image_points:
    :param common_right_image_points:
    :param left_intrinsics:
    :param left_distortion:
    :param right_intrinsics:
    :param right_distortion:
    :param l2r_rmat:
    :param l2r_tvec:
    :param device_tracking_array:
    :param pattern_tracking_array:
    :param left_pattern2marker_matrix:
    :return: matrix of residuals for Levenberg-Marquardt optimisation.
    """
    rvecs = []
    tvecs = []
    number_of_frames = len(common_object_points)

    h2e_rvec = np.zeros((3, 1))
    h2e_rvec[0][0] = x_0[0]
    h2e_rvec[1][0] = x_0[1]
    h2e_rvec[2][0] = x_0[2]

    h2e_tvec = np.zeros((3, 1))
    h2e_tvec[0][0] = x_0[3]
    h2e_tvec[1][0] = x_0[4]
    h2e_tvec[2][0] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(h2e_rvec, h2e_tvec)

    if left_pattern2marker_matrix is None:

        p2m_rvec = np.zeros((3, 1))
        p2m_rvec[0][0] = x_0[6]
        p2m_rvec[1][0] = x_0[7]
        p2m_rvec[2][0] = x_0[8]

        p2m_tvec = np.zeros((3, 1))
        p2m_tvec[0][0] = x_0[9]
        p2m_tvec[1][0] = x_0[10]
        p2m_tvec[2][0] = x_0[11]

        p2m = vu.extrinsic_vecs_to_matrix(p2m_rvec, p2m_tvec)

    else:

        p2m = left_pattern2marker_matrix

    for i in range(0, number_of_frames):

        p2c = h2e @ np.linalg.inv(device_tracking_array[i]) @ pattern_tracking_array[i] @ p2m
        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_stereo_2d_err(l2r_rmat,
                                       l2r_tvec,
                                       common_object_points,
                                       common_left_image_points,
                                       left_intrinsics,
                                       left_distortion,
                                       common_object_points,
                                       common_right_image_points,
                                       right_intrinsics,
                                       right_distortion,
                                       rvecs,
                                       tvecs
                                       )
    return proj
