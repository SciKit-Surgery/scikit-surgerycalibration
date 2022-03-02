# -*- coding: utf-8 -*-

""" Cost functions for video calibration, used with scipy. """

import numpy as np
import cv2
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm


def _stereo_2d_error_for_extrinsics(x_0,
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


def stereo_2d_and_3d_error(x_0,
                           left_object_points,
                           left_image_points,
                           left_distortion,
                           right_object_points,
                           right_image_points,
                           right_distortion):
    """
    Private method to compute RMSE cost function, where x_0 contains
    the l2r rvec, l2r tvec, left intrinsics, right intrinsics, and
    left camera extrinsics.
    """
    l2r_rvec = np.zeros((3, 1))
    l2r_rvec[0][0] = x_0[0]
    l2r_rvec[1][0] = x_0[1]
    l2r_rvec[2][0] = x_0[2]
    l2r_rmat = (cv2.Rodrigues(l2r_rvec))[0]
    l2r_tvec = np.zeros((3, 1))
    l2r_tvec[0][0] = x_0[3]
    l2r_tvec[1][0] = x_0[4]
    l2r_tvec[2][0] = x_0[5]
    left_camera = np.eye(3)
    left_camera[0][0] = x_0[6]
    left_camera[1][1] = x_0[7]
    left_camera[0][2] = x_0[8]
    left_camera[1][2] = x_0[9]
    right_camera = np.eye(3)
    right_camera[0][0] = x_0[10]
    right_camera[1][1] = x_0[11]
    right_camera[0][2] = x_0[12]
    right_camera[1][2] = x_0[13]

    sse = 0
    number_of_samples = 0
    number_of_frames = len(left_object_points)

    rvecs = []
    tvecs = []
    for i in range(0, number_of_frames):
        rvec = np.zeros((3, 1))
        rvec[0][0] = x_0[14 + 6 * i + 0]
        rvec[1][0] = x_0[14 + 6 * i + 1]
        rvec[2][0] = x_0[14 + 6 * i + 2]
        tvec = np.zeros((3, 1))
        tvec[0][0] = x_0[14 + 6 * i + 3]
        tvec[1][0] = x_0[14 + 6 * i + 4]
        tvec[2][0] = x_0[14 + 6 * i + 5]
        rvecs.append(rvec)
        tvecs.append(tvec)

    for i in range(0, number_of_frames):

        world_to_left_camera = vu.extrinsic_vecs_to_matrix(rvecs[0], tvecs[0])

        if i > 0:
            offset = vu.extrinsic_vecs_to_matrix(rvecs[i], tvecs[i])
            world_to_left_camera = np.matmul(offset, world_to_left_camera)

        rvec, tvec = vu.extrinsic_matrix_to_vecs(world_to_left_camera)
        rvecs[i] = rvec
        tvecs[i] = tvec

    tmp_sse, tmp_num = vm.compute_stereo_2d_err(l2r_rmat,
                                                l2r_tvec,
                                                left_object_points,
                                                left_image_points,
                                                left_camera,
                                                left_distortion,
                                                right_object_points,
                                                right_image_points,
                                                right_camera,
                                                right_distortion,
                                                rvecs,
                                                tvecs
                                                )
    sse = sse + tmp_sse
    number_of_samples = number_of_samples + tmp_num

    tmp_sse, tmp_num = \
        vm.compute_stereo_3d_error(l2r_rmat,
                                   l2r_tvec,
                                   left_object_points,
                                   left_image_points,
                                   left_camera,
                                   left_distortion,
                                   right_image_points,
                                   right_camera,
                                   right_distortion,
                                   rvecs,
                                   tvecs
                                   )

    sse = sse + tmp_sse
    number_of_samples = number_of_samples + tmp_num
    mse = sse / number_of_samples
    rmse = np.sqrt(mse)
    return rmse


def mono_reproj_err_for_intrin(x_0,
                               object_points,
                               image_points,
                               rvecs,
                               tvecs
                               ):
    """
    Private method to compute SSE projection error over multiple views,
    where x_0 should contain just camera matrix, and then distorion params.
    """
    camera_matrix = np.zeros((3, 3))
    camera_matrix[0][0] = x_0[0]
    camera_matrix[1][1] = x_0[1]
    camera_matrix[0][2] = x_0[2]
    camera_matrix[1][2] = x_0[3]
    number_of_distortion_params = x_0.shape[0] - 4
    distortion_parameters = np.zeros((1, number_of_distortion_params))
    for i in range(0, number_of_distortion_params):
        distortion_parameters[0][i] = x_0[4 + i]

    sse, num_samples = vm.compute_mono_2d_err(object_points,
                                              image_points,
                                              rvecs,
                                              tvecs,
                                              camera_matrix,
                                              distortion_parameters)
    mse = sse / num_samples
    rmse = np.sqrt(mse)
    return rmse


def mono_recon_err_for_ext(x_0,
                           ids,
                           object_points,
                           image_points,
                           camera_matrix,
                           distortion_parameters
                           ):
    """
    Private method to compute SSE reconstruction error over multiple views,
    where x_0 should contain just extrinsic parameters.
    """
    number_of_frames = len(object_points)
    rvecs = []
    tvecs = []
    for i in range(0, number_of_frames):
        rvec = np.zeros((3, 1))
        rvec[0] = x_0[i * 6 + 0]
        rvec[1] = x_0[i * 6 + 1]
        rvec[2] = x_0[i * 6 + 2]
        tvec = np.zeros((3, 1))
        tvec[0] = x_0[i * 6 + 3]
        tvec[1] = x_0[i * 6 + 4]
        tvec[2] = x_0[i * 6 + 5]
        rvecs.append(rvec)
        tvecs.append(tvec)

    #pylint:disable=too-many-function-args
    sse, num_samples = vm.compute_mono_2d_err(ids,
                                              object_points,
                                              image_points,
                                              rvecs,
                                              tvecs,
                                              camera_matrix,
                                              distortion_parameters)
    mse = sse / num_samples
    rmse = np.sqrt(mse)
    return rmse


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
    Method to return a vector of residuals of projected
    image points to actual image points, for a single camera,
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

    proj, recon = vm.compute_mono_2d_err(object_points,
                                      image_points,
                                      rvecs,
                                      tvecs,
                                      intrinsics,
                                      distortion)
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
    Method to return a vector of residuals of projected
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

    proj, recon = vm.compute_mono_2d_err(object_points,
                                         image_points,
                                         rvecs,
                                         tvecs,
                                         intrinsics,
                                         distortion)
    return proj


def mono_proj_err_h2e_g2w(x_0,
                          object_points,
                          image_points,
                          intrinsics,
                          distortion,
                          device_tracking
                          ):
    """
    Method to return a vector of residuals of projected
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

    residual = vm.compute_mono_2d_err(object_points,
                                      image_points,
                                      rvecs,
                                      tvecs,
                                      intrinsics,
                                      distortion,
                                      return_residuals=True)
    return residual


def mono_proj_err_h2e_int_dist(x_0,
                               object_points,
                               image_points,
                               device_tracking,
                               pattern_tracking,
                               pattern2marker_matrix
                               ):
    """
    Method to return a vector of residuals of projected
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

    proj, recon = vm.compute_mono_2d_err(object_points,
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
    Cost function to return residual error. x_0 should contain an array
    of combined chessboard-marker-to-device-marker tracking 6DOF (rvec, tvec),
    chessboard-pattern-to-marker 6DOF and the device-hand-to-eye matrix 6DOF.

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

    sse, _ = vm.compute_stereo_2d_err(l2r_rmat,
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

    return sse
