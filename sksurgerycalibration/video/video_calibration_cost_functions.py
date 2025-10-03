# -*- coding: utf-8 -*-

""" Cost functions for video calibration, used with scipy. """

# pylint:disable=invalid-name, too-many-positional-arguments

import numpy as np

import sksurgerycalibration.video.video_calibration_metrics as vm
import sksurgerycalibration.video.video_calibration_utils as vu


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
    Used with scipy.least_squares for Levenberg-Marquardt optimisation for example.

    Cost function to be used when you are only optimising
    the left camera extrinsic parameters for multiple views,
    and leaving intrinsics, distortion, and left-to-right constant.
    So x_0 should be of length 6 * number_of_frames, corresponding
    to 3 rvec (OpenCV Rodrigues rotation vector) and 3 tvec parameters per frame.

    Computes a vector of residuals between projected image points
    and actual image points, for left and right image.
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
    Used with scipy.minimize for Powell or Nelder-Mead optimisation for example.

    Cost function to be used when you are only optimising the hand-eye matrix.
    So, x_0 should be of length 6, corresponding to 3 rvec and 3 tvec parameters.

    Computes the SSE of projected image points and actual image points. Computes
    the pattern-to-camera matrix for each frame, using pattern-to-marker,
    marker-to-world (calibration pattern tracking), the inverse of the
    device-to-world (device tracking, e.g. laparoscope), and hand-to-eye.
    """
    assert len(x_0) == 6

    rvec = np.zeros((3, 1))
    rvec[0][0] = x_0[0]
    rvec[1][0] = x_0[1]
    rvec[2][0] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0][0] = x_0[3]
    tvec[1][0] = x_0[4]
    tvec[2][0] = x_0[5]

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
    Similar to mono_proj_err_h2e, except x_0 should be
    of length 12, corresponding to pattern2marker and hand2eye.

    So 3 pattern2marker rvec, 3 pattern2marker tvec,
    3 hand2eye rvec, 3 hand2eye tvec.
    """
    assert len(x_0) == 12

    rvec = np.zeros((3, 1))
    rvec[0][0] = x_0[0]
    rvec[1][0] = x_0[1]
    rvec[2][0] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0][0] = x_0[3]
    tvec[1][0] = x_0[4]
    tvec[2][0] = x_0[5]

    p2m = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    rvec[0][0] = x_0[6]
    rvec[1][0] = x_0[7]
    rvec[2][0] = x_0[8]

    tvec[0][0] = x_0[9]
    tvec[1][0] = x_0[10]
    tvec[2][0] = x_0[11]

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
    Similar to mono_proj_err_p2m_h2e, except, as is the case
    in robot-world-hand-eye calibration, we assume we have
    a stationary untracked pattern, and we are estimating
    both the hand2eye and grid2world (i.e. pattern2world).

    So, x_0 should be of length 12, corresponding to
    3 hand2eye rvec, 3 hand2eye tvec,
    3 grid2world rvec, 3 grid2world tvec.
    """
    assert len(x_0) == 12

    rvec = np.zeros((3, 1))
    rvec[0][0] = x_0[0]
    rvec[1][0] = x_0[1]
    rvec[2][0] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0][0] = x_0[3]
    tvec[1][0] = x_0[4]
    tvec[2][0] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(rvec, tvec)

    rvec[0][0] = x_0[6]
    rvec[1][0] = x_0[7]
    rvec[2][0] = x_0[8]

    tvec[0][0] = x_0[9]
    tvec[1][0] = x_0[10]
    tvec[2][0] = x_0[11]

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
    Similar to mono_proj_err_h2e, except we are also optimising
    intrinsic and distortion parameters. So, x_0 contains
    rvec and tvec for hand2eye (6 DOF), then 4 intrinsics
    (fx, fy, cx, cy), then 5 distortion parameters
    (k1, k2, p1, p2, k3).

    However, note that: https://doi.org/10.3390/s19122837
    says this is a bad idea in general, as you will overfit.
    """
    assert len(x_0) == 15

    rvec = np.zeros((3, 1))
    rvec[0][0] = x_0[0]
    rvec[1][0] = x_0[1]
    rvec[2][0] = x_0[2]

    tvec = np.zeros((3, 1))
    tvec[0][0] = x_0[3]
    tvec[1][0] = x_0[4]
    tvec[2][0] = x_0[5]

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


# pylint:disable=too-many-arguments
def stereo_proj_err_h2e(x_0,
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

        p2c = h2e \
              @ np.linalg.inv(device_tracking_array[i]) \
              @ pattern_tracking_array[i] \
              @ p2m

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


def stereo_proj_err_h2e_int_dist_l2r(x_0,
                                     common_object_points,
                                     common_left_image_points,
                                     common_right_image_points,
                                     device_tracking_array,
                                     pattern_tracking_array,
                                     left_pattern2marker_matrix
                                     ):
    """
    Computes the SSE of projected image points against actual
    image points. x_0 should be 30 DOF.
    """
    h2e_rvec = np.zeros((3, 1))
    h2e_rvec[0][0] = x_0[0]
    h2e_rvec[1][0] = x_0[1]
    h2e_rvec[2][0] = x_0[2]

    h2e_tvec = np.zeros((3, 1))
    h2e_tvec[0][0] = x_0[3]
    h2e_tvec[1][0] = x_0[4]
    h2e_tvec[2][0] = x_0[5]

    h2e = vu.extrinsic_vecs_to_matrix(h2e_rvec, h2e_tvec)

    l2r_rvec = np.zeros((3, 1))
    l2r_rvec[0][0] = x_0[6]
    l2r_rvec[1][0] = x_0[7]
    l2r_rvec[2][0] = x_0[8]

    l2r_tvec = np.zeros((3, 1))
    l2r_tvec[0][0] = x_0[9]
    l2r_tvec[1][0] = x_0[10]
    l2r_tvec[2][0] = x_0[11]

    l2r = vu.extrinsic_vecs_to_matrix(l2r_rvec, l2r_tvec)

    left_intrinsics = np.zeros((3, 3))
    left_intrinsics[0][0] = x_0[12]
    left_intrinsics[1][1] = x_0[13]
    left_intrinsics[0][2] = x_0[14]
    left_intrinsics[1][2] = x_0[15]

    left_distortion = np.zeros((1, 5))
    left_distortion[0][0] = x_0[16]
    left_distortion[0][1] = x_0[17]
    left_distortion[0][2] = x_0[18]
    left_distortion[0][3] = x_0[19]
    left_distortion[0][4] = x_0[20]

    right_intrinsics = np.zeros((3, 3))
    right_intrinsics[0][0] = x_0[21]
    right_intrinsics[1][1] = x_0[22]
    right_intrinsics[0][2] = x_0[23]
    right_intrinsics[1][2] = x_0[24]

    right_distortion = np.zeros((1, 5))
    right_distortion[0][0] = x_0[25]
    right_distortion[0][1] = x_0[26]
    right_distortion[0][2] = x_0[27]
    right_distortion[0][3] = x_0[28]
    right_distortion[0][4] = x_0[29]

    rvecs = []
    tvecs = []
    number_of_frames = len(common_object_points)

    for i in range(0, number_of_frames):

        p2c = h2e \
              @ np.linalg.inv(device_tracking_array[i]) \
              @ pattern_tracking_array[i] \
              @ left_pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(p2c)

        rvecs.append(rvec)
        tvecs.append(tvec)

    proj, _ = vm.compute_stereo_2d_err(l2r[0:3, 0:3],
                                       l2r[0:3, 3],
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
