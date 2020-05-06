# -*- coding: utf-8 -*-

""" Some more experimental video calibration routines. Use at your own risk. """

# pylint: disable=too-many-locals

import logging
import copy
import numpy as np
import cv2
from scipy.optimize import minimize
import sksurgerycore.transforms.matrix as skcm
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_cost_functions as vcf
import sksurgerycalibration.video.video_calibration_metrics as vm
import sksurgerycalibration.video.video_calibration_wrapper as vc

LOGGER = logging.getLogger(__name__)


def mono_video_calibration_expt(ids,
                                object_points,
                                image_points,
                                image_size):
    """
    Experimental.

    :param ids: Vector of ndarrays containing integer point ids.
    :param object_points: Vector of Vectors of 1x3 object points, float32
    :param image_points:  Vector of Vectors of 1x2 object points, float32
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return: rms, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    # First do a standard OpenCV calibration, using the wrapper in this project.
    _, camera_matrix, dist_coeffs, rvecs, tvecs = \
        vc.mono_video_calibration(object_points,
                                  image_points,
                                  image_size)

    number_of_views = len(rvecs)
    number_of_intrinsic_parameters = 4 + dist_coeffs.shape[1]
    number_of_extrinsic_parameters = 6 * len(rvecs)

    # Now alternately optimise intrinsics and extrinsics
    for _ in range(0, 5):

        # First optimise intrinsics, via projection error.
        x_0 = np.zeros(number_of_intrinsic_parameters)
        x_0[0] = camera_matrix[0][0]
        x_0[1] = camera_matrix[1][1]
        x_0[2] = camera_matrix[0][2]
        x_0[3] = camera_matrix[1][2]
        for i in range(4, number_of_intrinsic_parameters):
            x_0[i] = dist_coeffs[0][i - 4]

        res = minimize(vcf.mono_reproj_err_for_intrin, x_0,
                       args=(object_points,
                             image_points,
                             rvecs,
                             tvecs),
                       method='Nelder-Mead',
                       tol=1e-3,
                       options={'disp': False, 'maxiter': 1000})
        x_1 = res.x
        camera_matrix[0][0] = x_1[0]
        camera_matrix[1][1] = x_1[1]
        camera_matrix[0][2] = x_1[2]
        camera_matrix[1][2] = x_1[3]
        for i in range(4, number_of_intrinsic_parameters):
            dist_coeffs[0][i - 4] = x_1[i]

        # Now optimise extrinsics, via triangulation.
        x_0 = np.zeros(number_of_extrinsic_parameters)
        for i in range(0, number_of_views):
            x_0[6 * i + 0] = rvecs[i][0]
            x_0[6 * i + 1] = rvecs[i][1]
            x_0[6 * i + 2] = rvecs[i][2]
            x_0[6 * i + 3] = tvecs[i][0]
            x_0[6 * i + 4] = tvecs[i][1]
            x_0[6 * i + 5] = tvecs[i][2]

        res = minimize(vcf.mono_recon_err_for_ext, x_0,
                       args=(
                           ids,
                           object_points,
                           image_points,
                           camera_matrix,
                           dist_coeffs
                       ),
                       method='Nelder-Mead',
                       tol=1e-3,
                       options={'disp': False, 'maxiter': 1000})
        x_1 = res.x
        for i in range(0, number_of_views):
            rvecs[i][0] = x_1[6 * i + 0]
            rvecs[i][1] = x_1[6 * i + 1]
            rvecs[i][2] = x_1[6 * i + 2]
            tvecs[i][0] = x_1[6 * i + 3]
            tvecs[i][1] = x_1[6 * i + 4]
            tvecs[i][2] = x_1[6 * i + 5]

    sse, num = vm.compute_mono_2d_err(object_points,
                                      image_points,
                                      rvecs,
                                      tvecs,
                                      camera_matrix,
                                      dist_coeffs)

    mse = sse / num
    final_rms = np.sqrt(mse)

    return final_rms, camera_matrix, dist_coeffs, rvecs, tvecs


def stereo_video_calibration_expt(left_ids,
                                  left_object_points,
                                  left_image_points,
                                  right_ids,
                                  right_object_points,
                                  right_image_points,
                                  image_size):
    """
    Experimental.

    :param left_ids: Vector of ndarrays containing integer point ids.
    :param left_object_points: Vector of Vectors of 1x3 object points, float32
    :param left_image_points:  Vector of Vectors of 1x2 object points, float32
    :param right_ids: Vector of ndarrays containing integer point ids.
    :param right_object_points: Vector of Vectors of 1x3 object points, float32
    :param right_image_points: Vector of Vectors of 1x2 object points, float32
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return:
    """
    # First do standard OpenCV calibration, using the wrapper in this project.
    s_reproj, s_recon, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental = \
        vc.stereo_video_calibration(left_ids,
                                    left_object_points,
                                    left_image_points,
                                    right_ids,
                                    right_object_points,
                                    right_image_points,
                                    image_size)

    # For stereo reconstruction error we need common points.
    _, common_object_points, common_left_image_points, \
        common_right_image_points \
        = vu.filter_common_points_all_images(left_ids,
                                             left_object_points,
                                             left_image_points,
                                             right_ids,
                                             right_image_points, 10)

    l2r_rvec = (cv2.Rodrigues(l2r_r.T))[0]

    # Now compute a set of left extrinsics, where the parameters
    # are all relative to the first camera.
    camera_rvecs = copy.deepcopy(l_rvecs)
    camera_tvecs = copy.deepcopy(l_tvecs)
    number_of_frames = len(left_object_points)
    first_world_to_camera = \
        vu.extrinsic_vecs_to_matrix(camera_rvecs[0], camera_tvecs[0])
    first_camera_to_world = np.linalg.inv(first_world_to_camera)
    for i in range(1, number_of_frames):
        extrinsic = \
            vu.extrinsic_vecs_to_matrix(camera_rvecs[i], camera_tvecs[i])
        relative_to_first = np.matmul(extrinsic, first_camera_to_world)
        camera_rvecs[i], camera_tvecs[i] = \
            vu.extrinsic_matrix_to_vecs(relative_to_first)

    # Now we have to create a flat vector of parameters to optimise.
    # Optimsing l2r_r (3), l2r_t(3), l_c (4), r_c (4) + 6 DOF per camera.
    number_of_parameters = 3 + 3 + 4 + 4 + (number_of_frames * 6)
    x_0 = np.zeros(number_of_parameters)
    x_0[0] = l2r_rvec[0][0]
    x_0[1] = l2r_rvec[1][0]
    x_0[2] = l2r_rvec[2][0]
    x_0[3] = l2r_t[0][0]
    x_0[4] = l2r_t[1][0]
    x_0[5] = l2r_t[2][0]
    x_0[6] = l_c[0][0]
    x_0[7] = l_c[1][1]
    x_0[8] = l_c[0][2]
    x_0[9] = l_c[1][2]
    x_0[10] = r_c[0][0]
    x_0[11] = r_c[1][1]
    x_0[12] = r_c[0][2]
    x_0[13] = r_c[1][2]

    for i in range(0, number_of_frames):
        x_0[14 + i * 6 + 0] = camera_rvecs[i][0]
        x_0[14 + i * 6 + 1] = camera_rvecs[i][1]
        x_0[14 + i * 6 + 2] = camera_rvecs[i][2]
        x_0[14 + i * 6 + 3] = camera_tvecs[i][0]
        x_0[14 + i * 6 + 4] = camera_tvecs[i][1]
        x_0[14 + i * 6 + 5] = camera_tvecs[i][2]

    res = minimize(vcf.stereo_2d_and_3d_error, x_0,
                   args=(common_object_points,
                         common_left_image_points,
                         l_d,
                         common_object_points,
                         common_right_image_points,
                         r_d),
                   method='Powell',
                   tol=1e-4,
                   options={'disp': False, 'maxiter': 100000})

    LOGGER.info("Stereo Re-Calibration: success=%s", str(res.success))
    LOGGER.info("Stereo Re-Calibration: msg=%s", str(res.message))

    # Now need to unpack the results, into the same set of vectors,
    # as the stereo_video_calibration, so they are drop-in replacements.
    x_1 = res.x
    l2r_rvec[0][0] = x_1[0]
    l2r_rvec[1][0] = x_1[1]
    l2r_rvec[2][0] = x_1[2]
    l2r_r = (cv2.Rodrigues(l2r_rvec))[0]
    l2r_t[0][0] = x_1[3]
    l2r_t[1][0] = x_1[4]
    l2r_t[2][0] = x_1[5]
    l_c[0][0] = x_1[6]
    l_c[1][1] = x_1[7]
    l_c[0][2] = x_1[8]
    l_c[1][2] = x_1[9]
    r_c[0][0] = x_1[10]
    r_c[1][1] = x_1[11]
    r_c[0][2] = x_1[12]
    r_c[1][2] = x_1[13]

    for i in range(0, number_of_frames):
        camera_rvecs[i][0] = x_1[14 + i * 6 + 0]
        camera_rvecs[i][1] = x_1[14 + i * 6 + 1]
        camera_rvecs[i][2] = x_1[14 + i * 6 + 2]
        camera_tvecs[i][0] = x_1[14 + i * 6 + 3]
        camera_tvecs[i][1] = x_1[14 + i * 6 + 4]
        camera_tvecs[i][2] = x_1[14 + i * 6 + 5]

    # Still need to convert these parameters (which are relative to first
    # camera), into consistent world-to-camera rvecs and tvecs for left/right.
    first_world_to_camera = \
        vu.extrinsic_vecs_to_matrix(camera_rvecs[0], camera_tvecs[0])
    left_to_right = skcm.construct_rigid_transformation(l2r_r, l2r_t)
    for i in range(0, number_of_frames):
        if i == 0:
            left_world_to_camera = first_world_to_camera
        else:
            extrinsic = \
                vu.extrinsic_vecs_to_matrix(camera_rvecs[i], camera_tvecs[i])
            left_world_to_camera = np.matmul(extrinsic, first_world_to_camera)
        l_rvecs[i], l_tvecs[i] = \
            vu.extrinsic_matrix_to_vecs(left_world_to_camera)
        right_world_to_camera = \
            np.matmul(left_to_right, left_world_to_camera)
        r_rvecs[i], r_tvecs[i] = \
            vu.extrinsic_matrix_to_vecs(right_world_to_camera)

    # And recompute stereo RMS re-projection error.
    s_reproj_2 = \
        vm.compute_stereo_2d_err(l2r_r,
                                 l2r_t,
                                 common_object_points,
                                 common_left_image_points,
                                 l_c,
                                 l_d,
                                 common_object_points,
                                 common_right_image_points,
                                 r_c,
                                 r_d,
                                 l_rvecs,
                                 l_tvecs
                                 )

    s_recon_2 = \
        vm.compute_stereo_3d_error(l2r_r,
                                   l2r_t,
                                   common_object_points,
                                   common_left_image_points,
                                   l_c,
                                   l_d,
                                   common_right_image_points,
                                   r_c,
                                   r_d,
                                   l_rvecs,
                                   l_tvecs
                                   )

    LOGGER.info(
        "Stereo Re-Calib: reproj_1=%s, recon_1=%s, reproj_2=%s, recon_2=%s",
        str(s_reproj), str(s_recon), str(s_reproj_2), str(s_recon_2))

    return s_reproj_2, s_recon_2, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental
