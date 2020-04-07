# -*- coding: utf-8 -*-

""" Video Calibration functions. """

import logging
import copy
import numpy as np
import cv2
from scipy.optimize import minimize
import sksurgerycore.transforms.matrix as skcm
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm
import sksurgerycalibration.video.video_calibration_cost_functions as vc

LOGGER = logging.getLogger(__name__)


def mono_video_calibration(object_points, image_points, image_size):
    """
    Calibrates a video camera using Zhang's 2000 method, as implemented in
    OpenCV. We wrap it here, so we have a place to add extra validation code,
    and a space for documentation. The aim is to check everything before
    we pass it to OpenCV, and raise Exceptions consistently for any error
    we can detect before we pass it to OpenCV.

      - N = number of images
      - M = number of points for that image
      - rvecs = list of 1x3 Rodrigues rotation parameters
      - tvecs = list of 3x1 translation vectors
      - camera_matrix = [3x3] ndarray containing fx, fy, cx, cy
      - dist_coeffs = [1x5] ndarray, containing distortion coefficients

    :param object_points: Vector (N) of Vector (M) of 1x3 points of type float
    :param image_points: Vector (N) of Vector (M) of 1x2 points of type float
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return: retval, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    rms, camera_matrix, dist_coeffs, rvecs, tvecs \
        = cv2.calibrateCamera(object_points,
                              image_points,
                              image_size,
                              None, None)

    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


def stereo_video_calibration(left_ids,
                             left_object_points,
                             left_image_points,
                             right_ids,
                             right_object_points,
                             right_image_points,
                             image_size):
    """
    Default stereo calibration, using OpenCV methods.

    We wrap it here, so we have a place to add extra validation code,
    and a space for documentation. The aim is to check everything before
    we pass it to OpenCV, and raise Exceptions consistently for any error
    we can detect before we pass it to OpenCV.

    :param left_ids: Vector of ndarrays containing integer point ids.
    :param left_object_points: Vector of Vectors of 1x3 object points, float32
    :param left_image_points:  Vector of Vectors of 1x2 object points, float32
    :param right_ids: Vector of ndarrays containing integer point ids.
    :param right_object_points: Vector of Vectors of 1x3 object points, float32
    :param right_image_points: Vector of Vectors of 1x2 object points, float32
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return:
    """
    # Calibrate left, using all available points
    l_rms, l_c, l_d, l_rvecs, l_tvecs \
        = cv2.calibrateCamera(left_object_points,
                              left_image_points,
                              image_size,
                              None, None)

    # Calibrate right using all available points.
    r_rms, r_c, r_d, r_rvecs, r_tvecs \
        = cv2.calibrateCamera(right_object_points,
                              right_image_points,
                              image_size,
                              None, None)

    # But for stereo, OpenCV needs common points.
    _, common_object_points, common_left_image_points, \
        common_right_image_points \
        = vu.filter_common_points_all_images(left_ids,
                                             left_object_points,
                                             left_image_points,
                                             right_ids,
                                             right_image_points, 10)

    # So, now we can calibrate using only points that occur in left and right.
    s_rms, l_c, l_d, r_c, r_d, \
        l2r_r, l2r_t, essential, fundamental = cv2.stereoCalibrate(
            common_object_points,
            common_left_image_points,
            common_right_image_points,
            l_c,
            l_d,
            r_c,
            r_d,
            image_size,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    # And recompute rvecs and tvecs, consistently, given new l2r params.
    number_of_frames = len(left_object_points)
    left_to_right = skcm.construct_rigid_transformation(l2r_r, l2r_t)
    for i in range(0, number_of_frames):
        _, l_rvecs[i], l_tvecs[i] = cv2.solvePnP(
            common_object_points[i],
            common_left_image_points[i],
            l_c,
            l_d)
        left_chessboard_to_camera = \
            vu.extrinsic_vecs_to_matrix(l_rvecs[i], l_tvecs[i])
        right_chessboard_to_camera = \
            np.matmul(left_to_right, left_chessboard_to_camera)
        r_rvecs[i], r_tvecs[i] = \
            vu.extrinsic_matrix_to_vecs(right_chessboard_to_camera)

    # And recompute stereo projection error, given left camera and l2r.
    # We also use all points, not just common points, for comparison
    # with other methods outside of this function.
    s_reproj = vm.compute_stereo_rms_projection_error(l2r_r,
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

    s_recon = vm.compute_stereo_rms_reconstruction_error(l2r_r,
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

    LOGGER.info("Stereo Calib: l=%s, r=%s, opencv=%s, proj=%s, recon=%s",
                str(l_rms), str(r_rms), str(s_rms), str(s_reproj), str(s_recon))

    return s_reproj, s_recon, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental


def stereo_video_calibration_reoptimised(left_ids,
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
    # First do standard OpenCV calibration, as shown above.
    s_reproj, s_recon, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental = stereo_video_calibration(left_ids,
                                                          left_object_points,
                                                          left_image_points,
                                                          right_ids,
                                                          right_object_points,
                                                          right_image_points,
                                                          image_size)

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

    res = minimize(vc._project_stereo_points_cost_function, x_0,
                   args=(left_object_points,
                         left_image_points,
                         l_d,
                         right_object_points,
                         right_image_points,
                         r_d),
                   method='Powell',
                   tol=1e-4,
                   options={'disp': True, 'maxiter': 100000})

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
    s_reproj_2 = vm.compute_stereo_rms_projection_error(l2r_r,
                                                        l2r_t,
                                                        left_object_points,
                                                        left_image_points,
                                                        l_c,
                                                        l_d,
                                                        right_object_points,
                                                        right_image_points,
                                                        r_c,
                                                        r_d,
                                                        l_rvecs,
                                                        l_tvecs
                                                        )

    # For stereo reconstruction error we need common points.
    _, common_object_points, common_left_image_points, \
        common_right_image_points \
        = vu.filter_common_points_all_images(left_ids,
                                             left_object_points,
                                             left_image_points,
                                             right_ids,
                                             right_image_points, 10)

    s_recon_2 = vm.compute_stereo_rms_reconstruction_error(l2r_r,
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

    LOGGER.info("Stereo Re-Calib: reproj_1=%s, recon_1=%s, reproj_2=%s, recon_2=%s",
                str(s_reproj), str(s_recon), str(s_reproj_2), str(s_recon_2))

    return s_reproj_2, s_recon_2, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental
