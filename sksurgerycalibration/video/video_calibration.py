# -*- coding: utf-8 -*-

""" Video Calibration functions. """

import logging
import numpy as np
import cv2
import sksurgerycore.transforms.matrix as skcm
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm

LOGGER = logging.getLogger(__name__)


def mono_video_calibration(object_points, image_points, image_size, flags=0):
    """
    Calibrates a video camera using Zhang's 2000 method, as implemented in
    OpenCV. We wrap it here, so we have a place to add extra validation code,
    and a space for documentation. The aim is to check everything before
    we pass it to OpenCV, and raise Exceptions consistently for any error
    we can detect before we pass it to OpenCV, as OpenCV just dies
    without throwing exceptions.

      - N = number of images
      - M = number of points for that image
      - rvecs = list of 1x3 Rodrigues rotation parameters
      - tvecs = list of 3x1 translation vectors
      - camera_matrix = [3x3] ndarray containing fx, fy, cx, cy
      - dist_coeffs = [1x5] ndarray, containing distortion coefficients

    :param object_points: Vector (N) of Vector (M) of 1x3 points of type float
    :param image_points: Vector (N) of Vector (M) of 1x2 points of type float
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :param flags: OpenCV flags to pass to calibrateCamera().
    :return: rms, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    rms, camera_matrix, dist_coeffs, rvecs, tvecs \
        = cv2.calibrateCamera(object_points,
                              image_points,
                              image_size,
                              None, None,
                              flags=flags)

    # Recompute this, for consistency.
    sse, num = vm.compute_mono_2d_err(object_points,
                                      image_points,
                                      rvecs,
                                      tvecs,
                                      camera_matrix,
                                      dist_coeffs)
    mse = sse / num
    final_rms = np.sqrt(mse)

    return final_rms, camera_matrix, dist_coeffs, rvecs, tvecs


# pylint: disable=too-many-locals
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
    sse, num_samples = \
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
    mse = sse / num_samples
    s_reproj = np.sqrt(mse)

    sse, number_samples = \
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

    mse = sse / num_samples
    s_recon = np.sqrt(mse)

    LOGGER.info("Stereo Calib: l=%s, r=%s, opencv=%s, proj=%s, recon=%s",
                str(l_rms), str(r_rms), str(s_rms), str(s_reproj), str(s_recon))

    return s_reproj, s_recon, \
        l_c, l_d, l_rvecs, l_tvecs, \
        r_c, r_d, r_rvecs, r_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental
