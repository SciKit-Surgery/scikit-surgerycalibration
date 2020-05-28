# -*- coding: utf-8 -*-

""" Video Calibration functions, that wrap OpenCV functions mainly. """

import logging
from typing import List
import numpy as np
import cv2
import sksurgerycore.transforms.matrix as skcm
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm
import sksurgerycalibration.video.video_calibration_hand_eye as he

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
    if image_size[0] < 1:
        raise ValueError("Image width must be > 0.")
    if image_size[1] < 1:
        raise ValueError("Image height must be > 0.")
    if len(object_points) < 2:
        raise ValueError("Must have at least 2 sets of object points.")
    if len(image_points) < 2:
        raise ValueError("Must have at least 2 sets of image points.")
    if len(object_points) != len(image_points):
        raise ValueError("Image points and object points differ in length.")
    for i, _ in enumerate(object_points):
        if len(object_points[i]) < 3:
            raise ValueError(str(i) + ": Must have at least 3 object points.")
        if len(image_points[i]) < 3:
            raise ValueError(str(i) + ": Must have at least 3 image points.")
        if len(object_points[i]) != len(image_points[i]):
            raise ValueError(str(i) + ": Must have the same number of points.")

    _, camera_matrix, dist_coeffs, rvecs, tvecs \
        = cv2.calibrateCamera(object_points,
                              image_points,
                              image_size,
                              None, None,
                              flags=flags)

    # Recompute this, for consistency with stereo methods.
    # i.e. so we know what the calculation is exactly.
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
                             image_size,
                             flags=cv2.CALIB_USE_INTRINSIC_GUESS
                             ):
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
    :param flags: OpenCV flags to pass to calibrateCamera().
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

    # First do stereo calibration, using fixed intrinsics.
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
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_INTRINSIC)

    # Then do it again, using the passed in flags.
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
            flags=flags)

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

    sse, num_samples = \
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


#pylint:disable=too-many-arguments
def mono_handeye_calibration(object_points: List,
                             image_points: List,
                             ids: List,
                             camera_matrix: np.ndarray,
                             camera_distortion: np.ndarray,
                             device_tracking_array: List,
                             model_tracking_array: List,
                             rvecs: List[np.ndarray],
                             tvecs: List[np.ndarray],
                             quat_model2hand_array: List,
                             trans_model2hand_array: List):
    """Wrapper around handeye calibration functions and reprojection /
    reconstruction error metrics.

    :param object_points: Vector of Vectors of 1x3 object points, float32
    :type object_points: List
    :param image_points: Vector of Vectors of 1x2 object points, float32
    :type image_points: List
    :param ids: Vector of ndarrays containing integer point ids.
    :type ids: List
    :param camera_matrix: Camera intrinsic matrix
    :type camera_matrix: np.ndarray
    :param camera_distortion: Camera distortion coefficients
    :type camera_distortion: np.ndarray
    :param device_tracking_array: Tracking data for camera (hand)
    :type device_tracking_array: List
    :param model_tracking_array: Tracking data for calibration target
    :type model_tracking_array: List
    :param rvecs: Vector of 3x1 ndarray, Rodrigues rotations for each camera
    :type rvecs: List[np.ndarray]
    :param tvecs: Vector of [3x1] ndarray, translations for each camera
    :type tvecs: List[np.ndarray]
    :param quat_model2hand_array: Array of model to hand quaternions
    :type quat_model2hand_array: List
    :param trans_model2hand_array: Array of model to hand translaions
    :type trans_model2hand_array: List
    :return: Reprojection error, reconstruction error, handeye matrix,
    patter to marker matrix
    :rtype: float, float, np.ndarray, np.ndarray
    """
    handeye_matrix, pattern2marker_matrix =  \
        he.handeye_calibration(rvecs, tvecs, quat_model2hand_array,
                               trans_model2hand_array)

    sse, num_samples = vm.compute_mono_2d_err_handeye(object_points,
                                                      image_points,
                                                      camera_matrix,
                                                      camera_distortion,
                                                      device_tracking_array,
                                                      model_tracking_array,
                                                      handeye_matrix,
                                                      pattern2marker_matrix
                                                      )

    mse = sse / num_samples
    reproj_err = np.sqrt(mse)

    sse, num_samples = vm.compute_mono_3d_err_handeye(ids,
                                                      object_points,
                                                      image_points,
                                                      camera_matrix,
                                                      camera_distortion,
                                                      device_tracking_array,
                                                      model_tracking_array,
                                                      handeye_matrix,
                                                      pattern2marker_matrix)

    mse = sse / num_samples
    recon_err = np.sqrt(mse)

    return reproj_err, recon_err, handeye_matrix, pattern2marker_matrix


#pylint:disable=too-many-arguments
def stereo_handeye_calibration(l2r_rmat: np.ndarray,
                               l2r_tvec: np.ndarray,
                               left_ids: List,
                               left_object_points: List,
                               left_image_points: List,
                               right_ids: List,
                               right_image_points: List,
                               left_camera_matrix: np.ndarray,
                               left_camera_distortion: np.ndarray,
                               right_camera_matrix: np.ndarray,
                               right_camera_distortion: np.ndarray,
                               device_tracking_array: List,
                               calibration_tracking_array: List,
                               left_rvecs: List[np.ndarray],
                               left_tvecs: List[np.ndarray],
                               right_rvecs: List[np.ndarray],
                               right_tvecs: List[np.ndarray],
                               quat_model2hand_array: List,
                               trans_model2hand_array: List):
    """Wrapper around handeye calibration functions and reprojection /
    reconstruction error metrics.

    :param l2r_rmat: [3x3] ndarray, rotation for l2r transform
    :type l2r_rmat: np.ndarray
    :param l2r_tvec: [3x1] ndarray, translation for l2r transform
    :type l2r_tvec: np.ndarray
    :param left_ids: Vector of ndarrays containing integer point ids.
    :type left_ids: List
    :param left_object_points: Vector of Vector of 1x3 of type float32
    :type left_object_points: List
    :param left_image_points: Vector of Vector of 1x2 of type float32
    :type left_image_points: List
    :param right_ids: Vector of ndarrays containing integer point ids.
    :type right_ids: List
    :param right_image_points: Vector of Vector of 1x3 of type float32
    :type right_image_points: List
    :param left_camera_matrix: Camera intrinsic matrix
    :type left_camera_matrix: np.ndarray
    :param left_camera_distortion: Camera distortion coefficients
    :type left_camera_distortion: np.ndarray
    :param right_camera_matrix: Camera intrinsic matrix
    :type right_camera_matrix: np.ndarray
    :param right_camera_distortion: Camera distortion coefficients
    :type right_camera_distortion: np.ndarray
    :param device_tracking_array: Tracking data for camera (hand)
    :type device_tracking_array: List
    :param calibration_tracking_array: Tracking data for calibration target
    :type calibration_tracking_array: List
    :param left_rvecs: Vector of 3x1 ndarray, Rodrigues rotations for each
    camera
    :type left_rvecs: List[np.ndarray]
    :param left_tvecs: Vector of [3x1] ndarray, translations for each camera
    :type left_tvecs: List[np.ndarray]
    :param right_rvecs: Vector of 3x1 ndarray, Rodrigues rotations for each
    camera
    :type right_rvecs: List[np.ndarray]
    :param right_tvecs: Vector of [3x1] ndarray, translations for each camera
    :type right_tvecs: List[np.ndarray]
    :param quat_model2hand_array: Array of model to hand quaternions
    :type quat_model2hand_array: List
    :param trans_model2hand_array: Array of model to hand translaions
    :type trans_model2hand_array: List
    :return: Reprojection error, reconstruction error, left handeye matrix,
    left pattern to marker matrix, right handeye, right pattern to marker
    :rtype: float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    """
    # Do calibration
    left_handeye_matrix, left_pattern2marker_matrix =  \
        he.handeye_calibration(left_rvecs, left_tvecs, quat_model2hand_array,
                               trans_model2hand_array)

    right_handeye_matrix, right_pattern2marker_matrix =  \
        he.handeye_calibration(right_rvecs, right_tvecs, quat_model2hand_array,
                               trans_model2hand_array)

    # Filter common image points
    minimum_points = 10
    _, common_object_pts, common_l_image_pts, common_r_image_pts = \
    vu.filter_common_points_all_images(
        left_ids, left_object_points, left_image_points,
        right_ids, right_image_points,
        minimum_points)

    sse, num_samples = vm.compute_stereo_2d_err_handeye(
        common_object_pts,
        common_l_image_pts,
        left_camera_matrix, left_camera_distortion,
        common_r_image_pts,
        right_camera_matrix, right_camera_distortion,
        device_tracking_array, calibration_tracking_array,
        left_handeye_matrix, left_pattern2marker_matrix,
        right_handeye_matrix, right_pattern2marker_matrix
    )

    mse = sse / num_samples
    reproj_err = np.sqrt(mse)

    sse, num_samples = vm.compute_stereo_3d_err_handeye(
        l2r_rmat, l2r_tvec,
        common_object_pts,
        common_l_image_pts,
        left_camera_matrix, left_camera_distortion,
        common_r_image_pts,
        right_camera_matrix, right_camera_distortion,
        device_tracking_array, calibration_tracking_array,
        left_handeye_matrix, left_pattern2marker_matrix,
    )
    mse = sse / num_samples
    recon_err = np.sqrt(mse)

    return reproj_err, recon_err, \
        left_handeye_matrix, left_pattern2marker_matrix, \
        right_handeye_matrix, right_pattern2marker_matrix
