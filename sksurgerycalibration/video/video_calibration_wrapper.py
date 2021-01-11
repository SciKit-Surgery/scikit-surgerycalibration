# -*- coding: utf-8 -*-

""" Video Calibration functions, that wrap OpenCV functions mainly. """

import logging
import copy
from typing import List
import numpy as np
import cv2
from scipy.optimize import least_squares
import sksurgerycore.transforms.matrix as skcm
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm
import sksurgerycalibration.video.video_calibration_hand_eye as he
import sksurgerycalibration.video.video_calibration_cost_functions as vcf

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


# pylint: disable=too-many-locals, too-many-arguments
def stereo_video_calibration(left_ids,
                             left_object_points,
                             left_image_points,
                             right_ids,
                             right_object_points,
                             right_image_points,
                             image_size,
                             flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                             override_left_intrinsics=None,
                             override_left_distortion=None,
                             override_right_intrinsics=None,
                             override_right_distortion=None,
                             override_l2r_rmat=None,
                             override_l2r_tvec=None
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

    # We only do override if all override params are specified.
    # pylint:disable=too-many-boolean-expressions
    do_override = False
    if override_left_intrinsics is not None \
        and override_left_distortion is not None \
        and override_right_intrinsics is not None \
        and override_right_distortion is not None \
        and override_l2r_rmat is not None \
        and override_l2r_tvec is not None:

        do_override = True

        l_c = override_left_intrinsics
        l_d = override_left_distortion
        r_c = override_right_intrinsics
        r_d = override_right_distortion

    number_of_frames = len(left_object_points)

    l_rvecs = []
    l_tvecs = []
    r_rvecs = []
    r_tvecs = []

    if do_override:

        for i in range(0, number_of_frames):

            _, rvecs, tvecs = cv2.solvePnP(
                left_object_points[i],
                left_image_points[i],
                l_c,
                l_d)
            l_rvecs.append(rvecs)
            l_tvecs.append(tvecs)

            _, rvecs, tvecs = cv2.solvePnP(
                right_object_points[i],
                right_image_points[i],
                r_c,
                r_d)
            r_rvecs.append(rvecs)
            r_tvecs.append(tvecs)

    else:

        _, l_c, l_d, l_rvecs, l_tvecs \
            = cv2.calibrateCamera(left_object_points,
                                  left_image_points,
                                  image_size,
                                  None, None)

        _, r_c, r_d, r_rvecs, r_tvecs \
            = cv2.calibrateCamera(right_object_points,
                                  right_image_points,
                                  image_size,
                                  None, None)

    # For stereo, OpenCV needs common points.
    _, common_object_points, common_left_image_points, \
        common_right_image_points \
        = vu.filter_common_points_all_images(left_ids,
                                             left_object_points,
                                             left_image_points,
                                             right_ids,
                                             right_image_points, 10)

    if do_override:

        # Do OpenCV stereo calibration, using override intrinsics,
        # just so we can get the essential and fundamental matrix out.
        _, l_c, l_d, r_c, r_d, \
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

        l2r_r = override_l2r_rmat
        l2r_t = override_l2r_tvec

        assert np.allclose(l_c, override_left_intrinsics)
        assert np.allclose(l_d, override_left_distortion)
        assert np.allclose(r_c, override_right_intrinsics)
        assert np.allclose(r_d, override_right_distortion)

    else:

        # Do OpenCV stereo calibration, using intrinsics from OpenCV mono.
        _, l_c, l_d, r_c, r_d, \
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
        _, l_c, l_d, r_c, r_d, \
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

    if do_override:

        # Stereo calibration is hard for a laparoscope.
        # In clinical practice, the data may be way too variable.
        # For stereo scopes, they are often fixed focus,
        # i.e. fixed intrinsics, and fixed stereo.
        # So, we may prefer to just do the best possible calibration
        # in the lab, and then keep those values constant.
        # But we then would still want to optimise the camera extrinsics
        # as the camera poses directly affect the hand-eye calibration.

        _, l_rvecs, l_tvecs, \
            = stereo_calibration_extrinsics(
                common_object_points,
                common_left_image_points,
                common_right_image_points,
                l_rvecs,
                l_tvecs,
                l_c,
                l_d,
                r_c,
                r_d,
                l2r_r,
                l2r_t
            )

    else:

        # Normal OpenCV stereo calibration optimises intrinsics,
        # distortion, and stereo parameters, but doesn't output pose.
        # So here, we recompute the left camera pose.
        for i in range(0, number_of_frames):
            _, l_rvecs[i], l_tvecs[i] = cv2.solvePnP(
                common_object_points[i],
                common_left_image_points[i],
                l_c,
                l_d)

    # Here, we are computing the right hand side rvecs and tvecs
    # given the new left hand side rvecs, tvecs and the l2r.
    left_to_right = skcm.construct_rigid_transformation(l2r_r, l2r_t)
    for i in range(0, number_of_frames):
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

    LOGGER.info("Stereo Calib: proj=%s, recon=%s",
                str(s_reproj), str(s_recon))

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
    # Do calibration, using Guofang's method on each left/right channel.
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

    # Now try another optimisation.
    # Have one version of the pattern2marker matrix and handeye matrix,
    # and also optimise the tracking information to match.
    number_of_frames = len(common_object_pts)

    number_of_parameters = (number_of_frames * 6) + 12
    x_0 = np.zeros(number_of_parameters)

    for i in range(0, number_of_frames):
        m2h = np.linalg.inv(device_tracking_array[i]) \
              @ calibration_tracking_array[i]
        m2h_rvec, m2h_tvec = vu.extrinsic_matrix_to_vecs(m2h)
        x_0[i * 6 + 0] = m2h_rvec[0][0]
        x_0[i * 6 + 1] = m2h_rvec[1][0]
        x_0[i * 6 + 2] = m2h_rvec[2][0]
        x_0[i * 6 + 3] = m2h_tvec[0][0]
        x_0[i * 6 + 4] = m2h_tvec[1][0]
        x_0[i * 6 + 5] = m2h_tvec[2][0]

    p2m_rvec, p2m_tvec = vu.extrinsic_matrix_to_vecs(left_pattern2marker_matrix)
    x_0[number_of_frames * 6 + 0] = p2m_rvec[0][0]
    x_0[number_of_frames * 6 + 1] = p2m_rvec[1][0]
    x_0[number_of_frames * 6 + 2] = p2m_rvec[2][0]
    x_0[number_of_frames * 6 + 3] = p2m_tvec[0][0]
    x_0[number_of_frames * 6 + 4] = p2m_tvec[1][0]
    x_0[number_of_frames * 6 + 5] = p2m_tvec[2][0]

    h2e_rvec, h2e_tvec = vu.extrinsic_matrix_to_vecs(left_handeye_matrix)
    x_0[(number_of_frames + 1) * 6 + 0] = h2e_rvec[0][0]
    x_0[(number_of_frames + 1) * 6 + 1] = h2e_rvec[1][0]
    x_0[(number_of_frames + 1) * 6 + 2] = h2e_rvec[2][0]
    x_0[(number_of_frames + 1) * 6 + 3] = h2e_tvec[0][0]
    x_0[(number_of_frames + 1) * 6 + 4] = h2e_tvec[1][0]
    x_0[(number_of_frames + 1) * 6 + 5] = h2e_tvec[2][0]

    res = least_squares(vcf.stereo_handeye_error, x_0,
                        args=(common_object_pts,
                              common_l_image_pts,
                              common_r_image_pts,
                              left_camera_matrix,
                              left_camera_distortion,
                              right_camera_matrix,
                              right_camera_distortion,
                              l2r_rmat,
                              l2r_tvec),
                        method='lm',
                        x_scale='jac',
                        verbose=0)

    LOGGER.info("Stereo Handeye Re-Calibration: status=%s", str(res.status))
    LOGGER.info("Stereo Handeye Re-Calibration: success=%s", str(res.success))
    LOGGER.info("Stereo Handeye Re-Calibration: msg=%s", str(res.message))

    # Extract data from result object.
    x_1 = res.x
    tmp_rvec = np.zeros((3, 1))
    tmp_rvec[0][0] = x_1[number_of_frames * 6 + 0]
    tmp_rvec[1][0] = x_1[number_of_frames * 6 + 1]
    tmp_rvec[2][0] = x_1[number_of_frames * 6 + 2]
    tmp_tvec = np.zeros((3, 1))
    tmp_tvec[0][0] = x_1[number_of_frames * 6 + 3]
    tmp_tvec[1][0] = x_1[number_of_frames * 6 + 4]
    tmp_tvec[2][0] = x_1[number_of_frames * 6 + 5]
    left_pattern2marker_matrix = vu.extrinsic_vecs_to_matrix(tmp_rvec, tmp_tvec)

    tmp_rvec[0][0] = x_1[(number_of_frames + 1) * 6 + 0]
    tmp_rvec[1][0] = x_1[(number_of_frames + 1) * 6 + 1]
    tmp_rvec[2][0] = x_1[(number_of_frames + 1) * 6 + 2]

    tmp_tvec[0][0] = x_1[(number_of_frames + 1) * 6 + 3]
    tmp_tvec[1][0] = x_1[(number_of_frames + 1) * 6 + 4]
    tmp_tvec[2][0] = x_1[(number_of_frames + 1) * 6 + 5]
    left_handeye_matrix = vu.extrinsic_vecs_to_matrix(tmp_rvec, tmp_tvec)

    l2r_matrix = skcm.construct_rigid_transformation(l2r_rmat, l2r_tvec)
    right_handeye_matrix = l2r_matrix @ left_handeye_matrix
    right_pattern2marker_matrix = copy.deepcopy(left_pattern2marker_matrix)

    # Remember that we have optimised tracking matrices.
    # So computation of error statistics should include these new positions.
    dummy_dt_array = []
    dummy_ct_array = []

    for i in range(0, number_of_frames):

        tmp_rvec[0][0] = x_1[i * 6 + 0]
        tmp_rvec[1][0] = x_1[i * 6 + 1]
        tmp_rvec[2][0] = x_1[i * 6 + 2]

        tmp_tvec[0][0] = x_1[i * 6 + 3]
        tmp_tvec[1][0] = x_1[i * 6 + 4]
        tmp_tvec[2][0] = x_1[i * 6 + 5]

        calib_track_mat = vu.extrinsic_vecs_to_matrix(tmp_rvec, tmp_tvec)
        device_track_mat = np.eye(4)

        dummy_dt_array.append(device_track_mat)
        dummy_ct_array.append(calib_track_mat)

    # Now compute some output statistics.
    sse, num_samples = vm.compute_stereo_2d_err_handeye(
        common_object_pts,
        common_l_image_pts,
        left_camera_matrix,
        left_camera_distortion,
        common_r_image_pts,
        right_camera_matrix,
        right_camera_distortion,
        dummy_dt_array,
        dummy_ct_array,
        left_handeye_matrix,
        left_pattern2marker_matrix,
        right_handeye_matrix,
        right_pattern2marker_matrix
    )
    mse = sse / num_samples
    reproj_err = np.sqrt(mse)

    sse, num_samples = vm.compute_stereo_3d_err_handeye(
        l2r_rmat,
        l2r_tvec,
        common_object_pts,
        common_l_image_pts,
        left_camera_matrix,
        left_camera_distortion,
        common_r_image_pts,
        right_camera_matrix,
        right_camera_distortion,
        dummy_dt_array,
        dummy_ct_array,
        left_handeye_matrix,
        left_pattern2marker_matrix,
    )
    mse = sse / num_samples
    recon_err = np.sqrt(mse)

    return reproj_err, recon_err, \
        left_handeye_matrix, left_pattern2marker_matrix, \
        right_handeye_matrix, right_pattern2marker_matrix


def stereo_calibration_extrinsics(common_object_points,
                                  common_left_image_points,
                                  common_right_image_points,
                                  l_rvecs,
                                  l_tvecs,
                                  override_left_intrinsics,
                                  override_left_distortion,
                                  override_right_intrinsics,
                                  override_right_distortion,
                                  override_l2r_rmat,
                                  override_l2r_tvec):
    """
    Simply re-optimises the extrinsic parameters.
    :return: error, l_rvecs, l_tvecs
    """
    number_of_frames = len(common_object_points)
    number_of_parameters = 6 * number_of_frames
    x_0 = np.zeros(number_of_parameters)
    for i in range(0, number_of_frames):
        x_0[i * 6 + 0] = l_rvecs[i][0]
        x_0[i * 6 + 1] = l_rvecs[i][1]
        x_0[i * 6 + 2] = l_rvecs[i][2]
        x_0[i * 6 + 3] = l_tvecs[i][0]
        x_0[i * 6 + 4] = l_tvecs[i][1]
        x_0[i * 6 + 5] = l_tvecs[i][2]

    res = least_squares(vcf.stereo_2d_error, x_0,
                        args=(common_object_points,
                              common_left_image_points,
                              common_right_image_points,
                              override_left_intrinsics,
                              override_left_distortion,
                              override_right_intrinsics,
                              override_right_distortion,
                              override_l2r_rmat,
                              override_l2r_tvec),
                        method='lm',
                        x_scale='jac',
                        verbose=0)

    LOGGER.info("Stereo Re-Calibration: status=%s", str(res.status))
    LOGGER.info("Stereo Re-Calibration: success=%s", str(res.success))
    LOGGER.info("Stereo Re-Calibration: msg=%s", str(res.message))

    x_1 = res.x
    for i in range(0, number_of_frames):
        l_rvecs[i][0] = x_1[i * 6 + 0]
        l_rvecs[i][1] = x_1[i * 6 + 1]
        l_rvecs[i][2] = x_1[i * 6 + 2]
        l_tvecs[i][0] = x_1[i * 6 + 3]
        l_tvecs[i][1] = x_1[i * 6 + 4]
        l_tvecs[i][2] = x_1[i * 6 + 5]

    return res.fun, l_rvecs, l_tvecs
