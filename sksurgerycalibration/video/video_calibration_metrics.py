# -*- coding: utf-8 -*-

"""
Video calibration metrics, used in cost functions for optimisation,
and as measures of error generally.
"""

# pylint: disable=too-many-arguments

import logging
from typing import List

import cv2
import numpy as np
import sksurgerycore.transforms.matrix as mu

import sksurgerycalibration.algorithms.triangulate as at
import sksurgerycalibration.video.video_calibration_utils as vu

LOGGER = logging.getLogger(__name__)


def compute_stereo_2d_err(l2r_rmat,
                          l2r_tvec,
                          left_object_points,
                          left_image_points,
                          left_camera_matrix,
                          left_distortion,
                          right_object_points,
                          right_image_points,
                          right_camera_matrix,
                          right_distortion,
                          left_rvecs,
                          left_tvecs,
                          return_residuals=False):
    """
    Function to compute stereo re-projection error (SSE),
    or residuals, over multiple views.

    :param l2r_rmat: [3x3] ndarray, rotation for l2r transform
    :param l2r_tvec: [3x1] ndarray, translation for l2r transform
    :param left_object_points: Vector of Vector of 1x3 of type float32
    :param left_image_points: Vector of Vector of 1x2 of type float32
    :param left_camera_matrix: [3x3] ndarray
    :param left_distortion: [1x5] ndarray
    :param right_object_points: Vector of Vector of 1x3 of type float32
    :param right_image_points: Vector of Vector of 1x2 of type float32
    :param right_camera_matrix: [3x3] ndarray
    :param right_distortion: [1x5] ndarray
    :param left_rvecs: Vector of [3x1] ndarray, Rodrigues rotations, left camera
    :param left_tvecs: Vector of [3x1] ndarray, translations, left camera
    :param return_residuals: if True returns vector of residuals for LM,
    otherwise, returns SSE.
    :return: SSE, number_samples OR residuals
    """
    left_to_right = mu.construct_rigid_transformation(l2r_rmat, l2r_tvec)

    lse = 0
    rse = 0
    residuals = []
    number_of_samples = 0
    number_of_frames = len(left_object_points)

    for i in range(0, number_of_frames):

        projected_left, _ = cv2.projectPoints(left_object_points[i],
                                              left_rvecs[i],
                                              left_tvecs[i],
                                              left_camera_matrix,
                                              left_distortion)

        world_to_left_camera = vu.extrinsic_vecs_to_matrix(left_rvecs[i],
                                                           left_tvecs[i])

        world_to_right_camera = np.matmul(left_to_right, world_to_left_camera)

        rvec, tvec = vu.extrinsic_matrix_to_vecs(world_to_right_camera)

        projected_right, _ = cv2.projectPoints(right_object_points[i],
                                               rvec,
                                               tvec,
                                               right_camera_matrix,
                                               right_distortion)

        number_of_samples = number_of_samples \
            + len(left_image_points[i]) \
            + len(right_image_points[i])

        diff_left = left_image_points[i] - projected_left
        diff_right = right_image_points[i] - projected_right

        if return_residuals:
            residuals.append(diff_left.reshape((-1)))
            residuals.append(diff_right.reshape((-1)))
        else:
            lse = lse + np.sum(np.square(diff_left))
            rse = rse + np.sum(np.square(diff_right))

    if return_residuals:
        return np.hstack(residuals)

    return lse + rse, number_of_samples


def compute_stereo_3d_error(l2r_rmat,
                            l2r_tvec,
                            common_object_points,
                            common_left_image_points,
                            left_camera_matrix,
                            left_distortion,
                            common_right_image_points,
                            right_camera_matrix,
                            right_distortion,
                            left_rvecs,
                            left_tvecs,
                            return_residuals=False):
    """
    Function to compute stereo reconstruction error (SSE),
    or residuals over multiple views.

    :param l2r_rmat: [3x3] ndarray, rotation for l2r transform
    :param l2r_tvec: [3x1] ndarray, translation for l2r transform
    :param common_object_points: Vector of Vector of 1x3 of type float32
    :param common_left_image_points: Vector of Vector of 1x2 of type float32
    :param left_camera_matrix: [3x3] ndarray
    :param left_distortion: [1x5] ndarray
    :param common_right_image_points: Vector of Vector of 1x2 of type float32
    :param right_camera_matrix: [3x3] ndarray
    :param right_distortion: [1x5] ndarray
    :param left_rvecs: Vector of [3x1] ndarray, Rodrigues rotations, left camera
    :param left_tvecs: Vector of [3x1] ndarray, translations, left camera
    :param return_residuals: if True returns vector of residuals for LM,
    otherwise, returns SSE.
    :return: SSE re-reprojection error, number_samples
    """
    sse = 0
    residuals = []
    number_of_samples = 0
    number_of_frames = len(common_object_points)

    for i in range(0, number_of_frames):

        model_points = np.reshape(common_object_points[i], (-1, 3))

        left_undistorted = \
            cv2.undistortPoints(common_left_image_points[i],
                                left_camera_matrix,
                                left_distortion, None, left_camera_matrix)
        right_undistorted = \
            cv2.undistortPoints(common_right_image_points[i],
                                right_camera_matrix,
                                right_distortion, None, right_camera_matrix)

        # convert from Mx1x2 to Mx2
        left_undistorted = np.reshape(left_undistorted, (-1, 2))
        right_undistorted = np.reshape(right_undistorted, (-1, 2))

        image_points = np.zeros((left_undistorted.shape[0], 4))
        image_points[:, 0:2] = left_undistorted
        image_points[:, 2:4] = right_undistorted

        triangulated = at.triangulate_points_hartley(
            image_points,
            left_camera_matrix,
            right_camera_matrix,
            l2r_rmat,
            l2r_tvec)

        # Triangulated points, are with respect to left camera.
        # Need to map back to model (chessboard space) for comparison.
        # Or, map chessboard points into left-camera space. Chose latter.
        rmat = (cv2.Rodrigues(left_rvecs[i]))[0]
        rotated = np.matmul(rmat, np.transpose(model_points))
        translated = rotated + left_tvecs[i]  # uses broadcasting
        transformed = np.transpose(translated)

        # Now compute squared error
        diff = triangulated - transformed

        if return_residuals:
            residuals.append(diff.reshape((-1)))
        else:
            squared = np.square(diff)
            sum_square = np.sum(squared)
            sse = sse + sum_square
            number_of_samples = number_of_samples \
                                + len(common_left_image_points[i])

    if return_residuals:
        return np.hstack(residuals)

    LOGGER.debug("Stereo RMS reconstruction: sse=%s, num=%s",
                 str(sse), str(number_of_samples))
    return sse, number_of_samples


def compute_mono_2d_err(object_points,
                        image_points,
                        rvecs,
                        tvecs,
                        camera_matrix,
                        distortion,
                        return_residuals=False):
    """
    Function to compute mono reprojection (SSE) error,
    or residuals over multiple views of a mono camera.

    :param object_points: Vector of Vector of 1x3 of type float32
    :param image_points: Vector of Vector of 1x2 of type float32
    :param rvecs: Vector of [3x1] ndarray, Rodrigues rotations for each camera
    :param tvecs: Vector of [3x1] ndarray, translations for each camera
    :param camera_matrix: [3x3] ndarray
    :param distortion: [1x5] ndarray
    :param return_residuals: If True returns a big array of residuals for LM.
    :return: SSE re-reprojection error, number_samples OR residuals
    """
    sse = 0
    residuals = []
    number_of_samples = 0
    number_of_frames = len(object_points)

    for i in range(0, number_of_frames):

        projected, _ = cv2.projectPoints(object_points[i],
                                         rvecs[i],
                                         tvecs[i],
                                         camera_matrix,
                                         distortion)

        diff = image_points[i] - projected

        if return_residuals:
            residuals.append(diff.reshape((-1)))
        else:
            sse = sse + np.sum(np.square(diff))

        number_of_samples = number_of_samples + len(image_points[i])

    LOGGER.debug("Mono RMS reprojection: sse=%s, num=%s",
                 str(sse), str(number_of_samples))

    if return_residuals:
        return np.hstack(residuals)

    return sse, number_of_samples


def compute_mono_3d_err(ids,
                        object_points,
                        image_points,
                        rvecs,
                        tvecs,
                        camera_matrix,
                        distortion):
    """
    Function to compute mono reconstruction error (SSE) over multiple views.

    Here, to triangulate, we take the i^th camera as left camera, and
    the i+1^th camera as the right camera, compute l2r, and triangulate.

    Note: This may fail if the difference between two successive views
    is too large, and there are not enough common points.

    :param ids: Vector of ndarray of integer point ids
    :param object_points: Vector of Vector of 1x3 of type float32
    :param image_points: Vector of Vector of 1x2 of type float32
    :param rvecs: Vector of [3x1] ndarray, Rodrigues rotations for each camera
    :param tvecs: Vector of [3x1] ndarray, translations for each camera
    :param camera_matrix: [3x3] ndarray
    :param distortion: [1x5] ndarray
    :return: SSE re-reprojection error, number_samples
    """

    sse = 0
    number_of_samples = 0
    number_of_frames = len(object_points)

    # We are going to triangulate between a frame and the next frame.
    for i in range(0, number_of_frames):

        j = i + 1
        if j == number_of_frames:
            j = 0

        _, common_object_points, common_left_image_points, \
            common_right_image_points = \
            vu.filter_common_points_per_image(ids[i],
                                              object_points[i],
                                              image_points[i],
                                              ids[j],
                                              image_points[j],
                                              10)

        left_camera_to_world = vu.extrinsic_vecs_to_matrix(rvecs[i],
                                                           tvecs[i])

        right_camera_to_world = vu.extrinsic_vecs_to_matrix(rvecs[j],
                                                            tvecs[j])
        left_to_right = np.matmul(right_camera_to_world,
                                  np.linalg.inv(left_camera_to_world))

        l2r_rmat = left_to_right[0:3, 0:3]
        l2r_tvec = left_to_right[0:3, 3]

        c_obj = [common_object_points]
        c_li = [common_left_image_points]
        c_ri = [common_right_image_points]
        r_v = [rvecs[i]]
        t_v = [tvecs[i]]
        err, samp = compute_stereo_3d_error(l2r_rmat,
                                            l2r_tvec,
                                            c_obj,
                                            c_li,
                                            camera_matrix,
                                            distortion,
                                            c_ri,
                                            camera_matrix,
                                            distortion,
                                            r_v,
                                            t_v)
        sse = sse + err
        number_of_samples = number_of_samples + samp

    LOGGER.debug("Mono RMS reconstruction: sse=%s, num=%s",
                 str(sse), str(number_of_samples))

    return sse, number_of_samples


def compute_mono_2d_err_handeye(model_points: List,
                                image_points: List,
                                camera_matrix: np.ndarray,
                                camera_distortion: np.ndarray,
                                hand_tracking_array: List,
                                model_tracking_array: List,
                                handeye_matrix: np.ndarray,
                                pattern2marker_matrix: np.ndarray):
    """
    Function to compute mono reprojection error (SSE), mapping
    from the calibration pattern coordinate system to the
    camera coordinate system, via tracking matrices and
    hand-eye calibration.

    :param model_points: Vector of Vector of 1x3 float32
    :type model_points: List
    :param image_points: Vector of Vector of 1x2 float32
    :type image_points: List
    :param camera_matrix: Camera intrinsic matrix
    :type camera_matrix: np.ndarray
    :param camera_distortion: Camera distortion coefficients
    :type camera_distortion: np.ndarray
    :param hand_tracking_array:
    Vector of 4x4 tracking matrices for camera (hand)
    :type hand_tracking_array: List
    :param model_tracking_array:
    Vector of 4x4 tracking matrices for calibration model
    :type model_tracking_array: List
    :param handeye_matrix: Handeye matrix
    :type handeye_matrix: np.ndarray
    :param pattern2marker_matrix: Pattern to marker matrix
    :type pattern2marker_matrix: np.ndarray
    :return: SSE reprojection error, number of samples
    :rtype: float, float
    """

    number_of_frames = len(model_points)

    rvecs = []
    tvecs = []

    # Construct rvec/tvec array taking into account handeye calibration.
    # Then the rest of the calculation can use 'normal' compute_mono_2d_err()
    for i in range(number_of_frames):
        pattern_to_camera = \
            handeye_matrix @ np.linalg.inv(hand_tracking_array[i]) @ \
            model_tracking_array[i] @ pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(pattern_to_camera)
        rvecs.append(rvec)
        tvecs.append(tvec)

    sse, number_of_samples = compute_mono_2d_err(model_points,
                                                 image_points,
                                                 rvecs,
                                                 tvecs,
                                                 camera_matrix,
                                                 camera_distortion)

    LOGGER.debug("Mono Handeye RMS Reprojection: sse=%s, num=%s",
                 str(sse), str(number_of_samples))

    return sse, number_of_samples


def compute_mono_3d_err_handeye(ids: List,
                                model_points: List,
                                image_points: List,
                                camera_matrix: np.ndarray,
                                camera_distortion: np.ndarray,
                                hand_tracking_array: List,
                                model_tracking_array: List,
                                handeye_matrix: np.ndarray,
                                pattern2marker_matrix: np.ndarray):
    """
    Function to compute mono reconstruction error (SSE). Calculates new
    rvec/tvec values for pattern_to_camera based on handeye calibration and
    then calls compute_mono_3d_err().

    :param ids: Vector of ndarray of integer point ids
    :type ids: List
    :param model_points: Vector of Vector of 1x3 float32
    :type model_points: List
    :param image_points: Vector of Vector of 1x2 float32
    :type image_points: List
    :param camera_matrix: Camera intrinsic matrix
    :type camera_matrix: np.ndarray
    :param camera_distortion: Camera distortion coefficients
    :type camera_distortion: np.ndarray
    :param hand_tracking_array:
    Vector of 4x4 tracking matrices for camera (hand)
    :type hand_tracking_array: List
    :param model_tracking_array:
    Vector of 4x4 tracking matrices for calibration model
    :type model_tracking_array: List
    :param handeye_matrix: Handeye matrix
    :type handeye_matrix: np.ndarray
    :param pattern2marker_matrix: Pattern to marker matrix
    :type pattern2marker_matrix: np.ndarray
    :return: SSE reprojection error, number of samples
    :rtype: float, float
    """

    number_of_frames = len(model_points)

    rvecs = []
    tvecs = []

    # Construct rvec/tvec array taking into account handeye calibration.
    # Then the rest of the calculation can use 'normal' compute_mono_3d_err()
    for i in range(number_of_frames):
        pattern_to_camera = \
            handeye_matrix @ np.linalg.inv(hand_tracking_array[i]) @ \
            model_tracking_array[i] @ pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(pattern_to_camera)
        rvecs.append(rvec)
        tvecs.append(tvec)

    sse, number_of_samples = compute_mono_3d_err(ids,
                                                 model_points,
                                                 image_points,
                                                 rvecs,
                                                 tvecs,
                                                 camera_matrix,
                                                 camera_distortion)

    return sse, number_of_samples


def compute_stereo_2d_err_handeye(common_object_points: List,
                                  left_image_points: List,
                                  left_camera_matrix: np.ndarray,
                                  left_distortion: np.ndarray,
                                  right_image_points: List,
                                  right_camera_matrix: np.ndarray,
                                  right_distortion: np.ndarray,
                                  hand_tracking_array: List,
                                  model_tracking_array: List,
                                  left_handeye_matrix: np.ndarray,
                                  left_pattern2marker_matrix: np.ndarray,
                                  right_handeye_matrix: np.ndarray,
                                  right_pattern2marker_matrix: np.ndarray):
    """
    Function to compute stereo reprojection error (SSE), taking into account
    handeye calibration.

    :param common_object_points: Vector of Vector of 1x3 float32
    :type common_object_points: List
    :param left_image_points: Vector of Vector of 1x2 float32
    :type left_image_points: List
    :param left_camera_matrix: Left camera matrix
    :type left_camera_matrix: np.ndarray
    :param left_distortion: Left camera distortion coefficients
    :type left_distortion: np.ndarray
    :param right_image_points: Vector of Vector of 1x2 float32
    :type right_image_points: List
    :param right_camera_matrix: Right camera matrix
    :type right_camera_matrix: np.ndarray
    :param right_distortion: Right camera distortion coefficients
    :type right_distortion: np.ndarray
    :param hand_tracking_array:
    Vector of 4x4 tracking matrices for camera (hand)
    :type hand_tracking_array: List
    :param model_tracking_array:
    Vector of 4x4 tracking matrices for calibration model
    :type model_tracking_array: List
    :param left_handeye_matrix: Left handeye transform matrix
    :type left_handeye_matrix: np.ndarray
    :param left_pattern2marker_matrix: Left pattern to marker transform matrix
    :type left_pattern2marker_matrix: np.ndarray
    :param right_handeye_matrix: Right handeye transform matrix
    :type right_handeye_matrix: np.ndarray
    :param right_pattern2marker_matrix: Right pattern to marker transform matrix
    :type right_pattern2marker_matrix: np.ndarray
    :return: SSE reprojection error, number of samples
    :rtype: float, float
    """

    lse, l_samples = compute_mono_2d_err_handeye(common_object_points,
                                                 left_image_points,
                                                 left_camera_matrix,
                                                 left_distortion,
                                                 hand_tracking_array,
                                                 model_tracking_array,
                                                 left_handeye_matrix,
                                                 left_pattern2marker_matrix)

    rse, r_samples = compute_mono_2d_err_handeye(common_object_points,
                                                 right_image_points,
                                                 right_camera_matrix,
                                                 right_distortion,
                                                 hand_tracking_array,
                                                 model_tracking_array,
                                                 right_handeye_matrix,
                                                 right_pattern2marker_matrix)

    return lse + rse, l_samples + r_samples


def compute_stereo_3d_err_handeye(l2r_rmat: np.ndarray,
                                  l2r_tvec: np.ndarray,
                                  common_object_points: List,
                                  common_left_image_points: List,
                                  left_camera_matrix: np.ndarray,
                                  left_distortion: np.ndarray,
                                  common_right_image_points: List,
                                  right_camera_matrix: np.ndarray,
                                  right_distortion: np.ndarray,
                                  hand_tracking_array: List,
                                  model_tracking_array: List,
                                  left_handeye_matrix: np.ndarray,
                                  left_pattern2marker_matrix: np.ndarray):

    """
    Function to compute stereo reconstruction error (SSE), taking into account
    handeye calibration.

    :param l2r_rmat: Rotation for l2r transform
    :type l2r_rmat: np.ndarray
    :param l2r_tvec: Translation for l2r transform
    :type l2r_tvec: np.ndarray
    :param common_object_points: Vector of Vector of 1x3 float32
    :type common_object_points: List
    :param common_left_image_points: Vector of Vector of 1x2 float32
    :type common_left_image_points: List
    :param left_camera_matrix: Left camera matrix
    :type left_camera_matrix: np.ndarray
    :param left_distortion: Left camera distortion coefficients
    :type left_distortion: np.ndarray
    :param common_right_image_points: Vector of Vector of 1x2 float32
    :type common_right_image_points: List
    :param right_camera_matrix: Right camera matrix
    :type right_camera_matrix: np.ndarray
    :param right_distortion: Right camera distortion coefficients
    :type right_distortion: np.ndarray
    :param hand_tracking_array:
    Vector of 4x4 tracking matrices for camera (hand)
    :type hand_tracking_array: List
    :param model_tracking_array:
    Vector of 4x4 tracking matrices for calibration model
    :type model_tracking_array: List
    :param left_handeye_matrix: Left handeye transform matrix
    :type left_handeye_matrix: np.ndarray
    :param left_pattern2marker_matrix: Left pattern to marker transform matrix
    :type left_pattern2marker_matrix: np.ndarray
    :return: SSE reconstruction error, number of samples
    :rtype: float, float
    """

    number_of_frames = len(common_object_points)
    left_rvecs = []
    left_tvecs = []

    # Construct rvec/tvec array taking into account handeye calibration.
    # Then the rest of the calculation can use 'normal' compute_stereo_3d_err()
    for i in range(number_of_frames):

        pattern_to_left_camera = \
            left_handeye_matrix @ np.linalg.inv(hand_tracking_array[i]) @ \
            model_tracking_array[i] @ left_pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(pattern_to_left_camera)
        left_rvecs.append(rvec)
        left_tvecs.append(tvec)

    sse, number_of_samples = compute_stereo_3d_error(l2r_rmat,
                                                     l2r_tvec,
                                                     common_object_points,
                                                     common_left_image_points,
                                                     left_camera_matrix,
                                                     left_distortion,
                                                     common_right_image_points,
                                                     right_camera_matrix,
                                                     right_distortion,
                                                     left_rvecs,
                                                     left_tvecs)

    LOGGER.debug("Stereo Handeye RMS reconstruction: sse=%s, num=%s",
                 str(sse), str(number_of_samples))

    return sse, number_of_samples
