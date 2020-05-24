# -*- coding: utf-8 -*-

""" Video calibration metrics. """

# pylint: disable=too-many-arguments

import logging
import numpy as np
import cv2
import sksurgeryopencvpython as cvpy
import sksurgerycore.transforms.matrix as mu
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
                          left_tvecs):
    """
    Function to compute stereo SSE re-projection error, over multiple views.

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
    :return: SSE re-reprojection error, number_samples
    """
    left_to_right = mu.construct_rigid_transformation(l2r_rmat, l2r_tvec)

    lse = 0
    rse = 0
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

        diff_left = left_image_points[i] - projected_left

        lse = lse + np.sum(np.square(diff_left))

        diff_right = right_image_points[i] - projected_right
        rse = rse + np.sum(np.square(diff_right))
        number_of_samples = number_of_samples \
            + len(left_image_points[i]) \
            + len(right_image_points[i])

    LOGGER.debug("Stereo RMS reprojection: left sse=%s, right sse=%s, num=%s",
                 str(lse), str(rse), str(number_of_samples))
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
                            left_tvecs):
    """
    Function to compute stereo SSE reconstruction error over multiple views.

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
    :return: SSE re-reprojection error, number_samples
    """
    sse = 0
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

        # Triangulate using OpenCV
#        l2r_mat = mu.construct_rigid_transformation(l2r_rmat, l2r_tvec)
#        p_l = np.zeros((3, 4))
#        p_l[:, :-1] = left_camera_matrix
#        p_r = np.zeros((3, 4))
#        p_r[:, :-1] = right_camera_matrix
#        p_l = np.matmul(p_l, np.eye(4))
#        p_r = np.matmul(p_r, l2r_mat)
#        triangulated_cv = cv2.triangulatePoints(p_l,
#                                                p_r,
#                                                left_undistorted,
#                                                right_undistorted)

        # convert from Mx1x2 to Mx2
        left_undistorted = np.reshape(left_undistorted, (-1, 2))
        right_undistorted = np.reshape(right_undistorted, (-1, 2))

        image_points = np.zeros((left_undistorted.shape[0], 4))
        image_points[:, 0:2] = left_undistorted
        image_points[:, 2:4] = right_undistorted

        triangulated = cvpy.triangulate_points_using_hartley(
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
        squared = np.square(diff)
        sum_square = np.sum(squared)
        sse = sse + sum_square
        number_of_samples = number_of_samples + len(common_left_image_points[i])

    LOGGER.debug("Stereo RMS reconstruction: sse=%s, num=%s",
                 str(sse), str(number_of_samples))
    return sse, number_of_samples


def compute_mono_2d_err(object_points,
                        image_points,
                        rvecs,
                        tvecs,
                        camera_matrix,
                        distortion):
    """
    Function to compute stereo RMS reconstruction error over multiple views.

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

    for i in range(0, number_of_frames):

        projected, _ = cv2.projectPoints(object_points[i],
                                         rvecs[i],
                                         tvecs[i],
                                         camera_matrix,
                                         distortion)

        diff = image_points[i] - projected
        squared = np.square(diff)
        sum_square = np.sum(squared)
        sse = sse + sum_square
        number_of_samples = number_of_samples + len(image_points[i])

    LOGGER.debug("Mono RMS reprojection: sse=%s, num=%s",
                 str(sse), str(number_of_samples))
    return sse, number_of_samples


def compute_mono_3d_err(ids,
                                    object_points,
                                    image_points,
                                    rvecs,
                                    tvecs,
                                    camera_matrix,
                                    distortion):
    """
    Function to compute stereo RMS reconstruction error over multiple views.

    Here, to triangulate, we take the i^th camera as left camera, and
    the i+1^th camera as the right camera, compute l2r, and triangulate.

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

def compute_stereo_2d_err_handeye(common_object_points,
                                 left_image_points,
                                 left_camera_matrix,
                                 left_distortion,
                                 right_image_points,
                                 right_camera_matrix,
                                 right_distortion,
                                 hand_tracking_array,
                                 model_tracking_array,
                                 left_handeye_matrix,
                                 left_pattern2marker_matrix,
                                 right_handeye_matrix,
                                 right_pattern2marker_matrix):

    lse = 0
    rse = 0
    number_of_samples = 0
    number_of_frames = len(common_object_points)

    for i in range(number_of_frames):

        camera_to_pattern = \
            left_handeye_matrix @ np.linalg.inv(hand_tracking_array[i]) @ \
                 model_tracking_array[i] @ left_pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(camera_to_pattern)

        projected_left, _ = cv2.projectPoints(common_object_points[i],
                                               rvec,
                                               tvec,
                                               left_camera_matrix,
                                               left_distortion)

        diff_left = left_image_points[i] - projected_left
        lse = lse + np.sum(np.square(diff_left))

        number_of_samples += len(left_image_points[i])

        world_to_right_camera = \
            right_handeye_matrix @ np.linalg.inv(hand_tracking_array[i]) @ \
                 model_tracking_array[i] @ right_pattern2marker_matrix

        rvec, tvec = vu.extrinsic_matrix_to_vecs(world_to_right_camera)

        projected_right, _ = cv2.projectPoints(common_object_points[i],
                                               rvec,
                                               tvec,
                                               right_camera_matrix,
                                               right_distortion)

        diff_right = right_image_points[i] - projected_right
        rse = rse + np.sum(np.square(diff_right))
        
        number_of_samples += len(right_image_points[i])
        
    return lse + rse, number_of_samples

def compute_stereo_3d_err_handeye(l2r_rmat,
                            l2r_tvec,
                            common_object_points,
                            common_left_image_points,
                            left_camera_matrix,
                            left_distortion,
                            common_right_image_points,
                            right_camera_matrix,
                            right_distortion,
                            hand_tracking_array,
                            model_tracking_array,
                            left_handeye_matrix,
                            left_pattern2marker_matrix,
                            right_handeye_matrix,
                            right_pattern2marker_matrix):
    sse = 0
    number_of_samples = 0
    number_of_frames = len(common_object_points)

    for i in range(number_of_frames):

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

        triangulated = cvpy.triangulate_points_using_hartley(
            image_points,
            left_camera_matrix,
            right_camera_matrix,
            l2r_rmat,
            l2r_tvec)

        # Triangulated points, are with respect to left camera.
        # Need to map back to model (chessboard space) for comparison.
        # Or, map chessboard points into left-camera space. Chose former

        left_camera_to_pattern = \
            np.linalg.inv(left_handeye_matrix) @ \
            hand_tracking_array[i] @ \
            np.linalg.inv(model_tracking_array[i]) @ \
            np.linalg.inv(left_pattern2marker_matrix)

        rvec, tvec = vu.extrinsic_matrix_to_vecs(left_camera_to_pattern)
        rotated = triangulated @ rvec
        translated = rotated + np.transpose(tvec) # uses broadcasting
        transformed = translated

        # Now compute squared error
        diff = triangulated - model_points
        squared = np.square(diff)
        sum_square = np.sum(squared)
        sse = sse + sum_square
        number_of_samples = number_of_samples + len(common_left_image_points[i])

    LOGGER.debug("Stereo RMS reconstruction: sse=%s, num=%s",
                 str(sse), str(number_of_samples))

    return sse, number_of_samples
