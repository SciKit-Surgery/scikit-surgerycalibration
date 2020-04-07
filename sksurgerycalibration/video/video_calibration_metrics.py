# -*- coding: utf-8 -*-

import logging
import numpy as np
import cv2
import sksurgerycore.transforms.matrix as mu
import sksurgerycalibration.video.video_calibration_utils as vu

LOGGER = logging.getLogger(__name__)


def compute_stereo_rms_projection_error(l2r_rmat,
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
    Function to compute the combined stereo RMS re-projection error.

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
    :return:
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
            + 3*len(left_image_points[i]) \
            + 3*len(right_image_points[i])

    rmse = np.sqrt((lse + rse) / number_of_samples)

    LOGGER.debug("Stereo RMS reprojection: left sse=%s, right sse=%s, rms=%s",
                 str(lse), str(rse), str(rmse))
    return rmse
