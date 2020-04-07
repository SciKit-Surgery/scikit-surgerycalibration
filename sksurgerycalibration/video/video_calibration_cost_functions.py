# -*- coding: utf-8 -*-

import numpy as np
import cv2
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_metrics as vm


def _project_stereo_points_cost_function(x_0,
                                         left_object_points,
                                         left_image_points,
                                         left_distortion,
                                         right_object_points,
                                         right_image_points,
                                         right_distortion):
    """
    Private method to compute RMSE cost function.
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

    rmse = vm.compute_stereo_rms_projection_error(l2r_rmat,
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
    return rmse
