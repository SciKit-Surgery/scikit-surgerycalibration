# -*- coding: utf-8 -*-

""" Containers for video calibration parameters. """

import copy
import numpy as np
import sksurgerycore.transforms.matrix as sksm


class MonoCalibrationParams:
    """
    Holds a set of intrinsic and extrinsic camera parameters for 1 camera.
    """
    def __init__(self):
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.reinit()

    def reinit(self):
        """
        Resets data, to identity/empty arrays etc.
        """
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((1, 5))
        self.rvecs = []
        self.tvecs = []

    def set_data(self, camera_matrix, dist_coeffs, rvecs, tvecs):
        """
        Stores the provided parameters, by taking a copy.
        """
        self.camera_matrix = copy.deepcopy(camera_matrix)
        self.dist_coeffs = copy.deepcopy(dist_coeffs)
        self.rvecs = copy.deepcopy(rvecs)
        self.tvecs = copy.deepcopy(tvecs)


class StereoCalibrationParams:
    """
    Holds a pair of MonoCalibrationParams, and the left-to-right transform.
    """
    def __init__(self):
        self.left_params = None
        self.right_params = None
        self.l2r_rmat = None
        self.l2r_tvec = None

    def reinit(self):
        """
        Resets data, to identity/empty arrays etc.
        """
        self.left_params = MonoCalibrationParams()
        self.right_params = MonoCalibrationParams()
        self.l2r_rmat = np.eye(3)
        self.l2r_tvec = np.zeros((3, 1))

    def set_data(self,
                 left_cam_matrix, left_dist_coeffs, left_rvecs, left_tvecs,
                 right_cam_matrix, right_dist_coeffs, right_rvecs, right_tvecs,
                 l2r_rmat, l2r_tvec
                 ):
        """
        Stores the provided parameters, by taking a copy.
        """
        self.left_params.set_data(left_cam_matrix,
                                  left_dist_coeffs,
                                  left_rvecs,
                                  left_tvecs)
        self.right_params.set_data(right_cam_matrix,
                                   right_dist_coeffs,
                                   right_rvecs,
                                   right_tvecs)
        self.l2r_rmat = copy.deepcopy(l2r_rmat)
        self.l2r_tvec = copy.deepcopy(l2r_tvec)

    def get_l2r_as_4x4(self):
        """
        Extracts the left-to-right transform as 4x4 matrix.
        """
        return sksm.construct_rigid_transformation(self.l2r_rmat, self.l2r_tvec)

