# -*- coding: utf-8 -*-

""" Containers for video calibration parameters. """

import copy
import numpy as np
import sksurgerycore.transforms.matrix as sksm
import sksurgerycalibration.video.video_calibration_utils as sksu
import sksurgerycalibration.video.video_calibration_io as sksio


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

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves calibration parameters to a directory.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        intrinsics_file = sksio._get_intrinsics_file_name(dir_name,
                                                          file_prefix)
        np.savetxt(intrinsics_file, self.camera_matrix, fmt='%f')

        dist_coeff_file = sksio._get_distortion_file_name(dir_name,
                                                          file_prefix)
        np.savetxt(dist_coeff_file, self.dist_coeffs, fmt='%f')
        for i in enumerate(self.rvecs):
            extrinsics_file = sksio._get_extrinsics_file_name(dir_name,
                                                              file_prefix, i)
            extrinsics = sksu.extrinsic_vecs_to_matrix(self.rvecs[i],
                                                       self.tvecs[i])
            np.savetxt(extrinsics_file, extrinsics, fmt='%f')

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads calibration parameters from a directory.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        self.reinit()

        intrinsics_file = sksio._get_intrinsics_file_name(dir_name,
                                                          file_prefix)
        self.camera_matrix = np.loadtxt(intrinsics_file)

        dist_coeff_file = sksio._get_distortion_file_name(dir_name,
                                                          file_prefix)
        self.dist_coeffs = np.loadtxt(dist_coeff_file)

        extrinsic_files = sksio._get_extrinsic_file_names((dir_name,
                                                           file_prefix))
        for file in extrinsic_files:
            extrinsics = np.loadtxt(file)
            rvec, tvec = sksu.extrinsic_matrix_to_vecs(extrinsics)
            self.rvecs.append(rvec)
            self.tvecs.append(tvec)


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

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves calibration parameters to a directory.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        left_prefix = sksio._get_left_prefix(file_prefix)
        self.left_params.save_data(dir_name, left_prefix)
        right_prefix = sksio._get_right_prefix(file_prefix)
        self.left_params.save_data(dir_name, right_prefix)

        l2r_file = sksio._get_l2r_file_name(dir_name, file_prefix)
        np.savetxt(l2r_file, self.get_l2r_as_4x4())

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads calibration parameters from a directory.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        left_prefix = sksio._get_left_prefix(file_prefix)
        self.left_params.load_data(dir_name, left_prefix)
        right_prefix = sksio._get_right_prefix(file_prefix)
        self.right_params.load_data(dir_name, right_prefix)

        l2r_file = sksio._get_l2r_file_name(dir_name, file_prefix)
        stereo_ext = np.loadtxt(l2r_file)
        self.l2r_rmat = stereo_ext[0:3, 0:3]
        self.l2r_tvec = stereo_ext[0:3, 3]
