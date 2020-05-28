# -*- coding: utf-8 -*-

""" Containers for video calibration parameters. """

import os
import copy
import numpy as np
import sksurgerycore.transforms.matrix as sksm
import sksurgerycalibration.video.video_calibration_utils as sksu
import sksurgerycalibration.video.video_calibration_io as sksio


class BaseCalibrationParams:
    """
    Constructor, no member variables, so just a pure virtual interface.

    Not really necessary if you rely on duck-typing, but at least
    it shows the intention of what derived classes should implement,
    and means we can use this base class to type check against.
    """
    def __init__(self):
        return

    def reinit(self):
        """ Used to clear, re-initialise all member variables. """
        raise NotImplementedError("Derived classes should implement this.")

    def save_data(self, dir_name: str, file_prefix: str):
        """ Writes all contained data to disk. """
        raise NotImplementedError("Derived classes should implement this.")

    def load_data(self, dir_name: str, file_prefix: str):
        """ Loads all contained data from disk. """
        raise NotImplementedError("Derived classes should implement this.")


class MonoCalibrationParams(BaseCalibrationParams):
    """
    Holds a set of intrinsic and extrinsic camera parameters for 1 camera.
    """
    def __init__(self):
        super(MonoCalibrationParams, self).__init__()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.handeye_matrix = None
        self.pattern2marker_matrix = None
        self.reinit()

    def reinit(self):
        """
        Resets data, to identity/empty arrays etc.
        """
        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros((1, 5))
        self.rvecs = []
        self.tvecs = []
        self.handeye_matrix = np.eye(4)
        self.pattern2marker_matrix = np.eye(4)

    def set_data(self, camera_matrix, dist_coeffs, rvecs, tvecs):
        """
        Stores the provided parameters, by taking a copy.
        """
        self.camera_matrix = copy.deepcopy(camera_matrix)
        self.dist_coeffs = copy.deepcopy(dist_coeffs)
        self.rvecs = copy.deepcopy(rvecs)
        self.tvecs = copy.deepcopy(tvecs)

    def set_handeye(self, handeye_matrix, pattern2marker_matrix):
        """
        Stores the provided parameters, by taking a copy.
        """
        self.handeye_matrix = copy.deepcopy(handeye_matrix)
        self.pattern2marker_matrix = copy.deepcopy(pattern2marker_matrix)

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves calibration parameters to a directory.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        intrinsics_file = sksio.get_intrinsics_file_name(dir_name,
                                                         file_prefix)
        np.savetxt(intrinsics_file, self.camera_matrix, fmt='%.8f')

        dist_coeff_file = sksio.get_distortion_file_name(dir_name,
                                                         file_prefix)
        np.savetxt(dist_coeff_file, self.dist_coeffs, fmt='%.8f')
        for i in enumerate(self.rvecs):
            extrinsics_file = sksio.get_extrinsics_file_name(dir_name,
                                                             file_prefix,
                                                             i[0])
            extrinsics = sksu.extrinsic_vecs_to_matrix(self.rvecs[i[0]],
                                                       self.tvecs[i[0]])
            np.savetxt(extrinsics_file, extrinsics, fmt='%.8f')

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

        intrinsics_file = sksio.get_intrinsics_file_name(dir_name,
                                                         file_prefix)
        self.camera_matrix = np.loadtxt(intrinsics_file)

        dist_coeff_file = sksio.get_distortion_file_name(dir_name,
                                                         file_prefix)
        self.dist_coeffs = np.loadtxt(dist_coeff_file)

        extrinsic_files = sksio.get_extrinsic_file_names(dir_name,
                                                         file_prefix)
        for file in extrinsic_files:
            extrinsics = np.loadtxt(file)
            rvec, tvec = sksu.extrinsic_matrix_to_vecs(extrinsics)
            self.rvecs.append(rvec)
            self.tvecs.append(tvec)


class StereoCalibrationParams(BaseCalibrationParams):
    """
    Holds a pair of MonoCalibrationParams, and the left-to-right transform.
    """
    def __init__(self):
        super(StereoCalibrationParams, self).__init__()
        self.left_params = None
        self.right_params = None
        self.l2r_rmat = None
        self.l2r_tvec = None
        self.essential = None
        self.fundamental = None
        self.reinit()

    def reinit(self):
        """
        Resets data, to identity/empty arrays etc.
        """
        self.left_params = MonoCalibrationParams()
        self.right_params = MonoCalibrationParams()
        self.l2r_rmat = np.eye(3)
        self.l2r_tvec = np.zeros((3, 1))
        self.essential = np.eye(3)
        self.fundamental = np.eye(3)

    # pylint: disable=too-many-arguments
    def set_data(self,
                 left_cam_matrix, left_dist_coeffs, left_rvecs, left_tvecs,
                 right_cam_matrix, right_dist_coeffs, right_rvecs, right_tvecs,
                 l2r_rmat, l2r_tvec, essential, fundamental
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
        self.essential = copy.deepcopy(essential)
        self.fundamental = copy.deepcopy(fundamental)

    def set_handeye(self, left_handeye_matrix, left_pattern2marker_matrix,
                    right_handeye_matrix, right_pattern2marker_matrix):
        """
        Call the left/right set_handeye methods.
        """
        self.left_params.set_handeye(
            left_handeye_matrix, left_pattern2marker_matrix)

        self.right_params.set_handeye(
            right_handeye_matrix, right_pattern2marker_matrix)

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
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        left_prefix = sksio.get_left_prefix(file_prefix)
        self.left_params.save_data(dir_name, left_prefix)
        right_prefix = sksio.get_right_prefix(file_prefix)
        self.right_params.save_data(dir_name, right_prefix)

        l2r_file = sksio.get_l2r_file_name(dir_name, file_prefix)
        np.savetxt(l2r_file, self.get_l2r_as_4x4(), fmt='%.8f')

        ess_file = sksio.get_essential_matrix_file_name(dir_name,
                                                        file_prefix)
        np.savetxt(ess_file, self.essential, fmt='%.8f')

        fun_file = sksio.get_fundamental_matrix_file_name(dir_name,
                                                          file_prefix)
        np.savetxt(fun_file, self.fundamental, fmt='%.8f')

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

        left_prefix = sksio.get_left_prefix(file_prefix)
        self.left_params.load_data(dir_name, left_prefix)
        right_prefix = sksio.get_right_prefix(file_prefix)
        self.right_params.load_data(dir_name, right_prefix)

        l2r_file = sksio.get_l2r_file_name(dir_name, file_prefix)
        stereo_ext = np.loadtxt(l2r_file)

        self.l2r_rmat = stereo_ext[0:3, 0:3]
        tmp = stereo_ext[0:3, 3]
        self.l2r_tvec[0][0] = tmp[0]
        self.l2r_tvec[1][0] = tmp[1]
        self.l2r_tvec[2][0] = tmp[2]

        ess_file = sksio.get_essential_matrix_file_name(dir_name,
                                                        file_prefix)
        self.essential = np.loadtxt(ess_file)

        fun_file = sksio.get_fundamental_matrix_file_name(dir_name,
                                                          file_prefix)
        self.fundamental = np.loadtxt(fun_file)
