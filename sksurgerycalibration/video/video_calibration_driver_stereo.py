# -*- coding: utf-8 -*-

""" Class to do stateful video calibration of a stereo camera. """

import copy
import logging
import cv2
import sksurgeryimage.calibration.point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_base as vdb
import sksurgerycalibration.video.video_calibration_data as cd
import sksurgerycalibration.video.video_calibration_params as cp
import sksurgerycalibration.video.video_calibration_utils as cu
import sksurgerycalibration.video.video_calibration_wrapper as vc

LOGGER = logging.getLogger(__name__)


class StereoVideoCalibrationDriver(vdb.BaseVideoCalibrationDriver):
    """
    Class to do stateful video calibration of a stereo camera.
    """
    def __init__(self,
                 left_point_detector: pd.PointDetector,
                 right_point_detector: pd.PointDetector,
                 minimum_points_per_frame: int
                 ):
        """
        Stateful class for stereo video calibration.

        :param left_point_detector: Class derived from PointDetector
        :param right_point_detector: Class derived from PointDetector
        :param minimum_points_per_frame: Minimum number to accept frame
        """
        super().__init__(minimum_points_per_frame)

        self.left_point_detector = left_point_detector
        self.right_point_detector = right_point_detector

        # Create data holders, and parameter holders, specific to Stereo.
        calibration_data = cd.StereoVideoData()
        calibration_params = cp.StereoCalibrationParams()

        # Pass them to base class, so base class can access them.
        self._init_internal(calibration_data, calibration_params)

    def grab_data(self,
                  left_image,
                  right_image,
                  device_tracking=None,
                  calibration_object_tracking=None
                  ):
        """
        Extracts points, by passing it to the PointDetector.

        This will throw various exceptions if the input data is invalid,
        but will return empty arrays if no points were detected.
        So, no points is not an error. Its an expected condition.

        :param left_image: BGR image.
        :param right_image: BGR image.
        :param device_tracking: transformation for the tracked device
        :param calibration_object_tracking: transformation of tracked
        calibration object
        :return: The number of points grabbed.
        """
        number_left = 0
        number_right = 0

        # This can return None's if none are found.
        left_ids, left_object_points, left_image_points = \
            self.left_point_detector.get_points(left_image)

        if left_ids is not None:
            number_left = left_ids.shape[0]

        if number_left >= self.minimum_points_per_frame:

            right_ids, right_object_points, right_image_points = \
                self.right_point_detector.get_points(right_image)

            if right_ids is not None:
                number_right = right_ids.shape[0]

            if number_right >= self.minimum_points_per_frame:

                left_ids, left_image_points, left_object_points = \
                    cu.convert_pd_to_opencv(left_ids,
                                            left_object_points,
                                            left_image_points)

                right_ids, right_image_points, right_object_points = \
                    cu.convert_pd_to_opencv(right_ids,
                                            right_object_points,
                                            right_image_points)

                self.video_data.push(left_image,
                                     left_ids,
                                     left_object_points,
                                     left_image_points,
                                     right_image,
                                     right_ids,
                                     right_object_points,
                                     right_image_points)

                self.tracking_data.push(device_tracking,
                                        calibration_object_tracking)

        number_of_points = number_left + number_right

        LOGGER.info("Grabbed: Returning (%s+%s)=%s points.",
                    str(number_left),
                    str(number_right),
                    str(number_of_points))

        return number_left, number_right

    def calibrate(self,
                  flags=cv2.CALIB_USE_INTRINSIC_GUESS,
                  override_left_intrinsics=None,
                  override_left_distortion=None,
                  override_right_intrinsics=None,
                  override_right_distortion=None,
                  override_l2r_rmat=None,
                  override_l2r_tvec=None
                  ):
        """
        Do the stereo video calibration.

        This returns RMS projection error, which is a common metric, but also,
        the reconstruction / triangulation error.

        :param flags: OpenCV flags, eg. cv2.CALIB_FIX_INTRINSIC
        :return: RMS projection, reconstruction error.
        """
        s_reproj, s_recon, \
            l_c, l_d, l_rvecs, l_tvecs, \
            r_c, r_d, r_rvecs, r_tvecs, \
            l2r_r, l2r_t, \
            essential, fundamental \
            = vc.stereo_video_calibration(
                self.video_data.left_data.ids_arrays,
                self.video_data.left_data.object_points_arrays,
                self.video_data.left_data.image_points_arrays,
                self.video_data.right_data.ids_arrays,
                self.video_data.right_data.object_points_arrays,
                self.video_data.right_data.image_points_arrays,
                (self.video_data.left_data.images_array[0].shape[1],
                 self.video_data.left_data.images_array[0].shape[0]),
                flags,
                override_left_intrinsics,
                override_left_distortion,
                override_right_intrinsics,
                override_right_distortion,
                override_l2r_rmat,
                override_l2r_tvec
            )

        self.calibration_params.set_data(l_c, l_d, l_rvecs, l_tvecs, r_c, r_d,
                                         r_rvecs, r_tvecs, l2r_r, l2r_t,
                                         essential, fundamental)

        LOGGER.info("Calibrated: proj_err=%s, recon_err=%s.",
                    str(s_reproj), str(s_recon))
        return s_reproj, s_recon, copy.deepcopy(self.calibration_params)

    # pylint:disable=too-many-arguments
    def iterative_calibration(self,
                              number_of_iterations: int,
                              reference_ids,
                              reference_image_points,
                              reference_image_size,
                              flags: int = cv2.CALIB_USE_INTRINSIC_GUESS
                              ):
        """
        Does iterative calibration, like Datta 2009.
        """
        proj_err, recon_err, param_copy = self.calibrate(flags=flags)
        cached_left_images = copy.deepcopy(
            self.video_data.left_data.images_array)
        cached_right_images = copy.deepcopy(
            self.video_data.right_data.images_array)

        for i in range(0, number_of_iterations):
            left_images = copy.deepcopy(cached_left_images)
            right_images = copy.deepcopy(cached_right_images)

            cu.detect_points_in_stereo_canonical_space(
                self.left_point_detector,
                self.right_point_detector,
                self.minimum_points_per_frame,
                self.video_data.left_data,
                left_images,
                self.calibration_params.left_params.camera_matrix,
                self.calibration_params.left_params.dist_coeffs,
                self.video_data.right_data,
                right_images,
                self.calibration_params.right_params.camera_matrix,
                self.calibration_params.right_params.dist_coeffs,
                reference_ids,
                reference_image_points,
                reference_image_size)

            proj_err, recon_err, param_copy = \
                self.calibrate(flags)

            LOGGER.info("Iterative calibration: %s: proj_err=%s, recon_err=%s.",
                        str(i), str(proj_err), str(recon_err))
        return proj_err, recon_err, param_copy

    def handeye_calibration(self):
        """
        Do handeye calibration.

        This returns RMS projection error, which is a common metric, but also,
        the reconstruction / triangulation error.

        :return: reprojection, reconstruction error
        :rtype: float, float
        """

        # This combines chessboardmarker(model)-to-tracker and
        # device-to-tracker to get a fixed transformation.
        self.tracking_data.set_model2hand_arrays()

        # So, this is optimising the device hand-eye and the
        # chessboard-to-chessboard marker transformation.
        proj_err, recon_err, l_handeye, l_pattern2marker, \
            r_handeye, r_pattern2marker = \
                vc.stereo_handeye_calibration(
                    self.calibration_params.l2r_rmat,
                    self.calibration_params.l2r_tvec,
                    self.video_data.left_data.ids_arrays,
                    self.video_data.left_data.object_points_arrays,
                    self.video_data.left_data.image_points_arrays,
                    self.video_data.right_data.ids_arrays,
                    self.video_data.right_data.image_points_arrays,
                    self.calibration_params.left_params.camera_matrix,
                    self.calibration_params.left_params.dist_coeffs,
                    self.calibration_params.right_params.camera_matrix,
                    self.calibration_params.right_params.dist_coeffs,
                    self.tracking_data.device_tracking_array,
                    self.tracking_data.calibration_tracking_array,
                    self.calibration_params.left_params.rvecs,
                    self.calibration_params.left_params.tvecs,
                    self.calibration_params.right_params.rvecs,
                    self.calibration_params.right_params.tvecs,
                    self.tracking_data.quat_model2hand_array,
                    self.tracking_data.trans_model2hand_array
                    )

        self.calibration_params.left_params.set_handeye(
            l_handeye, l_pattern2marker)

        self.calibration_params.right_params.set_handeye(
            r_handeye, r_pattern2marker)

        return proj_err, recon_err, copy.deepcopy(self.calibration_params)
