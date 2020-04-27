# -*- coding: utf-8 -*-

import copy
import logging
import cv2
import sksurgeryimage.processing.point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_base as vdb
import sksurgerycalibration.video.video_calibration_data as cd
import sksurgerycalibration.video.video_calibration_params as cp
import sksurgerycalibration.video.video_calibration_utils as cu
import sksurgerycalibration.video.video_calibration as vc

LOGGER = logging.getLogger(__name__)


class StereoVideoCalibrationDriver(vdb.BaseVideoCalibrationDriver):

    def __init__(self,
                 point_detector: pd.PointDetector,
                 minimum_points_per_frame: int
                 ):
        """
        Stateful class for stereo video calibration.

        :param point_detector: Class derived from PointDetector
        :param minimum_points_per_frame: Minimum number to accept frame
        """
        super(StereoVideoCalibrationDriver, self).\
            __init__(point_detector,
                     minimum_points_per_frame)

        # Create data holders, and parameter holders, specific to Mono.
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

        :param left_image: RGB image.
        :param right_image: RGB image.
        :param device_tracking: transformation for the tracked device
        :param calibration_object_tracking: transformation of tracked calibration object
        :return: The number of points grabbed.
        """
        number_of_points = 0

        left_ids, left_object_points, left_image_points = \
            self.point_detector.get_points(left_image)

        if left_image_points.shape[0] >= self.minimum_points_per_frame:

            right_ids, right_object_points, right_image_points = \
                self.point_detector.get_points(right_image)

            if left_image_points.shape[0] >= self.minimum_points_per_frame:

                left_ids, left_image_points, left_object_points = \
                    cu.convert_point_detector_to_opencv(left_ids,
                                                        left_object_points,
                                                        left_image_points)

                right_ids, right_image_points, right_object_points = \
                    cu.convert_point_detector_to_opencv(right_ids,
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

                number_of_points = \
                    left_image_points.shape[0] + \
                    right_image_points.shape[0]

        LOGGER.info("Grabbed: Returning %s + %s = %s points.",
                    str(left_image_points),
                    str(right_image_points),
                    str(number_of_points))

        return number_of_points

    def calibrate(self, flags=cv2.CALIB_USE_INTRINSIC_GUESS):
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
                flags)

        self.calibration_params.set_data(l_c, l_d, l_rvecs, l_tvecs, r_c, r_d, r_rvecs, r_tvecs, l2r_r, l2r_t, essential, fundamental)

        LOGGER.info("Calibrated: proj_err=%s, recon_err=%s.",
                    str(s_reproj), str(s_recon))
        return s_reproj, s_recon, copy.deepcopy(self.calibration_params)
