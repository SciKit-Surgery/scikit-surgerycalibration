# -*- coding: utf-8 -*-

""" Class to do stateful video calibration of a mono camera. """

import copy
import logging
import numpy as np
import sksurgeryimage.calibration.point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_base as vdb
import sksurgerycalibration.video.video_calibration_data as cd
import sksurgerycalibration.video.video_calibration_params as cp
import sksurgerycalibration.video.video_calibration_metrics as cm
import sksurgerycalibration.video.video_calibration_utils as cu
import sksurgerycalibration.video.video_calibration_wrapper as vc

LOGGER = logging.getLogger(__name__)


class MonoVideoCalibrationDriver(vdb.BaseVideoCalibrationDriver):
    """ Class to do stateful video calibration of a mono camera. """
    def __init__(self,
                 point_detector: pd.PointDetector,
                 minimum_points_per_frame: int
                 ):
        """
        Stateful class for mono video calibration.

        :param point_detector: Class derived from PointDetector
        :param minimum_points_per_frame: Minimum number to accept frame
        """
        super(MonoVideoCalibrationDriver, self).\
            __init__(point_detector,
                     minimum_points_per_frame)

        # Create data holders, and parameter holders, specific to Mono.
        calibration_data = cd.MonoVideoData()
        calibration_params = cp.MonoCalibrationParams()

        # Pass them to base class, so base class can access them.
        self._init_internal(calibration_data, calibration_params)

    def grab_data(self,
                  image,
                  device_tracking=None,
                  calibration_object_tracking=None):
        """
        Extracts points, by passing it to the PointDetector.

        This will throw various exceptions if the input data is invalid,
        but will return empty arrays if no points were detected.
        So, no points is not an error. Its an expected condition.

        :param image: RGB image.
        :param device_tracking: transformation for the tracked device
        :param calibration_object_tracking: transformation of tracked
        calibration object
        :return: The number of points grabbed.
        """
        number_of_points = 0

        ids, object_points, image_points = \
            self.point_detector.get_points(image)

        if ids is not None and ids.shape[0] >= self.minimum_points_per_frame:

            ids, image_points, object_points = \
                cu.convert_pd_to_opencv(ids,
                                        object_points,
                                        image_points)

            self.video_data.push(image,
                                 ids,
                                 object_points,
                                 image_points)

            self.tracking_data.push(device_tracking,
                                    calibration_object_tracking)

            number_of_points = image_points.shape[0]

        LOGGER.info("Grabbed: Returning %s points.", str(number_of_points))
        return number_of_points

    def calibrate(self, flags=0):
        """
        Do the video calibration.

        This returns RMS projection error, which is a common metric, but also,
        the reconstruction error. If we have N views, we can take successive
        pairs of views, triangulate points, and see how well they match the
        model. Ideally, both metrics should be small.

        :param flags: OpenCV flags, eg. cv2.CALIB_FIX_ASPECT_RATIO
        :return: RMS projection, reconstruction error.
        """
        proj_err, camera_matrix, dist_coeffs, rvecs, tvecs = \
            vc.mono_video_calibration(
                self.video_data.object_points_arrays,
                self.video_data.image_points_arrays,
                (self.video_data.images_array[0].shape[1],
                 self.video_data.images_array[0].shape[0]),
                flags
            )

        sse, num_samples = \
            cm.compute_mono_3d_err(
                self.video_data.ids_arrays,
                self.video_data.object_points_arrays,
                self.video_data.image_points_arrays,
                rvecs,
                tvecs,
                camera_matrix,
                dist_coeffs
                )
        recon_err = np.sqrt(sse / num_samples)

        self.calibration_params.set_data(camera_matrix,
                                         dist_coeffs,
                                         rvecs,
                                         tvecs)

        LOGGER.info("Calibrated: proj_err=%s, recon_err=%s.",
                    str(proj_err), str(recon_err))
        return proj_err, recon_err, copy.deepcopy(self.calibration_params)

    def iterative_calibration(self,
                              number_of_iterations: int,
                              reference_ids,
                              reference_image_points,
                              reference_image_size,
                              flags: int = 0):
        """
        Does iterative calibration, like Datta 2009.
        """
        proj_err, recon_err, param_copy = self.calibrate(flags=flags)
        cached_images = copy.deepcopy(self.video_data.images_array)

        for i in range(0, number_of_iterations):
            images = copy.deepcopy(cached_images)
            cu.detect_points_in_canonical_space(
                self.point_detector,
                self.minimum_points_per_frame,
                self.video_data,
                images,
                self.calibration_params.camera_matrix,
                self.calibration_params.dist_coeffs,
                reference_ids,
                reference_image_points,
                reference_image_size)
            proj_err, recon_err, param_copy = self.calibrate(flags=flags)
            LOGGER.info("Iterative calibration: %s: proj_err=%s, recon_err=%s.",
                        str(i), str(proj_err), str(recon_err))

        return proj_err, recon_err, param_copy

    def handeye_calibration(self):
        """Do handeye calibration.

        This returns RMS projection error, which is a common metric, but also,
        the reconstruction error. If we have N views, we can take successive
        pairs of views, triangulate points, and see how well they match the
        model. Ideally, both metrics should be small.

        :return: reprojection, reconstruction error
        :rtype: float, float
        """
        self.tracking_data.set_model2hand_arrays()

        proj_err, recon_err, handeye, pattern2marker = \
            vc.mono_handeye_calibration(
                self.video_data.object_points_arrays,
                self.video_data.image_points_arrays,
                self.video_data.ids_arrays,
                self.calibration_params.camera_matrix,
                self.calibration_params.dist_coeffs,
                self.tracking_data.device_tracking_array,
                self.tracking_data.calibration_tracking_array,
                self.calibration_params.rvecs,
                self.calibration_params.tvecs,
                self.tracking_data.quat_model2hand_array,
                self.tracking_data.trans_model2hand_array
            )

        self.calibration_params.set_handeye(handeye, pattern2marker)

        return proj_err, recon_err
