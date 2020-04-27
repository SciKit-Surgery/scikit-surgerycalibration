# -*- coding: utf-8 -*-

import copy
import logging
import numpy as np
import sksurgeryimage.processing.point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_base as vdb
import sksurgerycalibration.video.video_calibration_data as cd
import sksurgerycalibration.video.video_calibration_params as cp
import sksurgerycalibration.video.video_calibration_metrics as cm
import sksurgerycalibration.video.video_calibration_utils as cu
import sksurgerycalibration.video.video_calibration as vc

LOGGER = logging.getLogger(__name__)


class MonoVideoCalibrationDriver(vdb.BaseVideoCalibrationDriver):

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
        self.calibration_data = cd.MonoVideoData()
        self.calibration_params = cp.MonoCalibrationParams()

        # Pass them to base class, so base class can access them.
        self._init_internal(self.calibration_data, self.calibration_params)

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
        :param calibration_object_tracking: transformation of tracked calibration object
        :return: The number of points grabbed.
        """
        number_of_points = 0

        ids, object_points, image_points = \
            self.point_detector.get_points(image)

        if image_points.shape[0] >= self.minimum_points_per_frame:

            ids, image_points, object_points = \
                cu.convert_point_detector_to_opencv(ids,
                                                    object_points,
                                                    image_points)

            self.calibration_data.push(image,
                                       ids,
                                       object_points,
                                       image_points,
                                       device_tracking,
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
                self.calibration_data.object_points_arrays,
                self.calibration_data.image_points_arrays,
                (self.calibration_data.images_array[0].shape[1],
                 self.calibration_data.images_array[0].shape[0]),
                flags
            )

        sse, num_samples = \
            cm.compute_mono_reconstruction_err(self.calibration_data.ids_arrays,
                                               self.calibration_data.object_points_arrays,
                                               self.calibration_data.image_points_arrays,
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
