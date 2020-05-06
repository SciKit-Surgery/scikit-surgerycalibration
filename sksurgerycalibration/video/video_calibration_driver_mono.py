# -*- coding: utf-8 -*-

""" Class to do stateful video calibration of a mono camera. """

import copy
import logging
import numpy as np
import cv2
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

        if image_points.shape[0] >= self.minimum_points_per_frame:

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
            cm.compute_mono_reconstruction_err(
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
        # pylint: disable=too-many-locals, consider-using-enumerate,
        # pylint: disable=invalid-name, unused-variable
        proj_err, recon_err, param_copy = self.calibrate(flags=flags)
        cached_images = copy.deepcopy(self.video_data.images_array)

        for i in range(0, number_of_iterations):
            images = copy.deepcopy(cached_images)
            self.video_data.reinit()
            for j in range(0, len(images)):
                undistorted = cv2.undistort(
                    images[j],
                    self.calibration_params.camera_matrix,
                    self.calibration_params.dist_coeffs,
                    self.calibration_params.camera_matrix
                    )
                ids, obj_pts, img_pts = \
                    self.point_detector.get_points(undistorted)
                common_points = cu.match_points_by_id(ids, img_pts,
                                                      reference_ids,
                                                      reference_image_points)
                homography, _ = \
                    cv2.findHomography(common_points[0:, 0:2],
                                       common_points[0:, 2:4])
                warped = cv2.warpPerspective(undistorted,
                                             homography,
                                             reference_image_size)

                ids, obj_pts, img_pts = self.point_detector.get_points(warped)

                # Map pts back to original space.
                inverted_points = \
                    cv2.perspectiveTransform(
                        img_pts.astype(np.float32).reshape(-1, 1, 2),
                        np.linalg.inv(homography))
                inverted_points = inverted_points.reshape(-1, 2)

                distorted_pts = np.zeros(inverted_points.shape)
                number_of_points = inverted_points.shape[0]

                # Now have to map undistorted points back to distorted points
                for counter in range(number_of_points):

                    # Distort point to match original input image.
                    relative_x = \
                        (inverted_points[counter][0]
                         - self.calibration_params.camera_matrix[0][2]) \
                        / self.calibration_params.camera_matrix[0][0]
                    relative_y = \
                        (inverted_points[counter][1]
                         - self.calibration_params.camera_matrix[1][2]) \
                        / self.calibration_params.camera_matrix[1][1]
                    r2 = relative_x * relative_x + relative_y * relative_y
                    radial = (
                        1
                        + self.calibration_params.dist_coeffs[0][0]
                        * r2
                        + self.calibration_params.dist_coeffs[0][1]
                        * r2 * r2
                        + self.calibration_params.dist_coeffs[0][4]
                        * r2 * r2 * r2
                        )
                    distorted_x = relative_x * radial
                    distorted_y = relative_y * radial

                    distorted_x = distorted_x + (
                        2 * self.calibration_params.dist_coeffs[0][2]
                        * relative_x * relative_y
                        + self.calibration_params.dist_coeffs[0][3]
                        * (r2 + 2 * relative_x * relative_x))

                    distorted_y = distorted_y + (
                        self.calibration_params.dist_coeffs[0][2]
                        * (r2 + 2 * relative_y * relative_y)
                        + 2 * self.calibration_params.dist_coeffs[0][3]
                        * relative_x * relative_y)

                    distorted_x = \
                        distorted_x * \
                        self.calibration_params.camera_matrix[0][0] \
                        + self.calibration_params.camera_matrix[0][2]
                    distorted_y = \
                        distorted_y * \
                        self.calibration_params.camera_matrix[1][1] \
                        + self.calibration_params.camera_matrix[1][2]

                    distorted_pts[counter][0] = distorted_x
                    distorted_pts[counter][1] = distorted_y

                ids, image_points, object_points = \
                    cu.convert_pd_to_opencv(ids,
                                            obj_pts,
                                            distorted_pts)

                self.video_data.push(images[j],
                                     ids,
                                     object_points,
                                     image_points)

            proj_err, recon_err, param_copy = self.calibrate(flags=flags)
            print("Matt: " + str(proj_err) + ":" + str(recon_err))

        return proj_err, recon_err, param_copy
