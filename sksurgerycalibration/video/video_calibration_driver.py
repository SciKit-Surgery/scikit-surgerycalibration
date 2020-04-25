# -*- coding: utf-8 -*-

import numpy as np
import sksurgeryimage.processing.point_detector as pd
import sksurgerycalibration.video.video_calibration_data as cd
import sksurgerycalibration.video.video_calibration_params as cp
import sksurgerycalibration.video.video_calibration_metrics as cm
import sksurgerycalibration.video.video_calibration as vc


def _convert_point_detector_to_opencv(object_points, image_points):
    dims = np.shape(image_points)
    image_points = np.reshape(image_points, (dims[0], 1, 2))
    image_points = image_points.astype(np.float32)
    object_points = np.reshape(object_points, (-1, 1, 3))
    object_points = object_points.astype(np.float32)
    return image_points, object_points


class MonoVideoCalibration:

    def __init__(self,
                 point_detector: pd.PointDetector,
                 minimum_points_per_frame: int
                 ):
        """
        Stateful class for mono video calibration.

        This class expects calling code to decide how many images are
        required to calibrate, and also, when to call reinit.

        The PointDetector is passed in using Dependency Injection.
        So, the PointDetector can be anything, like chessboards, ArUco,
        CharUco etc.

        This does mean that the underlying code can handle variable numbers
        of points in each view. OpenCV calibration code does this anyway.

        :param point_detector: Class derived from PointDetector
        :param minimum_points_per_frame: Minimum number to accept frame
        """
        self.point_detector = point_detector
        self.calibration_data = cd.MonoVideoData()
        self.calibration_params = cp.MonoCalibrationParams()
        self.minimum_points_per_frame = minimum_points_per_frame

    def reinit(self):
        """
        Resets the object, which means, removes stored calibration data
        and reset the calibration parameters to identity/zero.
        """
        self.calibration_data.reinit()
        self.calibration_params.reinit()

    def grab_data(self, image):
        """
        Extracts points, by passing it to the PointDetector.

        This will throw various exceptions if the input data is invalid,
        but will return empty arrays if no points were detected.
        So, no points is not an error. Its an expected condition.

        :param image: RGB image.
        :return: The number of points grabbed.
        """
        number_of_points = 0

        ids, object_points, image_points = \
            self.point_detector.get_points(image)

        if image_points.shape[0] > self.minimum_points_per_frame:

            image_points, object_points = \
                _convert_point_detector_to_opencv(image_points, object_points)
            self.calibration_data.push(image, ids, object_points, image_points)
            number_of_points = image_points.shape[0]

        return number_of_points

    def get_number_of_views(self):
        """
        Returns the current number of stored views.

        :return: number of views
        """
        return self.calibration_data.get_number_of_views()

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

        recon_err = \
            cm.compute_mono_reconstruction_err(self.calibration_data.ids_arrays,
                                               self.calibration_data.object_points_arrays,
                                               self.calibration_data.image_points_arrays,
                                               rvecs,
                                               tvecs,
                                               camera_matrix,
                                               dist_coeffs
                                               )
        return proj_err, recon_err

    def save_data(self,
                  dir_name: str,
                  file_prefix: str):
        """
        Saves the data to the given dir_name, with file_prefix.
        """
        self.calibration_data.save_data(dir_name, file_prefix)

    def load_data(self,
                  dir_name: str,
                  file_prefix: str):
        """
        Loads the data from dir_name, and populates this object.
        """
        self.calibration_data.load_data(dir_name, file_prefix)

    def save_params(self,
                    dir_name: str,
                    file_prefix: str):
        """
        Saves the calibration parameters to dir_name, with file_prefix.
        """
        self.calibration_params.save_data(dir_name, file_prefix)

    def load_params(self,
                    dir_name: str,
                    file_prefix: str):
        """
        Loads the calibration params from dir_name, using file_prefix.
        """
        self.calibration_params.load_data(dir_name, file_prefix)
        




