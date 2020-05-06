# -*- coding: utf-8 -*-

""" Base class for our mono and stereo video camera calibration drivers. """

import copy
import logging
import sksurgeryimage.calibration.point_detector as pd
import sksurgerycalibration.video.video_calibration_data as vcd
import sksurgerycalibration.video.video_calibration_params as vcp
import sksurgerycalibration.video.video_calibration_utils as vcu

LOGGER = logging.getLogger(__name__)


class BaseVideoCalibrationDriver:
    """
    Base class for video calibration drivers.
    """
    def __init__(self,
                 point_detector: pd.PointDetector,
                 minimum_points_per_frame: int
                 ):
        """
        Base class for video calibration drivers.

        This class expects calling code to decide how many images are
        required to calibrate, and also, when to call reinit.

        The PointDetector is passed in using Dependency Injection.
        So, the PointDetector can be anything, like chessboards, ArUco,
        CharUco etc.

        This does mean that the underlying code can handle variable numbers
        of points in each view. OpenCV calibration math does this anyway.

        :param point_detector: Class derived from PointDetector
        :param minimum_points_per_frame: Minimum number to accept frame
        """
        self.point_detector = point_detector
        self.minimum_points_per_frame = minimum_points_per_frame
        self.tracking_data = vcd.TrackingData()
        self.video_data = None
        self.calibration_params = None
        LOGGER.info("Constructed: Points per view=%s",
                    str(self.minimum_points_per_frame))

    def _init_internal(self,
                       video_data: vcd.BaseVideoCalibrationData,
                       calibration_params: vcp.BaseCalibrationParams):
        """
        Derived classes must call this, to assign to

        - self.calibration_data
        - self.calibration_params
        """
        self.video_data = video_data
        self.calibration_params = calibration_params

    def reinit(self):
        """
        Resets this object, which means, removes stored calibration data
        and reset the calibration parameters to identity/zero.
        """
        self.tracking_data.reinit()
        self.video_data.reinit()
        self.calibration_params.reinit()
        LOGGER.info("Reset: Now zero frames.")

    def pop(self):
        """
        Removes the last grabbed view of data.
        """
        self.tracking_data.pop()
        self.video_data.pop()
        LOGGER.info("Popped: Now %s views.", str(self.get_number_of_views()))

    def get_number_of_views(self):
        """
        Returns the current number of stored views.

        :return: number of views
        """
        return self.video_data.get_number_of_views()

    def calibrate(self, flags=0):
        """
        Do the video calibration. Derived classes must implement this.
        """
        raise NotImplementedError("Derived classes must implement this.")

    def save_data(self,
                  dir_name: str,
                  file_prefix: str):
        """
        Saves the data to the given dir_name, with file_prefix.
        """
        self.tracking_data.save_data(dir_name, file_prefix)
        self.video_data.save_data(dir_name, file_prefix)

    def load_data(self,
                  dir_name: str,
                  file_prefix: str):
        """
        Loads the data from dir_name, and populates this object.
        """
        self.tracking_data.load_data(dir_name, file_prefix)
        self.video_data.load_data(dir_name, file_prefix)

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

    def get_params(self):
        """
        Copies and returns the parameters.
        """
        return copy.deepcopy(self.calibration_params)

    def get_video_data(self):
        """
        Copies and returns the video data.
        """
        return copy.deepcopy(self.video_data)

    def get_tracking_data(self):
        """
        Copies and returns the tracking data.
        """
        return copy.deepcopy(self.tracking_data)

    def is_device_tracked(self):
        """
        Returns True if we have tracking data for the device.
        """
        result = \
            vcu.array_contains_tracking_data(
                self.tracking_data.device_tracking_array
            )
        return result

    def is_calibration_target_tracked(self):
        """
        Returns True if we have tracking data for the calibration target.
        """
        result = \
            vcu.array_contains_tracking_data(
                self.tracking_data.calibration_tracking_array
            )
        return result
