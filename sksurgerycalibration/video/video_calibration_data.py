# -*- coding: utf-8 -*-

""" Containers for video calibration data. """

import os
import copy
import cv2
import numpy as np
import sksurgerycalibration.video.video_calibration_io as sksio
import sksurgerycalibration.video.video_calibration_hand_eye as heye


class BaseVideoCalibrationData:
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

    def pop(self):
        """ Remove the last view of data. """
        raise NotImplementedError("Derived classes should implement this.")

    def get_number_of_views(self):
        """ Returns the number of views of data. """
        raise NotImplementedError("Derived classes should implement this.")

    def save_data(self, dir_name: str, file_prefix: str):
        """ Writes all contained data to disk. """
        raise NotImplementedError("Derived classes should implement this.")

    def load_data(self, dir_name: str, file_prefix: str):
        """ Loads all contained data from disk. """
        raise NotImplementedError("Derived classes should implement this.")


class TrackingData(BaseVideoCalibrationData):
    """
    Class for storing tracking data.
    """
    def __init__(self):
        super().__init__()
        self.device_tracking_array = None
        self.calibration_tracking_array = None
        self.use_quaternions = False
        self.quat_model2hand_array = None
        self.trans_model2hand_array = None
        self.reinit()

    def reinit(self):
        """
        Deletes all data.
        """
        self.device_tracking_array = []
        self.calibration_tracking_array = []

    def push(self, device_tracking, calibration_tracking):
        """
        Stores a pair of tracking data.

        :param device_tracking: transformation for the thing you're tracking
        :param calibration_tracking: transformation for tracked calibration obj
        """
        self.device_tracking_array.append(
            copy.deepcopy(device_tracking))
        self.calibration_tracking_array.append(
            copy.deepcopy(calibration_tracking))

    def pop(self):
        """
        Removes the last (most recent) view of data.
        """
        if self.device_tracking_array:
            self.device_tracking_array.pop(-1)
            self.calibration_tracking_array.pop(-1)

    def get_number_of_views(self):
        """
        Returns the number of views of data.
        :return: int
        """
        return len(self.device_tracking_array)

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves the tracking data to lots of different files.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        for i in enumerate(self.device_tracking_array):
            device_tracking_file = \
                sksio.get_device_tracking_file_name(dir_name,
                                                    file_prefix,
                                                    i[0])
            if self.device_tracking_array[i[0]] is not None:
                np.savetxt(device_tracking_file,
                           self.device_tracking_array[i[0]], fmt='%.8f')

        for i in enumerate(self.calibration_tracking_array):
            calibration_tracking_file = \
                sksio.get_calibration_tracking_file_name(dir_name,
                                                         file_prefix,
                                                         i[0])
            if self.calibration_tracking_array[i[0]] is not None:
                np.savetxt(calibration_tracking_file,
                           self.calibration_tracking_array[i[0]], fmt='%.8f')

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads tracking data from files.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        self.reinit()
        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "device_tracking",
                                                 ".txt")
        for file in files:
            device_data = np.loadtxt(file)
            self.device_tracking_array.append(device_data)

        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "calib_obj_tracking",
                                                 ".txt")
        for file in files:
            calibration_data = np.loadtxt(file)
            self.calibration_tracking_array.append(calibration_data)

    def set_model2hand_arrays(self, use_quaternions=False):
        """
        TODO: Docstring update
        Set the attributes model-to-hand quaternion and translation arrays
        from tracking data.
        """

        self.quat_model2hand_array, self.trans_model2hand_array = \
            heye.set_model2hand_arrays(self.calibration_tracking_array,
                                       self.device_tracking_array,
                                       use_quaternions)


class MonoVideoData(BaseVideoCalibrationData):
    """
    Stores data extracted from each video view of a mono calibration.
    """
    def __init__(self):
        super().__init__()
        self.images_array = None
        self.ids_arrays = None
        self.object_points_arrays = None
        self.image_points_arrays = None
        self.reinit()

    def reinit(self):
        """
        Deletes all data.
        """
        self.images_array = []
        self.ids_arrays = []
        self.object_points_arrays = []
        self.image_points_arrays = []

    def pop(self):
        """
        Removes the last (most recent) view of data.
        """
        if len(self.images_array) > 1:
            self.images_array.pop(-1)
            self.ids_arrays.pop(-1)
            self.object_points_arrays.pop(-1)
            self.image_points_arrays.pop(-1)

    def push(self, image, ids, object_points, image_points):
        """
        Stores another view of data. Copies data.
        """
        self.images_array.append(copy.deepcopy(image))
        self.ids_arrays.append(copy.deepcopy(ids))
        self.object_points_arrays.append(copy.deepcopy(object_points))
        self.image_points_arrays.append(copy.deepcopy(image_points))

    def get_number_of_views(self):
        """
        Returns the number of views.
        """
        return len(self.images_array)

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves the calibration data to lots of different files.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)

        for i in enumerate(self.images_array):
            image_file = sksio.get_images_file_name(dir_name,
                                                    file_prefix,
                                                    i[0])
            cv2.imwrite(image_file, self.images_array[i[0]])
        for i in enumerate(self.ids_arrays):
            id_file = sksio.get_ids_file_name(dir_name,
                                              file_prefix,
                                              i[0])
            np.savetxt(id_file, self.ids_arrays[i[0]], fmt='%d')
        for i in enumerate(self.object_points_arrays):
            object_points_file = sksio.get_objectpoints_file_name(dir_name,
                                                                  file_prefix,
                                                                  i[0])
            reshaped = np.reshape(self.object_points_arrays[i[0]], (-1, 3))
            np.savetxt(object_points_file, reshaped, fmt='%.8f')
        for i in enumerate(self.image_points_arrays):
            image_points_file = sksio.get_imagepoints_file_name(dir_name,
                                                                file_prefix,
                                                                i[0])
            reshaped = np.reshape(self.image_points_arrays[i[0]], (-1, 2))
            np.savetxt(image_points_file, reshaped, fmt='%.8f')

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads the calibration data.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        self.reinit()

        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "images",
                                                 ".png")
        for file in files:
            image = cv2.imread(file)
            self.images_array.append(image)

        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "ids",
                                                 ".txt")
        for file in files:
            ids = np.loadtxt(file)
            self.ids_arrays.append(ids)

        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "object_points",
                                                 ".txt")

        for file in files:
            object_points = np.loadtxt(file)
            reshaped = np.reshape(object_points, (object_points.shape[0], 1, 3))
            self.object_points_arrays.append(reshaped.astype(np.float32))

        files = sksio.get_filenames_by_glob_expr(dir_name,
                                                 file_prefix,
                                                 "image_points",
                                                 ".txt")

        for file in files:
            image_points = np.loadtxt(file)
            reshaped = np.reshape(image_points, (image_points.shape[0], 1, 2))
            self.image_points_arrays.append(reshaped.astype(np.float32))


class StereoVideoData(BaseVideoCalibrationData):
    """
    Stores data extracted from each view of a stereo calibration.
    """
    def __init__(self):
        super().__init__()
        self.left_data = MonoVideoData()
        self.right_data = MonoVideoData()
        self.reinit()

    def reinit(self):
        """
        Deletes all data.
        """
        self.left_data.reinit()
        self.right_data.reinit()

    def pop(self):
        """
        Removes the last (most recent) view of data.
        """
        self.left_data.pop()
        self.right_data.pop()

    def push(self,
             left_image, left_ids, left_object_points, left_image_points,
             right_image, right_ids, right_object_points, right_image_points):
        """
        Stores another view of data. Copies data.
        """
        self.left_data.push(
            left_image, left_ids, left_object_points, left_image_points)
        self.right_data.push(
            right_image, right_ids, right_object_points, right_image_points)

    def get_number_of_views(self):
        """
        Returns the number of views.
        """
        num_left = self.left_data.get_number_of_views()
        num_right = self.right_data.get_number_of_views()
        if num_left != num_right:
            raise ValueError("Different number of views in left and right??")
        return num_left

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves the calibration data to lots of different files.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        left_prefix = sksio.get_left_prefix(file_prefix)
        self.left_data.save_data(dir_name, left_prefix)
        right_prefix = sksio.get_right_prefix(file_prefix)
        self.right_data.save_data(dir_name, right_prefix)

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads the calibration data.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        self.reinit()
        left_prefix = sksio.get_left_prefix(file_prefix)
        self.left_data.load_data(dir_name, left_prefix)
        right_prefix = sksio.get_right_prefix(file_prefix)
        self.right_data.load_data(dir_name, right_prefix)
