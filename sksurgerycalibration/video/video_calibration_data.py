# -*- coding: utf-8 -*-

""" Containers for video calibration data. """

import copy
import cv2
import numpy as np
import sksurgerycalibration.video.video_calibration_io as sksio


class MonoVideoData:
    """
    Stores data extracted from each video view of a calibration.
    """
    def __init__(self):
        self.images_array = []
        self.ids_arrays = []
        self.object_points_arrays = []
        self.image_points_arrays = []

    def reinit(self):
        """
        Deletes all data.
        """
        self.images_array = []
        self.ids_arrays = []
        self.object_points_arrays = []
        self.image_points_arrays = []

    def get_number_of_views(self):
        """
        Returns the number of views.
        """
        return len(self.images_array)

    def push(self, image, ids, object_points, image_points):
        """
        Stores another view of data. Copies data.
        """
        self.images_array.append(copy.deepcopy(image))
        self.ids_arrays.append(copy.deepcopy(ids))
        self.object_points_arrays.append(copy.deepcopy(object_points))
        self.image_points_arrays.append(copy.deepcopy(image_points))

    def pop(self):
        """
        Removes the last (most recent) view of data.
        """
        if len(self.images_array) > 1:
            self.images_array.pop(-1)
            self.ids_arrays.pop(-1)
            self.object_points_arrays.pop(-1)
            self.image_points_arrays.pop(-1)

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves the calibration data to lots of different files.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        for i in enumerate(self.images_array):
            image_file = sksio._get_images_file_name(dir_name,
                                                     file_prefix,
                                                     i[0])
            cv2.imwrite(image_file, self.images_array[i])
        for i in enumerate(self.ids_arrays):
            id_file = sksio._get_ids_file_name(dir_name,
                                               file_prefix,
                                               i[0])
            np.savetxt(id_file, self.ids_arrays[i])
        for i in enumerate(self.object_points_arrays):
            object_points_file = sksio._get_objectpoints_file_name(dir_name,
                                                                   file_prefix,
                                                                   i[0])
            with open(object_points_file, 'w') as f:
                for j in range(0, len(self.object_points_arrays[i])):
                    np.savetxt(f, self.object_points_arrays[i][j], fmt='%f')
        for i in enumerate(self.image_points_arrays):
            image_points_file = sksio._get_imagepoints_file_name(dir_name,
                                                                 file_prefix,
                                                                 i[0])
            with open(image_points_file, 'w') as f:
                for j in range(0, len(self.image_points_arrays[i])):
                    np.savetxt(f, self.image_points_arrays[i][j], fmt='%f')

    def load_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Loads calibration data from a directory.

        :param dir_name: directory to load from
        :param file_prefix: prefix for all files
        """
        raise RuntimeError("Not implemented yet. Please volunteer.")


class StereoVideoData:
    """
    Stores data extracted from each view of a stereo calibration.
    """
    def __init__(self):
        self.left_data = MonoVideoData()
        self.right_data = StereoVideoData()

    def reinit(self):
        """
        Deletes all data.
        """
        self.left_data.reinit()
        self.right_data.reinit()

    def get_number_of_views(self):
        """
        Returns the number of views.
        """
        num_left = self.left_data.get_number_of_views()
        num_right = self.right_data.get_number_of_views()
        if num_left != num_right:
            raise ValueError("Different number of views in left and right??")
        return num_left

    def pop(self):
        """
        Removes the last (most recent) view of data.
        """
        self.left_data.pop()
        self.right_data.pop()

    def push(self,
             left_image, left_ids, left_object_points, left_image_points,
             right_image, right_ids, right_object_points, right_image_points
             ):
        """
        Stores another view of data. Copies data.
        """
        self.left_data.push(
            left_image, left_ids, left_object_points, left_image_points)
        self.right_data.push(
            right_image, right_ids, right_object_points, right_image_points)

    def save_data(self,
                  dir_name: str,
                  file_prefix: str
                  ):
        """
        Saves the calibration data to lots of different files.

        :param dir_name: directory to save to
        :param file_prefix: prefix for all files
        """
        left_prefix = sksio._get_left_prefix(file_prefix)
        self.left_data.save_data(dir_name, left_prefix)
        right_prefix = sksio._get_right_prefix(file_prefix)
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
        left_prefix = sksio._get_left_prefix(file_prefix)
        self.left_data.load_data(dir_name, left_prefix)
        right_prefix = sksio._get_right_prefix(file_prefix)
        self.right_data.load_data(dir_name, right_prefix)
