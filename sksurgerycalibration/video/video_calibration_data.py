# -*- coding: utf-8 -*-

""" Containers for video calibration data. """

import copy


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
