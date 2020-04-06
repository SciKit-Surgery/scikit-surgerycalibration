# -*- coding: utf-8 -*-

""" Video Calibration functions """

import collections
import numpy as np
import cv2


def convert_numpy2d_to_opencv(image_points):
    """
    Converts numpy array to Vector of 1x2 vectors containing float32.

    :param image_points: numpy Mx2 array.
    :return: vector (length M), of 1x2 vectors of float32.
    """
    return np.reshape(image_points, (-1, 1, 2)).astype(np.float32)


def convert_numpy3d_to_opencv(object_points):
    """
    Converts numpy array to Vector of 1x3 vectors containing float32.

    :param object_points: numpy Mx3 array.
    :return: vector (length M), of 1x3 vectors of float32.
    """
    return np.reshape(object_points, (-1, 1, 3)).astype(np.float32)


def mono_video_calibration(object_points, image_points, image_size):
    """
    Calibrates a video camera using Zhang's 2000 method, as implemented in
    OpenCV. We wrap it here, so we have a place to add extra validation code,
    and a space for documentation.

      - N = number of images
      - M = number of points for that image
      - rvecs = list of 1x3 Rodrigues rotation parameters
      - tvecs = list of 3x1 translation vectors
      - camera_matrix = [3x3] ndarray containing fx, fy, cx, cy
      - dist_coeffs = [1x5] ndarray, containing distortion coefficients

    :param object_points: Vector (N) of Vector (M) of 1x3 points of type float
    :param image_points: Vector (N) of Vector (M) of 1x2 points of type float
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return: retval, camera_matrix, dist_coeffs, rvecs, tvecs
    """
    retval, camera_matrix, dist_coeffs, rvecs, tvecs \
        = cv2.calibrateCamera(object_points,
                              image_points,
                              image_size,
                              None, None)

    return retval, camera_matrix, dist_coeffs, rvecs, tvecs


def filter_common_points_per_image(left_ids,
                                   left_object_points,
                                   left_image_points,
                                   right_ids,
                                   right_image_points,
                                   minimum_points
                                   ):
    """
    For stereo calibration, we need common points in left and right.
    Remember that a point detector, may provide different numbers of
    points for left and right, and they may not be sorted.

    :param left_ids: ndarray of integer point ids
    :param left_object_points: Vector of Vector of 1x3 float 32
    :param left_image_points: Vector of Vector of 1x2 float 32
    :param right_ids: ndarray of integer point ids
    :param right_image_points: Vector of Vector of 1x2 float 32
    :param minimum_points: the number of minimum common points to accept
    :return: common ids, object_points, left_image_points, right_image_points
    """

    # Filter obvious duplicates first.
    non_duplicate_left = np.asarray(
        [item for item, count in
         collections.Counter(left_ids).items() if count == 1])
    non_duplicate_right = np.asarray(
        [item for item, count in
         collections.Counter(right_ids).items() if count == 1])

    filtered_left = left_ids[
        np.isin(left_ids, non_duplicate_left)]
    filtered_right = right_ids[
        np.isin(right_ids, non_duplicate_right)]

    # Now find common points in left and right.
    ids = np.intersect1d(filtered_left, filtered_right)
    ids = np.sort(ids)

    if len(ids) < minimum_points:
        raise ValueError("Not enough common points in left and right images.")

    common_ids = \
        np.zeros((len(ids), 1), dtype=np.int)
    common_object_points = \
        np.zeros((len(ids), 1, 3), dtype=np.float32)
    common_left_image_points = \
        np.zeros((len(ids), 1, 2), dtype=np.float32)
    common_right_image_points = \
        np.zeros((len(ids), 1, 2), dtype=np.float32)

    counter = 0
    for position in ids:
        left_location = np.where(left_ids == position)
        common_ids[counter] = left_ids[left_location[0][0]]
        common_object_points[counter] \
            = left_object_points[left_location[0][0]]
        common_left_image_points[counter] \
            = left_image_points[left_location[0][0]]
        right_location = np.where(right_ids == position)
        common_right_image_points[counter] \
            = right_image_points[right_location[0][0]]
        counter = counter + 1

    number_of_left = len(common_left_image_points)
    number_of_right = len(common_right_image_points)

    if number_of_left != number_of_right:
        raise ValueError("Unequal number of common points in left and right.")

    return common_ids, common_object_points, common_left_image_points, \
        common_right_image_points


def filter_common_points_all_images(left_ids,
                                    left_object_points,
                                    left_image_points,
                                    right_ids,
                                    right_image_points,
                                    minimum_points
                                    ):
    """
    Loops over each images's data, filtering per image.
    See: filter_common_points_per_image
    :return: Vectors of outputs from filter_common_points_per_image
    """
    common_ids = []
    common_object_points = []
    common_left_image_points = []
    common_right_image_points = []

    # pylint:disable=consider-using-enumerate
    for counter in range(len(left_ids)):
        c_i, c_o, c_l, c_r = \
            filter_common_points_per_image(left_ids[counter],
                                           left_object_points[counter],
                                           left_image_points[counter],
                                           right_ids[counter],
                                           right_image_points[counter],
                                           minimum_points
                                           )
        common_ids.append(c_i)
        common_object_points.append(c_o)
        common_left_image_points.append(c_l)
        common_right_image_points.append(c_r)

    return common_ids, common_object_points, common_left_image_points, \
        common_right_image_points


def stereo_video_calibration(left_ids,
                             left_object_points,
                             left_image_points,
                             right_ids,
                             right_object_points,
                             right_image_points,
                             image_size):
    """
    Default stereo calibration, using OpenCV methods.

    :param left_ids: Vector of ndarrays containing integer point ids.
    :param left_object_points: Vector of Vectors of 1x3 object points, float32
    :param left_image_points:  Vector of Vectors of 1x2 object points, float32
    :param right_ids: Vector of ndarrays containing integer point ids.
    :param right_object_points: Vector of Vectors of 1x3 object points, float32
    :param right_image_points: Vector of Vectors of 1x2 object points, float32
    :param image_size: (x, y) tuple, size in pixels, e.g. (1920, 1080)
    :return:
    """
    # Calibrate left, using all available points
    _, l_c, l_d, _, _ \
        = cv2.calibrateCamera(left_object_points,
                              left_image_points,
                              image_size,
                              None, None)

    # Calibrate right using all available points.
    _, r_c, r_d, _, _ \
        = cv2.calibrateCamera(right_object_points,
                              right_image_points,
                              image_size,
                              None, None)

    # But for stereo, we need common points.
    _, common_object_points, common_left_image_points, \
        common_right_image_points \
        = filter_common_points_all_images(left_ids,
                                          left_object_points,
                                          left_image_points,
                                          right_ids,
                                          right_image_points, 10)

    # So, now we can calibrate using only points that occur in left and right.
    s_rms, l_c, l_d, r_c, r_d, \
        l2r_r, l2r_t, essential, fundamental = cv2.stereoCalibrate(
            common_object_points,
            common_left_image_points,
            common_right_image_points,
            l_c,
            l_d,
            r_c,
            r_d,
            image_size,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    return s_rms, l_c, l_d, r_c, r_d, l2r_r, l2r_t, essential, fundamental
