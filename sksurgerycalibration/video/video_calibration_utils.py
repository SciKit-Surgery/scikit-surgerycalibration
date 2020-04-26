# -*- coding: utf-8 -*-

""" Various utilities, converters etc., to help video calibration. """

import collections
import numpy as np
import cv2
import sksurgerycore.transforms.matrix as skcm


def convert_numpy2d_to_opencv(image_points):
    """
    Converts numpy array to Vector of 1x2 vectors containing float32.

    :param image_points: numpy [Mx2] array.
    :return: vector (length M), of 1x2 vectors of float32.
    """
    return np.reshape(image_points, (-1, 1, 2)).astype(np.float32)


def convert_numpy3d_to_opencv(object_points):
    """
    Converts numpy array to Vector of 1x3 vectors containing float32.

    :param object_points: numpy [Mx3] array.
    :return: vector (length M), of 1x3 vectors of float32.
    """
    return np.reshape(object_points, (-1, 1, 3)).astype(np.float32)


def extrinsic_vecs_to_matrix(rvec, tvec):
    """
    Method to convert rvec and tvec to a 4x4 matrix.

    :param rvec: [3x1] ndarray, Rodrigues rotation params
    :param rvec: [3x1] ndarray, translation params
    :return: [3x3] ndarray, Rotation Matrix
    """
    rotation_matrix = (cv2.Rodrigues(rvec))[0]
    transformation_matrix = \
        skcm.construct_rigid_transformation(rotation_matrix, tvec)
    return transformation_matrix


def extrinsic_matrix_to_vecs(transformation_matrix):
    """
    Method to convert a [4x4] rigid body matrix to an rvec and tvec.

    :param transformation_matrix: [4x4] rigid body matrix.
    :return [3x1] Rodrigues rotation vec, [3x1] translation vec
    """
    rmat = transformation_matrix[0:3, 0:3]
    rvec = (cv2.Rodrigues(rmat))[0]
    tvec = np.ones((3, 1))
    tvec[0:3, 0] = transformation_matrix[0:3, 3]
    return rvec, tvec


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


def convert_point_detector_to_opencv(ids, object_points, image_points):
    """
    The PointDetectors from scikit-surgeryimage aren't quite compatible
    with OpenCV.
    """
    dims = np.shape(image_points)
    ids = np.reshape(ids, dims[0])
    image_points = np.reshape(image_points, (dims[0], 1, 2))
    image_points = image_points.astype(np.float32)
    object_points = np.reshape(object_points, (-1, 1, 3))
    object_points = object_points.astype(np.float32)
    return ids, image_points, object_points
