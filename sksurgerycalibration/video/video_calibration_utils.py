# -*- coding: utf-8 -*-

""" Various utilities, converters etc., to help video calibration. """

# pylint:disable=invalid-name

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


def convert_pd_to_opencv(ids, object_points, image_points):
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


def array_contains_tracking_data(array_to_check):
    """
    Returns True if the array contains some tracking data.
    """
    result = False
    if array_to_check is not None:
        number_of_items = len(array_to_check)
        if number_of_items > 0:
            found_none = False
            for i in range(0, number_of_items):
                if array_to_check[i] is None:
                    found_none = True
            if not found_none:
                result = True
    return result


def match_points_by_id(ids_1, points_1, ids_2, points_2):
    """
    Returns an ndarray of matched points, matching by their identifier.

    :param ids_1: ndarray [Mx1] list of ids for points_1
    :param points_1: ndarray [Mx2 or 3] of 2D or 3D points
    :param ids_2: ndarray [Nx1] list of ids for points_2
    :param points_2: ndarray [Nx2 or 3] of 2D or 3D points
    :return: ndarray. Number of rows is the number of common points by ids.
    """
    common_ids = np.intersect1d(ids_1, ids_2)
    common_ids = np.sort(common_ids)
    indexes_1 = np.isin(ids_1, common_ids).reshape(-1)
    indexes_2 = np.isin(ids_2, common_ids).reshape(-1)
    points_1_selected = points_1[indexes_1, :]
    points_2_selected = points_2[indexes_2, :]
    result = np.zeros((common_ids.shape[0],
                       points_1_selected.shape[1] +
                       points_2_selected.shape[1]))
    result[:, 0:points_1_selected.shape[1]] \
        = points_1_selected[:, :]
    result[:, points_1_selected.shape[1]:points_1_selected.shape[1] +
           points_2_selected.shape[1]] = points_2_selected[:, :]
    return result


def distort_points(image_points, camera_matrix, distortion_coeffs):
    """
    Distorts image points, reversing the effects of cv2.undistortPoints.

    Slow, but should do for now, for offline calibration at least.

    :param image_points: undistorted image points.
    :param camera_matrix: [3x3] camera matrix
    :param distortion_coeffs: [1x5] distortion coefficients
    :return: distorted points
    """
    distorted_pts = np.zeros(image_points.shape)
    number_of_points = image_points.shape[0]

    # pylint: disable=invalid-name
    for counter in range(number_of_points):
        relative_x = (image_points[counter][0] - camera_matrix[0][2]) \
            / camera_matrix[0][0]
        relative_y = (image_points[counter][1] - camera_matrix[1][2]) \
            / camera_matrix[1][1]
        r2 = relative_x * relative_x + relative_y * relative_y
        radial = (
            1
            + distortion_coeffs[0][0]
            * r2
            + distortion_coeffs[0][1]
            * r2 * r2
            + distortion_coeffs[0][4]
            * r2 * r2 * r2
        )
        distorted_x = relative_x * radial
        distorted_y = relative_y * radial

        distorted_x = distorted_x + (
            2 * distortion_coeffs[0][2]
            * relative_x * relative_y
            + distortion_coeffs[0][3]
            * (r2 + 2 * relative_x * relative_x))

        distorted_y = distorted_y + (
            distortion_coeffs[0][2]
            * (r2 + 2 * relative_y * relative_y)
            + 2 * distortion_coeffs[0][3]
            * relative_x * relative_y)

        distorted_x = distorted_x * camera_matrix[0][0] + camera_matrix[0][2]
        distorted_y = distorted_y * camera_matrix[1][1] + camera_matrix[1][2]

        distorted_pts[counter][0] = distorted_x
        distorted_pts[counter][1] = distorted_y

    return distorted_pts


def map_points_from_canonical_to_original(images_array,
                                          image_index,
                                          video_data,
                                          ids,
                                          object_points,
                                          image_points,
                                          homography,
                                          camera_matrix,
                                          distortion_coeffs):
    """
    Utility method to map image points, detected in a canonical face
    on image, back to the original image space.

    :param images_array:
    :param image_index:
    :param video_data:
    :param ids:
    :param object_points:
    :param image_points:
    :param homography:
    :param camera_matrix:
    :param distortion_coeffs:
    :return:
    """
    inverted_points = \
        cv2.perspectiveTransform(
            image_points.astype(np.float32).reshape(-1, 1, 2),
            np.linalg.inv(homography))
    inverted_points = inverted_points.reshape(-1, 2)

    distorted_pts = distort_points(
        inverted_points,
        camera_matrix,
        distortion_coeffs)

    # Convert back to a format suitable for OpenCV calibration.
    ids, image_points, object_points = \
        convert_pd_to_opencv(ids,
                             object_points,
                             distorted_pts)

    # Store the image, with ids, object_points and image_points.
    video_data.push(images_array[image_index],
                    ids,
                    object_points,
                    image_points)


def detect_points_in_canonical_space(point_detector,
                                     minimum_points_per_frame,
                                     video_data,
                                     images,
                                     camera_matrix,
                                     distortion_coefficients,
                                     reference_ids,
                                     reference_image_points,
                                     reference_image_size
                                     ):
    """
    Method that does the bulk of the heavy lifting in Datta 2009.

    :param point_detector:
    :param minimum_points_per_frame:
    :param video_data:
    :param images:
    :param camera_matrix:
    :param distortion_coefficients:
    :param reference_ids:
    :param reference_image_points:
    :param reference_image_size:
    :return:
    """
    video_data.reinit()
    for j, _ in enumerate(images):

        undistorted = cv2.undistort(
            images[j],
            camera_matrix,
            distortion_coefficients,
            camera_matrix
        )

        ids, obj_pts, img_pts = point_detector.get_points(undistorted)

        if ids is not None and ids.shape[0] >= minimum_points_per_frame:

            common_points = match_points_by_id(ids, img_pts,
                                               reference_ids,
                                               reference_image_points)
            homography, _ = \
                cv2.findHomography(common_points[0:, 0:2],
                                   common_points[0:, 2:4])
            warped = cv2.warpPerspective(undistorted,
                                         homography,
                                         reference_image_size,)

            ids, obj_pts, img_pts = point_detector.get_points(warped)

            if ids is not None and ids.shape[0] >= minimum_points_per_frame:

                map_points_from_canonical_to_original(images,
                                                      j,
                                                      video_data,
                                                      ids,
                                                      obj_pts,
                                                      img_pts,
                                                      homography,
                                                      camera_matrix,
                                                      distortion_coefficients)


# pylint: disable=too-many-arguments
def detect_points_in_stereo_canonical_space(left_point_detector,
                                            right_point_detector,
                                            minimum_points_per_frame,
                                            left_video_data,
                                            left_images,
                                            left_camera_matrix,
                                            left_distortion_coeffs,
                                            right_video_data,
                                            right_images,
                                            right_camera_matrix,
                                            right_distortion_coeffs,
                                            reference_ids,
                                            reference_image_points,
                                            reference_image_size
                                            ):
    """
    Method that does the bulk of the heavy lifting in Datta 2009.

    The reason we need a combined stereo one, instead of calling the mono
    one twice, is because at any point, if either left or right channel
    fails feature detection, we need to drop that image from BOTH channels.

    :param left_point_detector:
    :param right_point_detector:
    :param minimum_points_per_frame:
    :param left_video_data:
    :param left_images:
    :param left_camera_matrix:
    :param left_distortion_coeffs:
    :param right_video_data:
    :param right_images:
    :param right_camera_matrix:
    :param right_distortion_coeffs:
    :param reference_ids:
    :param reference_image_points:
    :param reference_image_size:
    :return:
    """
    left_video_data.reinit()
    right_video_data.reinit()

    for j, _ in enumerate(left_images):
        left_undistorted = cv2.undistort(
            left_images[j],
            left_camera_matrix,
            left_distortion_coeffs,
            left_camera_matrix
        )
        left_ids, left_obj_pts, left_img_pts = \
            left_point_detector.get_points(left_undistorted)

        right_undistorted = cv2.undistort(
            right_images[j],
            right_camera_matrix,
            right_distortion_coeffs,
            right_camera_matrix
        )
        right_ids, right_obj_pts, right_img_pts = \
            right_point_detector.get_points(right_undistorted)

        if left_ids is not None \
                and left_ids.shape[0] >= minimum_points_per_frame \
                and right_ids is not None \
                and right_ids.shape[0] >= minimum_points_per_frame:

            left_common_points = match_points_by_id(left_ids,
                                                    left_img_pts,
                                                    reference_ids,
                                                    reference_image_points)
            left_homography, _ = \
                cv2.findHomography(left_common_points[0:, 0:2],
                                   left_common_points[0:, 2:4])
            left_warped = cv2.warpPerspective(left_undistorted,
                                              left_homography,
                                              reference_image_size,)

            left_ids, left_obj_pts, left_img_pts = \
                left_point_detector.get_points(left_warped)

            right_common_points = match_points_by_id(right_ids,
                                                     right_img_pts,
                                                     reference_ids,
                                                     reference_image_points)
            right_homography, _ = \
                cv2.findHomography(right_common_points[0:, 0:2],
                                   right_common_points[0:, 2:4])
            right_warped = cv2.warpPerspective(right_undistorted,
                                               right_homography,
                                               reference_image_size,)

            right_ids, right_obj_pts, right_img_pts = \
                right_point_detector.get_points(right_warped)

            if left_ids is not None \
                    and left_ids.shape[0] >= minimum_points_per_frame \
                    and right_ids is not None \
                    and right_ids.shape[0] >= minimum_points_per_frame:

                map_points_from_canonical_to_original(left_images,
                                                      j,
                                                      left_video_data,
                                                      left_ids,
                                                      left_obj_pts,
                                                      left_img_pts,
                                                      left_homography,
                                                      left_camera_matrix,
                                                      left_distortion_coeffs)

                map_points_from_canonical_to_original(right_images,
                                                      j,
                                                      right_video_data,
                                                      right_ids,
                                                      right_obj_pts,
                                                      right_img_pts,
                                                      right_homography,
                                                      right_camera_matrix,
                                                      right_distortion_coeffs)
