# pylint: disable=line-too-long, missing-module-docstring, invalid-name, missing-function-docstring

import os
from typing import Tuple
import numpy as np
import cv2
import sksurgeryimage.calibration.dotty_grid_point_detector as dotty_pd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc


def get_calib_data(directory: str, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Generate filenames for calibration data in a given directory. """
    left_image = cv2.imread(
        os.path.join(directory, f'calib.left.images.{idx}.png')
    )

    right_image = cv2.imread(
        os.path.join(directory, f'calib.right.images.{idx}.png')
    )

    chessboard_tracking = np.loadtxt(
        os.path.join(directory, f'calib.calib_obj_tracking.{idx}.txt')
    )

    scope_tracking = np.loadtxt(
        os.path.join(directory, f'calib.device_tracking.{idx}.txt')
    )

    return left_image, right_image, chessboard_tracking, scope_tracking


def get_point_detector(intrinsic_matrix, distortion_matrix):
    """
    Returns a point detector based on a set of
    camera intrinsics and distortion coefficient parameters.

    :param intrinsic_matrix: [3x3] matrix
    :param distortion_matrix: [1x5] matrix
    :return:
    """
    number_of_dots = [18, 25]
    dot_separation = 3
    fiducial_indexes = [133, 141, 308, 316]
    reference_image_size = [1900, 2600]
    pixels_per_mm = 40

    number_of_points = number_of_dots[0] * number_of_dots[1]
    model_points = np.zeros((number_of_points, 6))
    counter = 0
    for y_index in range(number_of_dots[0]):
        for x_index in range(number_of_dots[1]):
            model_points[counter][0] = counter
            model_points[counter][1] = (x_index + 1) * pixels_per_mm
            model_points[counter][2] = (y_index + 1) * pixels_per_mm
            model_points[counter][3] = x_index * dot_separation
            model_points[counter][4] = y_index * dot_separation
            model_points[counter][5] = 0
            counter = counter + 1

    point_detector = \
        dotty_pd.DottyGridPointDetector(
            model_points,
            fiducial_indexes,
            intrinsic_matrix,
            distortion_matrix,
            reference_image_size=(reference_image_size[1],
                                  reference_image_size[0])
            )
    return point_detector


def get_calib_driver(calib_dir: str):

    left_point_detector = get_point_detector(
        np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.left.intrinsics.txt"),
        np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.left.distortion.txt"))

    right_point_detector = get_point_detector(
        np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.right.intrinsics.txt"),
        np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.right.distortion.txt"))

    minimum_points = 50

    calibration_driver = sc.StereoVideoCalibrationDriver(left_point_detector,
                                                         right_point_detector,
                                                         minimum_points)

    for i in range(3):
        l_img, r_img, chessboard, scope = get_calib_data(calib_dir, i)
        calibration_driver.grab_data(l_img, r_img, scope, chessboard)

    return calibration_driver


def get_ref_dot_detector():

    camera_matrix = np.eye(3)
    distortion_coefficients = np.zeros(5)

    ref_point_detector = get_point_detector(camera_matrix,
                                            distortion_coefficients)
    return ref_point_detector


def test_dotty_grid_iterative():

    os.makedirs("tests/output/iterative", exist_ok=True)

    calib_dir = 'tests/data/dot_calib/11_19_11'

    calib_driver = get_calib_driver(calib_dir)
    ref_detector = get_ref_dot_detector()

    ref_image = cv2.imread('tests/data/dot_calib/circles-25x18-r40-s3.png')
    ref_ids, _, ref_image_points = \
        ref_detector.get_points(ref_image)

    reprojection_err, recon_err, _ = calib_driver.iterative_calibration(
        3,
        ref_ids,
        ref_image_points,
        (ref_image.shape[1],
         ref_image.shape[0]))

    assert reprojection_err < 2
    assert recon_err < 2
