# pylint: disable=line-too-long, missing-module-docstring, invalid-name, missing-function-docstring

import os
from typing import Tuple
import numpy as np
import cv2
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector \
    as charuco_pd
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


def get_calib_driver(calib_dir: str):
    """ Create left/right charuco point detectors and load calibration images from directory. """
    reference_image = cv2.imread("tests/data/2020_01_20_storz/pattern_4x4_19x26_5_4_with_inset_9x14.png")
    minimum_points = 50

    number_of_squares = [19, 26]
    square_tag_sizes = [5, 4]
    filter_markers = True
    number_of_chessboard_squares = [9, 14]
    chessboard_square_size = 3
    chessboard_id_offset = 500

    left_pd = \
        charuco_pd.CharucoPlusChessboardPointDetector(
            reference_image,
            minimum_number_of_points=minimum_points,
            number_of_charuco_squares=number_of_squares,
            size_of_charuco_squares=square_tag_sizes,
            charuco_filtering=filter_markers,
            number_of_chessboard_squares=number_of_chessboard_squares,
            chessboard_square_size=chessboard_square_size,
            chessboard_id_offset=chessboard_id_offset
        )

    right_pd = \
        charuco_pd.CharucoPlusChessboardPointDetector(
            reference_image,
            minimum_number_of_points=minimum_points,
            number_of_charuco_squares=number_of_squares,
            size_of_charuco_squares=square_tag_sizes,
            charuco_filtering=filter_markers,
            number_of_chessboard_squares=number_of_chessboard_squares,
            chessboard_square_size=chessboard_square_size,
            chessboard_id_offset=chessboard_id_offset
        )

    calibration_driver = sc.StereoVideoCalibrationDriver(left_pd,
                                                         right_pd,
                                                         minimum_points)

    for i in range(3):
        l_img, r_img, chessboard, scope = get_calib_data(calib_dir, i)
        calibration_driver.grab_data(l_img, r_img, scope, chessboard)

    return calibration_driver


# Two datasets, A and B.
# Independently calibrating them gives Stereo reprojection error < 1
# But if we pass the intrinsics from A as precalibration for B, then
# error is ~4, so potentially something fishy going on.
def test_charuco_dataset_A():

    calib_dir = 'tests/data/precalib/precalib_base_data'
    calib_driver = get_calib_driver(calib_dir)

    stereo_reproj_err, stereo_recon_err, _ = \
        calib_driver.calibrate()

    tracked_reproj_err, tracked_recon_err, _ = \
        calib_driver.handeye_calibration()

    print(stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err)
    assert stereo_reproj_err < 1
    assert stereo_recon_err < 4
    assert tracked_reproj_err < 3
    assert tracked_recon_err < 4


def test_charuco_dataset_B():

    calib_dir = 'tests/data/precalib/data_moved_scope'
    calib_driver = get_calib_driver(calib_dir)

    stereo_reproj_err, stereo_recon_err, _ = \
        calib_driver.calibrate()

    tracked_reproj_err, tracked_recon_err, _ = \
        calib_driver.handeye_calibration()

    print(stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err)
    assert stereo_reproj_err < 1
    assert stereo_recon_err < 3
    assert tracked_reproj_err < 4
    assert tracked_recon_err < 3


def test_precalbration():
    """ Use intrinsics from A to calibration B, currently failing. """
    left_intrinsics = np.loadtxt('tests/data/precalib/precalib_base_data/calib.left.intrinsics.txt')
    left_distortion = np.loadtxt('tests/data/precalib/precalib_base_data/calib.left.distortion.txt')
    right_intrinsics = np.loadtxt('tests/data/precalib/precalib_base_data/calib.right.intrinsics.txt')
    right_distortion = np.loadtxt('tests/data/precalib/precalib_base_data/calib.right.distortion.txt')
    l2r = np.loadtxt('tests/data/precalib/precalib_base_data/calib.l2r.txt')
    l2r_rmat = l2r[0:3, 0:3]
    l2r_tvec = l2r[0:3, 3]

    calib_dir = 'tests/data/precalib/data_moved_scope'
    calib_driver = get_calib_driver(calib_dir)

    stereo_reproj_err, stereo_recon_err, _ = \
        calib_driver.calibrate(
            override_left_intrinsics=left_intrinsics,
            override_left_distortion=left_distortion,
            override_right_intrinsics=right_intrinsics,
            override_right_distortion=right_distortion,
            override_l2r_rmat=l2r_rmat,
            override_l2r_tvec=l2r_tvec)

    tracked_reproj_err, tracked_recon_err, _ = \
        calib_driver.handeye_calibration()

    print(stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err)
    assert stereo_reproj_err < 4.5
    assert stereo_recon_err < 4.5
    assert tracked_reproj_err < 4.6
    assert tracked_recon_err < 4.6
