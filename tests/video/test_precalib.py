# pylint: disable=line-too-long, missing-module-docstring, invalid-name, missing-function-docstring

import cv2
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector \
    as charuco_pd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc
import tests.video.test_load_calib_utils as lcu


def get_calib_driver(calib_dir: str):
    """ Create left/right charuco point detectors and load calibration images from directory. """
    minimum_points = 50
    number_of_squares = [19, 26]
    square_tag_sizes = [5, 4]
    number_of_chessboard_squares = [9, 14]
    chessboard_square_size = 3
    chessboard_id_offset = 500

    left_pd = \
        charuco_pd.CharucoPlusChessboardPointDetector(
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            minimum_number_of_points=minimum_points,
            number_of_charuco_squares=number_of_squares,
            size_of_charuco_squares=square_tag_sizes,
            number_of_chessboard_squares=number_of_chessboard_squares,
            chessboard_square_size=chessboard_square_size,
            chessboard_id_offset=chessboard_id_offset
        )

    right_pd = \
        charuco_pd.CharucoPlusChessboardPointDetector(
            dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
            minimum_number_of_points=minimum_points,
            number_of_charuco_squares=number_of_squares,
            size_of_charuco_squares=square_tag_sizes,
            number_of_chessboard_squares=number_of_chessboard_squares,
            chessboard_square_size=chessboard_square_size,
            chessboard_id_offset=chessboard_id_offset
        )

    calibration_driver = sc.StereoVideoCalibrationDriver(left_pd,
                                                         right_pd,
                                                         minimum_points)

    for i in range(3):
        l_img, r_img, chessboard, scope = lcu.get_calib_data(calib_dir, i)
        calibration_driver.grab_data(l_img, r_img, scope, chessboard)

    return calibration_driver


def test_charuco_dataset_A():

    calib_dir = 'tests/data/precalib/precalib_base_data'
    calib_driver = get_calib_driver(calib_dir)

    stereo_reproj_err, stereo_recon_err, _ = \
        calib_driver.calibrate()

    tracked_reproj_err, tracked_recon_err, _ = \
        calib_driver.handeye_calibration(use_opencv=False)

    print(stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err)
    assert stereo_reproj_err < 0.5
    assert stereo_recon_err < 1.5
    assert tracked_reproj_err < 0.5
    assert tracked_recon_err < 1.5


def test_charuco_dataset_B():

    calib_dir = 'tests/data/precalib/data_moved_scope'
    calib_driver = get_calib_driver(calib_dir)

    stereo_reproj_err, stereo_recon_err, _ = \
        calib_driver.calibrate()

    tracked_reproj_err, tracked_recon_err, _ = \
        calib_driver.handeye_calibration(use_opencv=False)

    print(stereo_reproj_err, stereo_recon_err, tracked_reproj_err, tracked_recon_err)
    assert stereo_reproj_err < 0.5
    assert stereo_recon_err < 2.0
    assert tracked_reproj_err < 0.5
    assert tracked_recon_err < 1.0
