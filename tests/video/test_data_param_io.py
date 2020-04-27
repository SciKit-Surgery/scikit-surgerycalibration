# -*- coding: utf-8 -*-

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.processing.chessboard_point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_mono as mc
import sksurgerycalibration.video.video_calibration_driver_stereo as sc


def test_chessboard_mono_io():

    images = []

    files = glob.glob('tests/data/laparoscope_calibration/left/*.png')
    for file in files:
        image = cv2.imread(file)
        images.append(image)

    chessboard_detector = pd.ChessboardPointDetector((14, 10), 3, (1, 1))
    calibrator = mc.MonoVideoCalibrationDriver(chessboard_detector, 140)
    for image in images:
        successful = calibrator.grab_data(image, np.eye(4), np.eye(3))
        assert successful > 0

    reproj_err, recon_err, params_1 = calibrator.calibrate()
    calibrator.save_data('tests/output/test_chessboard_mono_io', '')
    calibrator.save_params('tests/output/test_chessboard_mono_io', '')


def test_chessboard_stereo_io():

    left_images = []
    files = glob.glob('tests/data/laparoscope_calibration/left/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        left_images.append(image)
    assert(len(left_images) == 9)

    right_images = []
    files = glob.glob('tests/data/laparoscope_calibration/right/*.png')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        right_images.append(image)
    assert (len(right_images) == 9)

    chessboard_detector = \
        pd.ChessboardPointDetector((14, 10),
                                   3,
                                   (1, 1)
                                   )

    calibrator = \
        sc.StereoVideoCalibrationDriver(chessboard_detector, 140)

    for i in range(0, len(left_images)):
        successful = calibrator.grab_data(left_images[i], right_images[i], np.eye(4), np.eye(3))
        assert successful > 0

    # Then do calibration
    reproj_err, recon_err, params = calibrator.calibrate()
    calibrator.save_data('tests/output/test_chessboard_stereo_io', '')
    calibrator.save_params('tests/output/test_chessboard_stereo_io', '')