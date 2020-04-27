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
