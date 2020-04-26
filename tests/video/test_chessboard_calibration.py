# -*- coding: utf-8 -*-

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.processing.chessboard_point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_mono as mc


def test_chessboard_mono():

    images = []

    files = glob.glob('tests/data/laparoscope_calibration/left/*.png')
    for file in files:
        image = cv2.imread(file)
        images.append(image)

    chessboard_detector = \
        pd.ChessboardPointDetector((14, 10), # (Width, height), number of internal corners
                                   3,        # Actual square size in mm
                                   (1, 1)    # Scale factors. Here, images are 1920x1080 so no scaling needed.
                                   )         # If images were 1920x540, you'd pass in (1920, 540),
                                             # and PointDetector base class would scale it up.

    calibrator = \
        mc.MonoVideoCalibration(chessboard_detector, 140)

    for image in images:
        successful = calibrator.grab_data(image)
        assert successful > 0

    # Extra checking, as its a unit test
    assert calibrator.get_number_of_views() == 9
    calibrator.pop()
    assert calibrator.get_number_of_views() == 8
    successful = calibrator.grab_data(images[-1])
    assert successful
    assert calibrator.get_number_of_views() == 9

    reproj_err, recon_err, params = calibrator.calibrate()

    print(reproj_err)
    print(recon_err)
    print(params)



