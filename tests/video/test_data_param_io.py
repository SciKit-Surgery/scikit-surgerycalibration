# -*- coding: utf-8 -*-

# pylint: disable=unused-import, superfluous-parens, line-too-long, missing-module-docstring, unused-variable, missing-function-docstring, invalid-name

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as pd
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

    reproj_err_1, recon_err_1, params_1 = calibrator.calibrate()
    calibrator.save_data('tests/output/test_chessboard_mono_io', '')
    calibrator.save_params('tests/output/test_chessboard_mono_io', '')

    # Now, load data back in, recalibrate, should get same results.
    # Technically, you aren't running the 'grab' bit again.
    # The calibration works off the already extracted points.
    calibrator.load_data('tests/output/test_chessboard_mono_io', '')
    reproj_err_2, recon_err_2, params_2 = calibrator.calibrate()
    assert (np.isclose(reproj_err_1, reproj_err_2))
    assert (np.isclose(recon_err_1, recon_err_2))

    calibrator.load_params('tests/output/test_chessboard_mono_io', '')
    params_3 = calibrator.get_params()
    assert np.allclose(params_2.camera_matrix, params_3.camera_matrix)
    assert np.allclose(params_2.dist_coeffs, params_3.dist_coeffs)
    for i, _ in enumerate(params_3.rvecs):
        assert np.allclose(params_2.rvecs[i], params_3.rvecs[i])
        assert np.allclose(params_2.tvecs[i], params_3.tvecs[i])


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

    for i, _ in enumerate(left_images):
        successful = calibrator.grab_data(left_images[i], right_images[i], np.eye(4), np.eye(3))
        assert successful > 0

    # Then do calibration
    reproj_err_1, recon_err_1, params_1 = calibrator.calibrate()
    calibrator.save_data('tests/output/test_chessboard_stereo_io', '')
    calibrator.save_params('tests/output/test_chessboard_stereo_io', '')

    # Load data, re-do calibration, check for same result.
    calibrator.load_data('tests/output/test_chessboard_stereo_io', '')
    reproj_err_2, recon_err_2, params_2 = calibrator.calibrate()
    assert (np.isclose(reproj_err_1, reproj_err_2))
    assert (np.isclose(recon_err_1, recon_err_2))

    # Now load parameters back in.
    calibrator.load_params('tests/output/test_chessboard_stereo_io', '')
    params_2 = calibrator.get_params()

    assert np.allclose(params_1.l2r_rmat,
                       params_2.l2r_rmat)
    assert np.allclose(params_1.l2r_tvec,
                       params_2.l2r_tvec)
    assert np.allclose(params_1.essential,
                       params_2.essential)
    assert np.allclose(params_1.fundamental,
                       params_2.fundamental)
