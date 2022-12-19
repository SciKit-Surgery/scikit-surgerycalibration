# -*- coding: utf-8 -*-

# pylint: disable=unused-import, superfluous-parens, line-too-long, missing-module-docstring, unused-variable, missing-function-docstring, invalid-name

import glob
import pytest
import numpy as np
import cv2
import sksurgeryimage.calibration.charuco as ch
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc


def test_stereo_davinci():

    left_images = []
    files = glob.glob('tests/data/ChAruco_LR_frames_Steve_Axis_Tests/ExtractedFrames_L/*.jpg')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        print("Loaded:" + str(file))
        left_images.append(image)
    assert(len(left_images) == 59)

    right_images = []
    files = glob.glob('tests/data/ChAruco_LR_frames_Steve_Axis_Tests/ExtractedFrames_R/*.jpg')
    files.sort()
    for file in files:
        image = cv2.imread(file)
        print("Loaded:" + str(file))
        right_images.append(image)
    assert (len(right_images) == 59)

    ref_img = cv2.imread('tests/data/2020_01_20_storz/pattern_4x4_19x26_5_4_with_inset_9x14.png')

    minimum_number_of_points_per_image = 50
    detector = pd.CharucoPlusChessboardPointDetector(ref_img,
                                                     error_if_no_chessboard=False) # Try to accept as many as possible.
    calibrator = sc.StereoVideoCalibrationDriver(detector, detector, minimum_number_of_points_per_image)
    for i, _ in enumerate(left_images):
        try:
            number_left, number_right = calibrator.grab_data(left_images[i], right_images[i])
            if number_left < minimum_number_of_points_per_image:
                print("Image pair:" + str(i) + ", left image, SKIPPED, due to not enough points")
            if number_right < minimum_number_of_points_per_image:
                print("Image pair:" + str(i) + ", right image, SKIPPED, due to not enough points")
        except ValueError as ve:
            print("Image pair:" + str(i) + ", FAILED, due to:" + str(ve))
        except TypeError as te:
            print("Image pair:" + str(i) + ", FAILED, due to:" + str(te))

    reproj_err, recon_err, params = calibrator.calibrate()
    print("Reproj:" + str(reproj_err))
    print("Recon:" + str(recon_err))
    assert reproj_err < 1.1
    assert recon_err < 6.4

    # Now try iterative.
    reference_image = ch.make_charuco_with_chessboard()
    reference_ids, object_pts, reference_pts = detector.get_points(reference_image)
    reproj_err, recon_err, params = calibrator.iterative_calibration(2,
                                                                     reference_ids,
                                                                     reference_pts,
                                                                     (reference_image.shape[1], reference_image.shape[0]))
    print("Reproj:" + str(reproj_err))
    print("Recon:" + str(recon_err))
    assert reproj_err < 0.9
    assert recon_err < 2.78
    calibrator.save_params('tests/output/ChAruco_LR_frames_Steve_Axis_Tests/params', '')
    calibrator.save_data('tests/output/ChAruco_LR_frames_Steve_Axis_Tests/data', '')
