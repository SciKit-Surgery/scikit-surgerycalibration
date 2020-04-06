# -*- coding: utf-8 -*-

import glob
import pytest
import numpy as np
import sksurgerycalibration.video.video_calibration as vc


def test_mono_calibration():

    image_points = []
    object_points = []

    model = np.loadtxt('tests/data/laparoscope_calibration/chessboard_14_10_3.txt')

    files = glob.glob('tests/data/laparoscope_calibration/left/*.txt')
    for file in files:
        points = np.loadtxt(file)
        image_points.append(vc.convert_numpy2d_to_opencv(points))
        object_points.append(vc.convert_numpy3d_to_opencv(model))

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = \
        vc.mono_video_calibration(object_points, image_points, (1920, 1080))

    print(camera_matrix)


def test_stereo_calibration():

    left_image_points = []
    right_image_points = []
    object_points = []

    model = np.loadtxt('tests/data/laparoscope_calibration/chessboard_14_10_3.txt')

    files = glob.glob('tests/data/laparoscope_calibration/left/*.txt')
    files.sort()
    for file in files:
        points = np.loadtxt(file)
        left_image_points.append(vc.convert_numpy2d_to_opencv(points))
        object_points.append(vc.convert_numpy3d_to_opencv(model))

    files = glob.glob('tests/data/laparoscope_calibration/right/*.txt')
    files.sort()
    for file in files:
        points = np.loadtxt(file)
        right_image_points.append(vc.convert_numpy2d_to_opencv(points))

    # Generates, array of arrays
    ids = []
    for i in range(9):
        ids.append(np.asarray(range(140)))

    s_rms, l_c, l_d, r_c, r_d, l2r_r, l2r_t, essential, fundamental\
        = vc.stereo_video_calibration(ids,
                                      object_points,
                                      left_image_points,
                                      ids,
                                      object_points,
                                      right_image_points,
                                      (1920, 1080))
    print(l2r_r)
    print(l2r_t)

