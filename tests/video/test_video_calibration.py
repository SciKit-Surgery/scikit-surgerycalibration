# -*- coding: utf-8 -*-

# pylint: disable=unused-import, superfluous-parens, line-too-long, missing-module-docstring, unused-variable, missing-function-docstring, invalid-name

import glob
import pytest
import numpy as np
import sksurgerycalibration.video.video_calibration_utils as vu
import sksurgerycalibration.video.video_calibration_wrapper as vc


def test_mono_left_video_calibration():

    image_points = []
    object_points = []

    model = np.loadtxt('tests/data/laparoscope_calibration/chessboard_14_10_3.txt')

    files = glob.glob('tests/data/laparoscope_calibration/left/*.txt')
    for file in files:
        points = np.loadtxt(file)
        image_points.append(vu.convert_numpy2d_to_opencv(points))
        object_points.append(vu.convert_numpy3d_to_opencv(model))

    retval, camera_matrix, dist_coeffs, rvecs, tvecs = \
        vc.mono_video_calibration(object_points, image_points, (1920, 1080))

    assert(np.abs(retval - 0.57759896) < 0.000001)


def load_first_stereo_data():

    left_image_points = []
    right_image_points = []
    object_points = []

    model = np.loadtxt('tests/data/laparoscope_calibration/chessboard_14_10_3.txt')

    files = glob.glob('tests/data/laparoscope_calibration/left/*.txt')
    files.sort()
    for file in files:
        points = np.loadtxt(file)
        left_image_points.append(vu.convert_numpy2d_to_opencv(points))
        object_points.append(vu.convert_numpy3d_to_opencv(model))

    files = glob.glob('tests/data/laparoscope_calibration/right/*.txt')
    files.sort()
    for file in files:
        points = np.loadtxt(file)
        right_image_points.append(vu.convert_numpy2d_to_opencv(points))

    # Generates, array of arrays
    ids = []
    for i in range(9):
        ids.append(np.asarray(range(140)))

    return object_points, \
           left_image_points, \
           right_image_points, \
           ids


def test_stereo_video_calibration():

    object_points, left_image_points, right_image_points, ids = \
        load_first_stereo_data()

    s_reproj, s_recon, \
        l_c, l_d, left_rvecs, left_tvecs, \
        r_c, r_d, right_rvecs, right_tvecs, \
        l2r_r, l2r_t, \
        essential, fundamental = \
        vc.stereo_video_calibration(ids,
                                    object_points,
                                    left_image_points,
                                    ids,
                                    object_points,
                                    right_image_points,
                                    (1920, 1080))

    assert (np.abs(s_reproj - 0.63022577) < 0.000001)
    assert (np.abs(s_recon - 1.64274596) < 0.000001)

# def test_experimental_mono_stereo_calib():
#
#   This isn't worth the bother. Takes 5 minutes to run.
#
#    image_points = []
#    object_points = []
#
#    model = np.loadtxt('tests/data/laparoscope_calibration/chessboard_14_10_3.txt')
#
#    files = glob.glob('tests/data/laparoscope_calibration/left/*.txt')
#    for file in files:
#        points = np.loadtxt(file)
#        image_points.append(vu.convert_numpy2d_to_opencv(points))
#        object_points.append(vu.convert_numpy3d_to_opencv(model))
#
#    # Generates, array of arrays
#    ids = []
#    for i in range(9):
#        ids.append(np.asarray(range(140)))
#
#    retval, camera_matrix, dist_coeffs, rvecs, tvecs = \
#        ve.mono_video_calibration_expt(ids, object_points, image_points, (1920, 1080))
#
#    print(camera_matrix)
#    print(dist_coeffs)
#    print(rvecs)
#    print(tvecs)

#def test_experimental_video_stereo_calib():
#
#   This isn't worth the bother. Takes 1.5 hours to run.
#
#    object_points, left_image_points, right_image_points, ids = \
#        load_first_stereo_data()
#
#    s_reproj, s_recon, \
#        l_c, l_d, left_rvecs, left_tvecs, \
#        r_c, r_d, right_rvecs, right_tvecs, \
#        l2r_r, l2r_t, \
#        essential, fundamental = \
#        ve.stereo_video_calibration_expt(ids,
#                                         object_points,
#                                         left_image_points,
#                                         ids,
#                                         object_points,
#                                         right_image_points,
#                                         (1920, 1080))
#    print(s_reproj)
#    print(s_recon)
#    print(l2r_r)
#    print(l2r_t)
