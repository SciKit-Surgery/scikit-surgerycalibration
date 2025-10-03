# -*- coding: utf-8 -*-

# pylint:disable=line-too-long

""" Various tests to test the SmartLiver calibration using dotty pattern. """

import os
import numpy as np
import cv2
import sksurgerycore.transforms.matrix as skcm
import sksurgerycore.algorithms.procrustes as pbr
import sksurgeryimage.calibration.dotty_grid_point_detector as dgpd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc
import tests.video.test_load_calib_utils as lcu


def get_dotty_calib_driver(calib_dir: str):
    """
    Utility function to setup stereo calibration driver and load data.
    """
    minimum_points_per_frame = 36
    number_of_dots = [18, 25]
    dot_separation = 5
    fiducial_indexes = [133, 141, 308, 316]
    reference_image_size_in_pixels = [1900, 2600]
    pixels_per_mm = 20

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

    left_intrinsic_matrix = np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.left.intrinsics.txt")
    left_distortion_matrix = np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.left.distortion.txt")
    right_intrinsic_matrix = np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.right.intrinsics.txt")
    right_distortion_matrix = np.loadtxt("tests/data/laparoscope_calibration/cbh-viking/calib.right.distortion.txt")

    threshold_offset = 20
    threshold_window_size = 151

    left_pd = dgpd.DottyGridPointDetector(model_points,
                                          list_of_indexes=fiducial_indexes,
                                          camera_intrinsics=left_intrinsic_matrix,
                                          distortion_coefficients=left_distortion_matrix,
                                          reference_image_size=reference_image_size_in_pixels,
                                          threshold_window_size=threshold_window_size,
                                          threshold_offset=threshold_offset
                                          )

    right_pd = dgpd.DottyGridPointDetector(model_points,
                                           list_of_indexes=fiducial_indexes,
                                           camera_intrinsics=right_intrinsic_matrix,
                                           distortion_coefficients=right_distortion_matrix,
                                           reference_image_size=reference_image_size_in_pixels,
                                           threshold_window_size=threshold_window_size,
                                           threshold_offset=threshold_offset
                                           )

    calibration_driver = sc.StereoVideoCalibrationDriver(left_pd,
                                                         right_pd,
                                                         minimum_points_per_frame)

    for i in range(0,10,2): #issue 59, reduced number of frames used for test
        l_img, r_img, chessboard, scope = lcu.get_calib_data(calib_dir, i)
        num_points = calibration_driver.grab_data(l_img, r_img, scope, chessboard)
        print("Grabbed " + str(num_points) + " points")

    return calibration_driver


def get_pattern_to_marker():
    """
    Creates a pattern 2 marker transform for the SmartLiver calibration pattern.
    """
    plate_points = np.zeros((4, 3))
    plate_points[1][0] = 120
    plate_points[2][1] = 85
    plate_points[3][0] = 120
    plate_points[3][1] = 85

    marker_points = np.ones((4, 3))
    marker_points[0][0] = -21.77
    marker_points[0][2] = -19.25
    marker_points[1][0] = -21.77
    marker_points[1][2] = 100.75
    marker_points[2][0] = -106.77
    marker_points[2][2] = -19.25
    marker_points[3][0] = -106.77
    marker_points[3][2] = 100.75

    p2m_r, p2m_t, _ = pbr.orthogonal_procrustes(marker_points, plate_points)

    pattern2marker = skcm.construct_rigid_transformation(p2m_r, p2m_t)
    return pattern2marker


def get_projection_errs_and_params(dir_name,
                                   pattern2marker,
                                   use_open_cv,
                                   save_data=False
                                   ):
    """
    Calibration driver.
    """
    print("dir_name=" + dir_name)

    calib_driver = get_dotty_calib_driver(dir_name)

    reproj_err, recon_err, _ = calib_driver.calibrate()

    print("After initial calibration, reproj_err=" + str(reproj_err) + ", recon_err=" + str(recon_err))

    tracked_reproj_err, tracked_recon_err, params_2 = \
        calib_driver.handeye_calibration(override_pattern2marker=pattern2marker,
                                         use_opencv=use_open_cv
                                         )

    print("After handeye calibration, tracked_reproj_err=" + str(tracked_reproj_err) + ", tracked_recon_err=" + str(tracked_recon_err))
    p2m = params_2.left_params.pattern2marker_matrix
    if p2m is not None:
        print("p2m=" + str(np.transpose(p2m[0:3, 3])))
    print("h2e=" + str(np.transpose(params_2.left_params.handeye_matrix[0:3, 3])))

    if save_data:
        output_calib_dir = os.path.join('tests', 'output', 'test_handeye_dots', dir_name)
        calib_driver.save_data(output_calib_dir, 'calib')
        calib_driver.save_params(output_calib_dir, 'calib')

    return reproj_err, recon_err, tracked_reproj_err, tracked_recon_err, params_2


def get_dirs(is_paper=False):
    """
    Just gets some data directories.
    """
    dirs = []
    if is_paper:
        dirs.append('tests/data/2022_02_11_calibration_viking_paper/18_36_09')
        dirs.append('tests/data/2022_02_11_calibration_viking_paper/18_41_28')
        dirs.append('tests/data/2022_02_11_calibration_viking_paper/18_44_06')
    else:
#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/15_56_22')
#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/15_57_13')
#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/15_58_14')

#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/16_13_39')
#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/16_20_03')
#        dirs.append('tests/data/2022_02_13_calibration_viking_metal/16_24_24')

        dirs.append('tests/data/2022_02_28_calibration_viking_metal/calibration/14_58_31')
        dirs.append('tests/data/2022_02_28_calibration_viking_metal/calibration/15_18_54')
        dirs.append('tests/data/2022_02_28_calibration_viking_metal/calibration/15_22_44')

    return dirs


def get_fixed_pattern_dirs():
    """
    Just gets some data directories.
    """
    dirs = []
    #dirs.append('tests/data/2022_02_28_fixed_position_calibs/calibration/14_58_31')
    dirs.append('tests/data/2022_02_28_fixed_position_calibs/calibration/15_18_54')
#    dirs.append('tests/data/2022_02_28_fixed_position_calibs/calibration/15_22_44')
    return dirs


def get_calibrations(dirs, pattern2marker, use_opencv, save_data=False):
    """
    Calibration driver.
    """
    results = np.zeros((len(dirs), 8))
    counter = 0
    for dir_name in dirs:
        _, _, tracked_reproj_err, tracked_recon_err, params_2 = \
            get_projection_errs_and_params(dir_name, pattern2marker, use_opencv, save_data)
        rvec = np.zeros((3,1))
        cv2.Rodrigues(params_2.left_params.handeye_matrix[0:3, 0:3], rvec)
        results[counter][0] = tracked_reproj_err
        results[counter][1] = tracked_recon_err
        results[counter][2] = params_2.left_params.handeye_matrix[0][3]
        results[counter][3] = params_2.left_params.handeye_matrix[1][3]
        results[counter][4] = params_2.left_params.handeye_matrix[2][3]
        results[counter][5] = rvec[0][0]
        results[counter][6] = rvec[1][0]
        results[counter][7] = rvec[2][0]
        counter = counter + 1
    return results


def test_gx_vs_cv():
    """
    Compares consistency of Guofang's method, with OpenCV.
    """
    dirs = get_dirs()
    results_gx = get_calibrations(dirs, None, False, False)
    results_gx_m = np.mean(results_gx, axis=0)
    results_gx_s = np.std(results_gx, axis=0)
    results_opencv = get_calibrations(dirs, None, True, False)
    results_opencv_m = np.mean(results_opencv, axis=0)
    results_opencv_s = np.std(results_opencv, axis=0)

    print("Guofang's mean=\n" + str(results_gx_m))
    print("OpenCV's mean=\n" + str(results_opencv_m))

    print("Guofang's stddev=\n" + str(results_gx_s))
    print("OpenCV's stddev=\n" + str(results_opencv_s))

    # 1. Just looking at data, not much obvious difference.
    # 2. Interestingly, looking at stddev of rotation and translation of
    #    hand-eye, most difference is translational.
    # 3. Ultimately, chose OpenCV, as default, as then
    #    we can have more consistent implementation across different use-cases.

    # Just for regression testing:

    # Projection error < 4, should be ok for a stereo laparoscope
    assert results_gx_m[0] < 4
    assert results_opencv_m[0] < 4

    # Reconstruction error < 1mm
    assert results_gx_m[1] < 1
    assert results_opencv_m[1] < 1


def test_fixed_p2m():
    """
    Compares consistency of having a fixed pattern to marker.
    """
    dirs = get_dirs()
    p2m = get_pattern_to_marker()
    results_no_p2m = get_calibrations(dirs, None, True, True)
    results_no_p2m_m = np.mean(results_no_p2m, axis=0)
    results_no_p2m_s = np.std(results_no_p2m, axis=0)
    results_with_p2m = get_calibrations(dirs, p2m, True, True)
    results_with_p2m_m = np.mean(results_with_p2m, axis=0)
    results_with_p2m_s = np.std(results_with_p2m, axis=0)

    print("No p2m mean=\n" + str(results_no_p2m_m))
    print("With p2m mean=\n" + str(results_with_p2m_m))

    print("No p2m stddev=\n" + str(results_no_p2m_s))
    print("With p2m stddev=\n" + str(results_with_p2m_s))

    # Just for regression testing:

    # Projection error < 4, should be ok for a stereo laparoscope
    assert results_no_p2m_m[0] < 4
    assert results_with_p2m_m[0] < 4

    # Reconstruction error < 1mm
    assert results_no_p2m_m[1] < 1
    assert results_with_p2m_m[1] < 1.2


def test_tracked_vs_stationary():
    """
    Mainly to check code runs, as we don't yet have functional requirements.
    Unfortunately, below, the stationary pattern calibrations fail.
    I suspect there was not enough movement of the laparoscope.

    Needs more data. Not a complete unit test, as we aren't
    testing anything other than whether the code runs without failing.
    """
    dirs = get_fixed_pattern_dirs()
    results_fixed = get_calibrations(dirs, None, True, False)
    results_fixed_m = np.mean(results_fixed, axis=0)
    results_fixed_s = np.std(results_fixed, axis=0)
    dirs = get_dirs()
    results_opencv = get_calibrations(dirs, None, True, False)
    results_opencv_m = np.mean(results_opencv, axis=0)
    results_opencv_s = np.std(results_opencv, axis=0)

    print("Fixed pattern mean=\n" + str(results_fixed_m))
    print("OpenCV's mean=\n" + str(results_opencv_m))

    print("Fixed pattern stddev=\n" + str(results_fixed_s))
    print("OpenCV's stddev=\n" + str(results_opencv_s))


def tests_for_smart_liver():
    """
    Additional tests for SmartLiver laparoscope system.
    """
    output_dir = 'tests/output/SmartLiver/'
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    p2m = get_pattern_to_marker()
    np.savetxt(os.path.join(output_dir, 'p2m.txt'), p2m)

    # For documentation, this should re-run calibration, as SmartLiver has it, so should match the input folder.
    results_sl = get_calibrations(
        ['tests/data/2022_02_28_fixed_position_calibs/calibration/14_58_31'],
        pattern2marker=None,
        use_opencv=False,
        save_data=False)

    # This should save a new calibration in tests/output,
    # as an example of using a fixed pattern2marker, and OpenCV methods.
    results_2022_03_04 = get_calibrations(
        ['tests/data/2022_02_28_fixed_position_calibs/calibration/14_58_31'],
        pattern2marker=p2m,
        use_opencv=True,
        save_data=True)

    print("SmartLiver, proj=" + str(results_sl[0][0]) + ", recon=" + str(results_sl[0][1]))
    print("As of 2022-03-04, proj=" + str(results_2022_03_04[0][0]) + ", recon=" + str(results_2022_03_04[0][1]))
