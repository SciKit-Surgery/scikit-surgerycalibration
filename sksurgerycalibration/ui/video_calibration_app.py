# coding=utf-8

""" Functions to run video calibration. """

import os
import cv2
from sksurgerycore.configuration.configuration_manager import \
        ConfigurationManager
import sksurgeryimage.calibration.chessboard_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_mono as mc

# pylint:disable=too-many-nested-blocks


def run_video_calibration(config_file, save_dir, prefix):
    """
    Performs Video Calibration using OpenCV
    source and scikit-surgerycalibration.

    :param config_file: mandatory location of config file.
    :param save_dir: optional directory name to dump calibrations to.
    :param prefix: file name prefix when saving
    """
    if config_file is None or len(config_file) == 0:
        raise ValueError("Config file must be provided.")
    if save_dir is not None and prefix is None:
        raise ValueError("If you provide -s/--save, "
                         "you must provide -p/--prefix")
    if prefix is not None and save_dir is None:
        raise ValueError("If you provide -p/--prefix, "
                         "you must provide -s/--save")

    configurer = ConfigurationManager(config_file)
    configuration = configurer.get_copy()

    # For now just doing chessboards.
    # The underlying framework works for several point detectors,
    # but each would have their own parameters etc.
    source = configuration.get("source", 1)
    corners = configuration.get("corners", [14, 10])
    corners = (corners[0], corners[1])
    size = configuration.get("square size in mm", 3)
    min_num_views = configuration.get("minimum number of views", 5)

    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    detector = cpd.ChessboardPointDetector(corners, size)
    calibrator = mc.MonoVideoCalibrationDriver(detector,
                                               corners[0] * corners[1])

    print("Press 'q' to quit and 'c' to capture an image.")
    print("Minimum number of views to calibrate:" + str(min_num_views))

    while True:
        _, frame = cap.read()
        cv2.imshow("imshow", frame)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('c'):
            number_points = calibrator.grab_data(frame)
            if number_points > 0:
                corners = calibrator.video_data.image_points_arrays[-1]
                img = cv2.drawChessboardCorners(frame, (8, 6),
                                                corners,
                                                number_points)
                cv2.imshow("detected points", img)

                number_of_views = calibrator.get_number_of_views()
                print("Number of frames = " + str(number_of_views))

                if number_of_views >= min_num_views:
                    proj_err, recon_err, params = calibrator.calibrate()
                    print("Reprojection (2D) error is:" + str(proj_err))
                    print("Reconstruction (3D) error is:" + str(recon_err))
                    print("Intrinsics are:")
                    print(params.camera_matrix)
                    print("Distortion matrix is:")
                    print(params.dist_coeffs)

                    if save_dir:

                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)

                        calibrator.save_data(save_dir, "calib")
                        calibrator.save_params(save_dir, "calib")
            else:
                print("Failed to detect points")

    cap.release()
    cv2.destroyAllWindows()
