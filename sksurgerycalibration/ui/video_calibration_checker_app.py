# coding=utf-8

""" Application to detect chessboards, and assess calibration accuracy. """

import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as cpd
from sksurgerycalibration.video.video_calibration_params import \
                MonoCalibrationParams

# pylint: disable=too-many-branches

def run_video_calibration_checker(configuration = None,
                calib_dir = './', prefix = None):
    """
    Application that detects a calibration pattern, runs
    solvePnP, and prints information out to enable you to
    check how accurate a calibration actually is.

    :param config_file: location of configuration file.
    :param calib_dir: the location of the calibration directory you want to
        check
    :param prefix: the file prefix for the calibration data you want to check

    :raises ValueError: if no configuration provided.
    :raises RuntimeError: if can't open source.
    """
    if configuration is None:
        raise ValueError("Calibration Checker requires a config file")

    source = configuration.get("source", 0)
    corners = configuration.get("corners", [14, 10])
    corners = (corners[0], corners[1])
    size = configuration.get("square size in mm", 3)
    window_size = configuration.get("window size", None)
    keypress_delay = configuration.get("keypress delay", 10)
    interactive = configuration.get("interactive", True)
    sample_frequency = configuration.get("sample frequency", 1)

    existing_calibration = MonoCalibrationParams()
    existing_calibration.load_data(calib_dir, prefix, halt_on_ioerror = False)
    intrinsics = existing_calibration.camera_matrix
    distortion = existing_calibration.dist_coeffs

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera:" + str(source))

    if window_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
        print("Video feed set to ("
              + str(window_size[0]) + " x " + str(window_size[1]) + ")")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Video feed defaults to ("
              + str(width) + " x " + str(height) + ")")

    # For detecting the chessboard points
    detector = cpd.ChessboardPointDetector(corners, size)
    num_pts = corners[0] * corners[1]
    captured_positions = np.zeros((0, 3))
    frames_sampled = 0
    while True:
        frame_ok, frame = cap.read()

        key = None
        frames_sampled += 1

        if not frame_ok:
            print("Reached end of video source or read failure.")
            key = ord('q')
        else:
            undistorted = cv2.undistort(frame, intrinsics, distortion)
            if interactive:
                cv2.imshow("live image", undistorted)
                key = cv2.waitKey(keypress_delay)
            else:
                if frames_sampled % sample_frequency == 0:
                    key = ord('a')

        if key == ord('q'):
            break

        image_points = np.array([])
        if key in [ord('c'), ord('m'), ord('t'), ord('a')]:
            _, object_points, image_points = \
                detector.get_points(undistorted)
            if image_points.shape[0] == 0:
                print("Failed to detect points")

        pnp_ok = False
        img = None
        tvec = None
        if image_points.shape[0] > 0:
            img = cv2.drawChessboardCorners(undistorted, corners,
                                            image_points,
                                            num_pts)
            if interactive:
                cv2.imshow("detected points", img)

            pnp_ok, _, tvec = cv2.solvePnP(object_points,
                                           image_points,
                                           intrinsics,
                                           None)
        if pnp_ok:
            captured_positions = np.append(captured_positions,
                                           np.transpose(tvec),
                                           axis=0)

        if key in [ord('t'), ord('a')] and captured_positions.shape[0] > 1:
            print(str(captured_positions[-1][0]
                  - captured_positions[-2][0]) + " "
                  + str(captured_positions[-1][1]
                  - captured_positions[-2][1]) + " "
                  + str(captured_positions[-1][2]
                  - captured_positions[-2][2]) + " ")
        if key in [ord('m'), ord('a')] and \
                            captured_positions.shape[0] > 1:
            print("Mean:"
                  + str(np.mean(captured_positions, axis=0)))
            print("StdDev:"
                  + str(np.std(captured_positions, axis=0)))

        if key in [ord('c'), ord('a')] and pnp_ok:
            print("Pose" + str(tvec[0][0]) + " "
                  + str(tvec[1][0]) + " "
                  + str(tvec[2][0]))

        if not pnp_ok and image_points.shape[0] > 0:
            print("Failed to solve PnP")

    cap.release()
    cv2.destroyAllWindows()
