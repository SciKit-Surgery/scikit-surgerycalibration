# coding=utf-8

""" Application to detect chessboards, and assess calibration accuracy. """

import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as cpd
from sksurgerybard.algorithms.bard_config_algorithms import \
    configure_camera, replace_calibration_dir


# pylint: disable=too-many-branches

def run_video_calibration_checker(configuration, calib_dir):
    """
    Simple app that detects a calibration pattern, runs
    solvePnP, and prints information out to enable you to
    check how accurate a calibration actually is.

    :param config_file: mandatory location of config file, containing params.
    """
    if config_file is None or len(config_file) == 0:
        raise ValueError("Config file must be provided.")
    if calib_dir is None or len(calib_dir) == 0:
        raise ValueError("Calibration directory must be specified")

    configuration = replace_calibration_dir(configuration, calib_dir)

    source = configuration.get("source", 1)
    corners = configuration.get("corners")
    if corners is None:
        raise ValueError("You must specify the number of internal corners")
    corners = (corners[0], corners[1])
    size = configuration.get("square size in mm")
    if size is None:
        raise ValueError("You must specify the size of each square in mm.")

    window_size = configuration.get("window size")
    if window_size is None:
        raise ValueError("You must specify the window size.")

    _, intrinsics, distortion, _ = configure_camera(configuration)
    if intrinsics is None:
        raise ValueError("Couldn't load intrinsic parameters")
    if distortion is None:
        raise ValueError("Couldn't load distortion parameters")

    cap = cv2.VideoCapture(int(source))
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera:" + str(source))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_size[1])
    print("Video feed set to ("
          + str(window_size[0]) + " x " + str(window_size[1]) + ")")

    # For detecting the chessboard points
    detector = cpd.ChessboardPointDetector(corners, size)
    num_pts = corners[0] * corners[1]
    captured_positions = np.zeros((0, 3))
    while True:
        _, frame = cap.read()
        undistorted = cv2.undistort(frame, intrinsics, distortion)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
        if key == ord('c') or key == ord('m') or key == ord('t'):
            _, object_points, image_points = detector.get_points(undistorted)
            if image_points.shape[0] > 0:
                img = cv2.drawChessboardCorners(undistorted, corners,
                                                image_points,
                                                num_pts)
                retval, _, tvec = cv2.solvePnP(object_points,
                                               image_points,
                                               intrinsics,
                                               None)
                if retval:
                    captured_positions = np.append(captured_positions,
                                                   np.transpose(tvec),
                                                   axis=0)
                    cv2.imshow("detected points", img)
                    if key == ord('t') and captured_positions.shape[0] > 1:
                        print(
                            str(captured_positions[-1][0]
                                - captured_positions[-2][0]) + " "
                            + str(captured_positions[-1][1]
                                  - captured_positions[-2][1]) + " "
                            + str(captured_positions[-1][2]
                                  - captured_positions[-2][2]) + " ")
                    elif key == ord('m'):
                        print("Mean:"
                              + str(np.mean(captured_positions, axis=0)))
                        print("StdDev:"
                              + str(np.std(captured_positions, axis=0)))
                    else:
                        print(str(tvec[0][0]) + " "
                              + str(tvec[1][0]) + " "
                              + str(tvec[2][0]))
                else:
                    print("Failed to solve PnP")
            else:
                print("Failed to detect points")
        else:
            cv2.imshow("live image", undistorted)

    cap.release()
    cv2.destroyAllWindows()
