# coding=utf-8

""" Application to detect chessboards, and assess calibration accuracy. """

import numpy as np
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as cpd
from sksurgerycalibration.video.video_calibration_params import \
                MonoCalibrationParams

# pylint: disable=too-many-branches

def run_video_calibration_checker(configuration, calib_dir, prefix):
    """
    Application that detects a calibration pattern, runs
    solvePnP, and prints information out to enable you to
    check how accurate a calibration actually is.

    :param config_file: mandatory location of config file, containing params.
    """
    if configuration is None:
        raise ValueError("Configuration must be provided.")
    if calib_dir is None:
        raise ValueError("Calibration directory must be specified")
    if prefix is None:
        raise ValueError("Calibration directory must be specified")

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
    handeye = existing_calibration.handeye_matrix

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
    while True:
        _, frame = cap.read()
        undistorted = cv2.undistort(frame, intrinsics, distortion)
        key = cv2.waitKey(keypress_delay)
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
