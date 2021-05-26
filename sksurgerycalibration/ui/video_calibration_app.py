# coding=utf-8

""" Functions to run video calibration. """

import os
import cv2
import sksurgeryimage.calibration.chessboard_point_detector as cpd
import sksurgerycalibration.video.video_calibration_driver_mono as mc

# pylint:disable=too-many-nested-blocks,too-many-branches


def run_video_calibration(configuration = {}, save_dir = None, prefix = None):
    """
    Performs Video Calibration using OpenCV
    source and scikit-surgerycalibration.
    Currently only chessboards are supported

    :param config_file: mandatory location of config file.
    :param save_dir: optional directory name to dump calibrations to.
    :param prefix: file name prefix when saving

    :raises ValueError: if configuration is None or invalid
    """
    if prefix is not None and save_dir is None:
        save_dir = "./"

    # For now just doing chessboards.
    # The underlying framework works for several point detectors,
    # but each would have their own parameters etc.
    method = configuration.get("method", "chessboard")
    if method != "chessboard":
        raise ValueError("Only chessboard calibration is currently supported")

    source = configuration.get("source", 0)
    corners = configuration.get("corners", [14, 10])
    corners = (corners[0], corners[1])
    size = configuration.get("square size in mm", 3)
    min_num_views = configuration.get("minimum number of views", 5)
    keypress_delay = configuration.get("keypress delay", 10)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera.")

    window_size = configuration.get("window size", None)
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

    detector = cpd.ChessboardPointDetector(corners, size)
    calibrator = mc.MonoVideoCalibrationDriver(detector,
                                               corners[0] * corners[1])

    print("Press 'q' to quit and 'c' to capture an image.")
    print("Minimum number of views to calibrate:" + str(min_num_views))

    while True:
        frame_ok, frame = cap.read()
        
        if not frame_ok:
            print("Reached end of video source or read failure.")
            break

        cv2.imshow("live image", frame)
        key = cv2.waitKey(keypress_delay)
        if key == ord('q'):
            break
        if key == ord('c'):
            number_points = calibrator.grab_data(frame)
            if number_points > 0:
                img_pts = calibrator.video_data.image_points_arrays[-1]
                img = cv2.drawChessboardCorners(frame, corners,
                                                img_pts,
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

                    if save_dir is not None:
                        if not os.path.isdir(save_dir):
                            os.makedirs(save_dir)

                        calibrator.save_data(save_dir, prefix)
                        calibrator.save_params(save_dir, prefix)
            else:
                print("Failed to detect points")

    cap.release()
    cv2.destroyAllWindows()
