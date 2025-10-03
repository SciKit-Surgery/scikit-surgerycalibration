# -*- coding: utf-8 -*-

""" Various routines for Hand-Eye calibration. """

# pylint:disable=invalid-name,consider-using-enumerate

import logging
from typing import List, Tuple

import cv2
import numpy as np
import sksurgerycore.transforms.matrix as mu
from scipy.optimize import least_squares

import sksurgerycalibration.video.quaternion_utils as qu
import sksurgerycalibration.video.video_calibration_utils as vu

LOGGER = logging.getLogger(__name__)


def set_model2hand_arrays(calibration_tracking_array: List,
                          device_tracking_array: List,
                          use_quaternions=False) \
                            -> Tuple[np.ndarray, np.ndarray]:
    """
    Guofang Xiao's method. Set the model-to-hand quaternion
    and translation arrays from tracking data.

    :param calibration_tracking_array: Array of tracking data for
    calibration target
    :type calibration_tracking_array: List of tracking data
    :param device_tracking_array: Array of tracking data for
    device (e.g. camera)
    :type device_tracking_array: List of tracking data
    :param use_quaternions: If True input should be quaternions
    :type device_tracking_array: bool
    :return: quaternion model to hand array and translation model to hand
    array
    :rtype: np.ndarray, np.ndarray
    """

    if len(calibration_tracking_array) != len(device_tracking_array):
        raise ValueError('Calibration target and device tracking array \
                            should be the same size')

    number_of_frames = len(calibration_tracking_array)

    quat_model2hand_array = np.zeros((number_of_frames, 4))
    trans_model2hand_array = np.zeros((number_of_frames, 3))

    for i in range(number_of_frames):
        if use_quaternions:
            quat_model = \
                calibration_tracking_array[i][0, 0:4]
            trans_model = \
                calibration_tracking_array[i][0, 4:7]
            quat_hand = \
                device_tracking_array[i][0, 0:4]
            trans_hand = \
                device_tracking_array[i][0, 4:7]

            quat_model2hand = qu.quat_multiply(qu.quat_conjugate(quat_hand),
                                            quat_model)
            q_translation = np.append([0], (trans_model - trans_hand))
            quat_hand_conj = qu.quat_conjugate(quat_hand)

            trans_model2hand = qu.quat_multiply(quat_hand_conj,
                                                qu.quat_multiply(q_translation,
                                                                 quat_hand))
            trans_model2hand = trans_model2hand[1:4]
        else:
            tracking_model = calibration_tracking_array[i]
            tracking_hand = device_tracking_array[i]
            model_to_hand = np.linalg.inv(tracking_hand) @ tracking_model

            quat_model2hand = qu.rotm_to_quat(model_to_hand[0:3, 0:3])
            trans_model2hand = model_to_hand[0:3, 3]

        quat_model2hand_array[i] = quat_model2hand
        trans_model2hand_array[i] = trans_model2hand

    quat_model2hand_array = \
        qu.to_one_hemisphere(quat_model2hand_array)

    return quat_model2hand_array, trans_model2hand_array


def _gx_solve_2quaternions(qx, q_A, q_B):
    """
    Guofang Xiao's method. Provide cost function for least-square optimisation
    to solve quaternions qx1 and qx2 in: q_B = qx1 * q_A * qx2

    :param qx: 1x8 vector of 2 quaternions
    :param q_A: Nx4 matrices of N quaternions
    :param q_B: Nx4 matrices of N quaternions
    :return:
    """
    if np.shape(q_A) != np.shape(q_B):
        raise AssertionError("q_A and q_B must have the same shape.")

    number_of_frames = np.shape(q_A)[0]

    f = []

    qx_1 = qx[0:4]
    qx_2 = qx[4:8]

    qx_1 = qx_1/np.linalg.norm(qx_1)
    qx_2 = qx_2/np.linalg.norm(qx_2)

    for i in range(0, number_of_frames):
        f.append(qu.quat_multiply(qu.quat_multiply(qx_1, q_A[i]), qx_2)
                 - q_B[i])

    f = np.ndarray.flatten(np.asarray(f))

    return f


def _gx_solve_2translations(q_handeye,
                            quat_model2hand_array,
                            trans_model2hand_array,
                            trans_extrinsics_array):
    """
    Guofang Xiao's method. Solve translations t_handeye and t_pattern2marker.
    translations = [t_handeye t_pattern2marker]

    :param q_handeye: 1x4 matrix quaternion
    :param quat_model2hand_array: Nx4 matrix of N quaternions
    :param trans_model2hand_array: Nx3 matrices of N translations
    :param trans_extrinsics_array: Nx3 matrices of extrinsic translation vectors
    :return:
    """
    if np.shape(quat_model2hand_array)[0] \
            != np.shape(trans_model2hand_array)[0]:
        raise AssertionError("Wrong dimensions in solve_2translations().")
    if np.shape(trans_model2hand_array) != np.shape(trans_extrinsics_array):
        raise AssertionError("Wrong dimensions in solve_2translations().")

    number_of_frames = np.shape(quat_model2hand_array)[0]

    A = np.zeros((3 * number_of_frames, 6))
    B = np.zeros((3 * number_of_frames, 1))

    for i in range(0, number_of_frames):
        qHEqMH = qu.quat_multiply(q_handeye, quat_model2hand_array[i])
        rHErMH = qu.quat_to_rotm(qHEqMH)
        rHE = qu.quat_to_rotm(q_handeye)

        A[i*3:i*3+3, :] = np.hstack([np.identity(3), rHErMH])
        b_i = np.transpose(trans_extrinsics_array[i, :])\
            - np.matmul(rHE, np.transpose(trans_model2hand_array[i, :]))
        B[i*3:i*3+3] = np.reshape(b_i, (3, 1))

    A_T = np.transpose(A)
    translations = np.linalg.inv(A_T @ A) @ A_T @ B
    translations = np.transpose(translations)

    return translations.flatten()


def _gx_handeye_optimisation(
        quat_extrinsics_array: np.ndarray,
        trans_extrinsics_array: np.ndarray,
        quat_model2hand_array: np.ndarray,
        trans_model2hand_array: np.ndarray) ->  \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Guofang Xiao's method. Solve handeye and pattern-to-marker transformations.

    :param quat_extrinsics_array: An array of quaternions representing the
        rotations of the camera extrinsics.
    :type quat_extrinsics_array: np.ndarray
    :param trans_extrinsics_array:  Array of the translation vectors of the
        camera extrinsics.
    :type trans_extrinsics_array: np.ndarray
    :param quat_model2hand_array: An array of model to hand quaternions.
        :type quat_model2hand_array: np.ndarray
    :param quat_model2hand_array: An array of model to hand
        translations arrays.
        :type quat_model2hand_array: np.ndarray
    :return: rotations in quaternion form and translations of the handeye
        pattern-to-marker transformation.
    :rtype: np.ndarray
    """

    # self.quat_model2hand_array should already be in one hemisphere.

    quat_extrinsics_array = qu.to_one_hemisphere(quat_extrinsics_array)

    qx_0 = [1, 0, 0, 0, 1, 0, 0, 0]
    lb = [-1, -1, -1, -1, -1, -1, -1, -1]
    ub = [1, 1, 1, 1, 1, 1, 1, 1]

    op_result = least_squares(_gx_solve_2quaternions, qx_0,
                              bounds=(lb, ub),
                              args=(quat_model2hand_array,
                                    quat_extrinsics_array)
                              )

    q_handeye = op_result.x[0:4]
    q_pattern2marker = op_result.x[4:8]

    q_handeye = q_handeye / np.linalg.norm(q_handeye)
    q_pattern2marker = q_pattern2marker / np.linalg.norm(q_pattern2marker)

    translations = _gx_solve_2translations(q_handeye,
                                           quat_model2hand_array,
                                           trans_model2hand_array,
                                           trans_extrinsics_array)

    t_handeye = translations[0:3]
    t_pattern2marker = translations[3:6]

    return q_handeye, t_handeye, q_pattern2marker, t_pattern2marker


def guofang_xiao_handeye_calibration(rvecs: List[np.ndarray],
                                     tvecs: List[np.ndarray],
                                     quat_model2hand_array: np.ndarray,
                                     trans_model2hand_array: np.ndarray) \
                                     -> Tuple[np.ndarray, np.ndarray]:
    """
    Guofang Xiao's method. Solve for the hand-eye transformation,
    as well as the transformation from the pattern to the tracking markers
    on the calibration object.

    This solves rotation and then translation in two steps, using least-squares
    optimisation. It uses quaternions to represent rotations.

    This method, developed for the SmartLiver project, assumes both
    device (laparoscope), and the calibration object are tracked. While
    both calibration object and laparoscope are tracked,
    so you could move things freely, we also developed a calibration
    rig. So, in general, the laparoscope is stationary on the rig and the calibration
    object is moved. This is mostly to reduce image blur and jitter
    caused by hand movements. See Dowrick  et al., 2023,
    https://doi.org/10.1002/mp.16310.

    :param rvecs: Array of rotation vectors, from OpenCV, camera extrinsics
    (pattern to camera transform)
    :type rvecs: List[np.ndarray]
    :param tvecs: Array of translation vectors, from OpenCV, camera extrinsics
    (pattern to camera transform)
    :type tvecs: List[np.ndarray]
    :param quat_model2hand_array: Array of quaternions representing rotational
    part of marker-to-hand.
    :type quat_model2hand_array: np.ndarray
    :param trans_model2hand_array: Array of quaternions representing
    translational part of marker-to-hand.
    :type trans_model2hand_array: np.ndarray
    :return: two 4x4 matrices as np.ndarray, representing handeye_matrix,
    pattern2marker_matrix
    :rtype: np.ndarray, np.ndarray
    """

    number_of_frames = len(rvecs)
    quat_extrinsics_array = np.zeros((number_of_frames, 4))

    for i in range(number_of_frames):
        quat_extrinsics_array[i] = qu.rvec_to_quat(rvecs[i].flatten())

    trans_extrinsics_array = np.reshape(tvecs, (number_of_frames, 3))

    q_handeye, t_handeye, q_pattern2marker, t_pattern2marker = \
        _gx_handeye_optimisation(quat_extrinsics_array, trans_extrinsics_array,
                                 quat_model2hand_array, trans_model2hand_array)

    handeye_matrix = np.eye(4)
    handeye_matrix[0:3, 0:3] = qu.quat_to_rotm(q_handeye)
    handeye_matrix[0:3, 3] = t_handeye

    pattern2marker_matrix = np.eye(4)
    pattern2marker_matrix[0:3, 0:3] = qu.quat_to_rotm(q_pattern2marker)
    pattern2marker_matrix[0:3, 3] = t_pattern2marker

    return handeye_matrix, pattern2marker_matrix


def calibrate_hand_eye_using_stationary_pattern(
        camera_rvecs: List[np.ndarray],
        camera_tvecs: List[np.ndarray],
        tracking_matrices: List[np.ndarray],
        method=cv2.CALIB_HAND_EYE_TSAI,
        invert_camera=False
        ):
    """
    Hand-eye calibration using standard OpenCV methods. This method assumes
    a single set of tracking data, so it is useful for a stationary, untracked
    calibration pattern, and a tracked video device, e.g. laparoscope.

    Please do read Ali et al., 2019, https://doi.org/10.3390/s19122837.

    OpenCV implements Tsai, Horaud and Park methods, which compute
    rotation then translation, as well as Andref and Daniilidis methods.
    See cv.calibrateHandEye and enum cv::HandEyeCalibrationMethod.
    Internally this method calls cv.calibrateHandEye.

    On average, in Ali's evaluation, Horaud might be a good default,
    but I've left it with Tsai's as the most well known (perhaps?).

    :param camera_rvecs: list of rvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param camera_tvecs: list of tvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param tracking_matrices: list of tracking matrices for the tracked
    device, marker_to_tracker.
    :param method: Choice of OpenCV Hand-Eye method.
    :param invert_camera: if True, we invert camera matrices before
    hand-eye calibration.
    :return hand-eye transform as 4x4 matrix.
    """
    if len(camera_rvecs) != len(camera_tvecs):
        raise ValueError("Camera rotation and translation vector "
                         "lists must be the same length.")

    if len(camera_tvecs) != len(tracking_matrices):
        raise ValueError("The number of camera extrinsic transforms must "
                         "equal the number of tracking matrices.")

    if len(camera_rvecs) < 3:
        raise ValueError("You must have at least 3 views, include movements "
                         "around at least 2 different rotation axes.")

    # Convert tracking matrices to rvecs/tvecs for OpenCV.
    tracking_rvecs = []
    tracking_tvecs = []
    for i in range(len(camera_rvecs)):
        rvecs, tvecs = vu.extrinsic_matrix_to_vecs(tracking_matrices[i])
        tracking_rvecs.append(rvecs)
        tracking_tvecs.append(tvecs)

    cam_rvecs = []
    cam_tvecs = []
    for i in range(len(camera_rvecs)):
        mat = vu.extrinsic_vecs_to_matrix(camera_rvecs[i], camera_tvecs[i])
        if invert_camera:
            mat = np.linalg.inv(mat)
        rvecs, tvecs = vu.extrinsic_matrix_to_vecs(mat)
        cam_rvecs.append(rvecs)
        cam_tvecs.append(tvecs)

    # OpenCV outputs eye-to-hand.
    e2h_rmat, e2h_tvec = cv2.calibrateHandEye(tracking_rvecs,
                                              tracking_tvecs,
                                              cam_rvecs,
                                              cam_tvecs,
                                              method=method
                                              )

    e2h = mu.construct_rigid_transformation(e2h_rmat, e2h_tvec)
    h2e = np.linalg.inv(e2h)

    return h2e


# pylint:disable=too-many-positional-arguments
def calibrate_hand_eye_using_tracked_pattern(
        camera_rvecs: List[np.ndarray],
        camera_tvecs: List[np.ndarray],
        device_tracking_matrices: List[np.ndarray],
        pattern_tracking_matrices: List[np.ndarray],
        method=cv2.CALIB_HAND_EYE_TSAI,
        invert_camera=False
        ):
    """
    See comments for calibrate_hand_eye_using_stationary_pattern.

    Here, we also have pattern_tracking_matrices. So, device_tracking_matrices
    and pattern_tracking_matrices are combined to give marker-to-hand
    transformation, and then calibrate_hand_eye_using_stationary_pattern called.

    You will notice, this method is NOT calculating pattern_to_marker. (i.e.
    the transformation between a calibration pattern, and the tracking marker
    attached to it). So, essentially, this method is simply treating the
    calibration pattern as the reference for the tracker space.
    This means you can hold the calibration pattern, and laparoscope,
    and move both of them around independently, but the maths is solved
    by assuming the calibration pattern is stationary, and the laparoscope
    is moving around the calibration pattern.
    """
    if len(device_tracking_matrices) != len(pattern_tracking_matrices):
        raise ValueError("The number of device tracking matrices must "
                         "equal the number of pattern tracking matrices.")

    if len(device_tracking_matrices) < 3:
        raise ValueError("You must have at least 3 tracking matrices")

    for i in range(len(pattern_tracking_matrices)):
        if pattern_tracking_matrices[i] is None:
            raise ValueError("Pattern tracking matrix:"
                             + str(i) + str(" is None."))

    # Essentially, as we have a tracked calibration object,
    # the calibration object becomes the reference for the tracker space.
    tracking_matrices = []
    for i in range(len(device_tracking_matrices)):
        mat = np.linalg.inv(pattern_tracking_matrices[i]) \
              @ device_tracking_matrices[i]
        tracking_matrices.append(mat)

    h2e = calibrate_hand_eye_using_stationary_pattern(camera_rvecs,
                                                      camera_tvecs,
                                                      tracking_matrices,
                                                      method,
                                                      invert_camera)
    return h2e


def calibrate_pattern_to_tracking_marker(camera_rvecs: List[np.ndarray],
                                         camera_tvecs: List[np.ndarray],
                                         tracking_matrices: List[np.ndarray],
                                         method=cv2.CALIB_HAND_EYE_TSAI
                                         ):
    """
    This is really a convenience method that is analagous to
    calibrate_hand_eye_using_stationary_pattern. i.e. you have a stationary
    camera but a moving calibration pattern.

    So, here, you are JUST calculating pattern_to_marker transformation.

    :param camera_rvecs: list of rvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param camera_tvecs: list of tvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param tracking_matrices: list of pattern tracking matrices for the tracked
    calibration pattern, marker_to_tracker.
    :param method: Choice of OpenCV Hand-Eye method.
    :return pattern_to_marker
    """
    h2e = calibrate_hand_eye_using_stationary_pattern(camera_rvecs,
                                                      camera_tvecs,
                                                      tracking_matrices,
                                                      method,
                                                      invert_camera=True)

    p2m = np.linalg.inv(h2e)

    return p2m


def calibrate_hand_eye_and_pattern_to_marker(
        camera_rvecs: List[np.ndarray],
        camera_tvecs: List[np.ndarray],
        device_tracking_matrices: List[np.ndarray],
        pattern_tracking_matrices: List[np.ndarray],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        ):
    """
    Hand-eye calibration using standard OpenCV methods. This method assumes
    you are tracking both the device that needs hand-eye calibration, and
    the calibration pattern, and internally calls cv.calibrateRobotWorldHandEye.

    Again, please do read Ali et al., 2019, https://doi.org/10.3390/s19122837.

    OpenCV already implements Shah and Li's methods. See cv.calibrateRobotWorldHandEye.

    :param camera_rvecs: list of rvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param camera_tvecs: list of tvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param device_tracking_matrices: list of tracking matrices for the
    tracked device, e.g. laparoscope, marker_to_tracker.
    :param pattern_tracking_matrices: list of tracking matrices for the
    calibration object, marker_to_tracker
    :param method: Choice of OpenCV RobotWorldHandEye method.
    :return hand-eye, pattern-to-marker transforms as 4x4 matrices
    """
    if len(camera_rvecs) != len(camera_tvecs):
        raise ValueError("Camera rotation and translation vector lists "
                         "must be the same length.")

    if len(camera_tvecs) != len(device_tracking_matrices):
        raise ValueError("The number of camera extrinsic transforms must "
                         "equal the number of device tracking matrices.")

    if len(camera_tvecs) != len(pattern_tracking_matrices):
        raise ValueError("The number of camera extrinsic transforms must "
                         "equal the number of pattern tracking matrices.")

    if len(camera_rvecs) < 3:
        raise ValueError("You must have at least 3 views, include movements "
                         "around at least 2 different rotation axes.")

    # Convert tracking matrices to rvecs/tvecs for OpenCV.
    # Look at OpenCV documentation for cv.calibrationRobotWorldHandEye
    # Gripper = tracking marker on calibration pattern
    # Robot base = tracking marker on video device, eg. laparoscope
    # World = video camera on laparoscope, assumed to be stationary
    # So, we move the calibration pattern, not the video device.

    base2gripper_rvecs = []
    base2gripper_tvecs = []
    for i in range(len(camera_rvecs)):
        b2g = np.linalg.inv(pattern_tracking_matrices[i]) \
              @ device_tracking_matrices[i]
        rvecs, tvecs = vu.extrinsic_matrix_to_vecs(b2g)
        base2gripper_rvecs.append(rvecs)
        base2gripper_tvecs.append(tvecs)

    cam_rvecs = []
    cam_tvecs = []
    for i in range(len(camera_rvecs)):
        mat = vu.extrinsic_vecs_to_matrix(camera_rvecs[i], camera_tvecs[i])
        mat = np.linalg.inv(mat)
        rvecs, tvecs = vu.extrinsic_matrix_to_vecs(mat)
        cam_rvecs.append(rvecs)
        cam_tvecs.append(tvecs)

    b2w_rmat, b2w_t, g2c_rmat, g2c_t = \
        cv2.calibrateRobotWorldHandEye(cam_rvecs,
                                       cam_tvecs,
                                       base2gripper_rvecs,
                                       base2gripper_tvecs,
                                       method=method)

    h2e = mu.construct_rigid_transformation(b2w_rmat, b2w_t)
    m2p = mu.construct_rigid_transformation(g2c_rmat, g2c_t)
    p2m = np.linalg.inv(m2p)

    return h2e, p2m


def calibrate_hand_eye_and_grid_to_world(
        camera_rvecs: List[np.ndarray],
        camera_tvecs: List[np.ndarray],
        device_tracking_matrices: List[np.ndarray],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
        ):
    """
    Similar to calibrate_hand_eye_and_pattern_to_marker, except
    here, the calibration pattern is untracked, and assumed to be
    stationary.

    :param camera_rvecs: list of rvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param camera_tvecs: list of tvecs that we get from OpenCV camera
    extrinsics, pattern_to_camera.
    :param device_tracking_matrices: list of tracking matrices for the
    tracked device, e.g. laparoscope, marker_to_tracker.
    :param method: Choice of OpenCV RobotWorldHandEye method.
    :return hand-eye, grid-to-world transforms as 4x4 matrices
    """
    if len(camera_rvecs) != len(camera_tvecs):
        raise ValueError("Camera rotation and translation vector "
                         "lists must be the same length.")

    if len(camera_tvecs) != len(device_tracking_matrices):
        raise ValueError("The number of camera extrinsic transforms "
                         "must equal the number of device tracking matrices.")

    if len(camera_rvecs) < 3:
        raise ValueError("You must have at least 3 views, include "
                         "movements around at least 2 different rotation axes.")

    base2gripper_rvecs = []
    base2gripper_tvecs = []
    for i in range(len(device_tracking_matrices)):
        b2g = np.linalg.inv(device_tracking_matrices[i])
        rvec, tvec = vu.extrinsic_matrix_to_vecs(b2g)
        base2gripper_rvecs.append(rvec)
        base2gripper_tvecs.append(tvec)

    b2w_rmat, b2w_t, g2c_rmat, g2c_t = \
        cv2.calibrateRobotWorldHandEye(camera_rvecs,
                                       camera_tvecs,
                                       base2gripper_rvecs,
                                       base2gripper_tvecs,
                                       method=method)

    h2e = mu.construct_rigid_transformation(g2c_rmat, g2c_t)
    b2w = mu.construct_rigid_transformation(b2w_rmat, b2w_t)
    w2b = np.linalg.inv(b2w)

    return h2e, w2b
