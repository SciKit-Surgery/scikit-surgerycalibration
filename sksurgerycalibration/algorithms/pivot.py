#  -*- coding: utf-8 -*-

""" Functions for pivot calibration. """

import random
import numpy as np

# pylint: disable=literal-comparison


def pivot_calibration(tracking_matrices):

    """
    Performs Pivot Calibration, using Algebraic One Step method,
    and returns Residual Error.

    See `Yaniv 2015 <https://dx.doi.org/10.1117/12.2081348>`_.

    :param tracking_matrices: N x 4 x 4 ndarray, of tracking matrices.
    :returns: pointer offset, pivot point and RMS Error about centroid of pivot.
    :raises: TypeError, ValueError
    """
    if not isinstance(tracking_matrices, np.ndarray):
        raise TypeError("tracking_matrices is not a numpy array'")

    if not tracking_matrices.shape[1] == 4:
        raise ValueError("tracking_matrices should have 4 rows per matrix")

    if not tracking_matrices.shape[2] == 4:
        raise ValueError("tracking_matrices should have 4 columns per matrix")

    number_of_matrices = tracking_matrices.shape[0]

    # See equation in section 2.1.2 of Yaniv 2015.
    # Ax = b.

    size_a = 3 * number_of_matrices, 6
    a_values = np.zeros(size_a, dtype=np.float64)

    size_b = 3 * number_of_matrices, 1
    b_values = np.zeros(size_b, dtype=np.float64)

    for i in range(number_of_matrices):

        # Column vector containing -1 * translation from each tracking matrix.
        b_values[i * 3 + 0, 0] = -1 * tracking_matrices[i, 0, 3]
        b_values[i * 3 + 1, 0] = -1 * tracking_matrices[i, 1, 3]
        b_values[i * 3 + 2, 0] = -1 * tracking_matrices[i, 2, 3]

        # A contains rotation matrix from each tracking matrix.
        a_values[i * 3 + 0, 0] = tracking_matrices[i, 0, 0]
        a_values[i * 3 + 1, 0] = tracking_matrices[i, 1, 0]
        a_values[i * 3 + 2, 0] = tracking_matrices[i, 2, 0]
        a_values[i * 3 + 0, 1] = tracking_matrices[i, 0, 1]
        a_values[i * 3 + 1, 1] = tracking_matrices[i, 1, 1]
        a_values[i * 3 + 2, 1] = tracking_matrices[i, 2, 1]
        a_values[i * 3 + 0, 2] = tracking_matrices[i, 0, 2]
        a_values[i * 3 + 1, 2] = tracking_matrices[i, 1, 2]
        a_values[i * 3 + 2, 2] = tracking_matrices[i, 2, 2]

        # A contains -I for each tracking matrix.
        a_values[i * 3 + 0, 3] = -1
        a_values[i * 3 + 1, 3] = 0
        a_values[i * 3 + 2, 3] = 0
        a_values[i * 3 + 0, 4] = 0
        a_values[i * 3 + 1, 4] = -1
        a_values[i * 3 + 2, 4] = 0
        a_values[i * 3 + 0, 5] = 0
        a_values[i * 3 + 1, 5] = 0
        a_values[i * 3 + 2, 5] = -1

    # To calculate Singular Value Decomposition

    u_values, s_values, v_values = np.linalg.svd(a_values, full_matrices=False)
    c_values = np.dot(u_values.T, b_values)
    w_values = np.dot(np.diag(1 / s_values), c_values)
    x_values = np.dot(v_values.T, w_values)

    # Calculating the rank, and removing close to zero singular values.
    rank = 0
    for i, item in enumerate(s_values):
        if item < 0.01:
            item = 0
        if item != 0:
            rank += 1

    if rank < 6:
        raise ValueError("PivotCalibration: Failed. Rank < 6")

    # Compute RMS error.
    residual_matrix = (np.dot(a_values, x_values) - b_values)
    residual_error = np.sum(residual_matrix * residual_matrix)
    residual_error = residual_error / float(number_of_matrices * 3)
    residual_error = np.sqrt(residual_error)

    return x_values, residual_error


def pivot_calibration_with_ransac(tracking_matrices,
                                  number_iterations,
                                  error_threshold,
                                  concensus_threshold,
                                  early_exit=False
                                  ):
    """
    Written as an exercise for implementing RANSAC.

    :param tracking_matrices: N x 4 x 4 ndarray, of tracking matrices.
    :param number_iterations: the number of iterations to attempt.
    :param error_threshold: distance in millimetres from pointer position
    :param concensus_threshold: the minimum percentage of inliers to finish
    :param early_exit: If True, returns model as soon as thresholds are met
    :returns: pointer offset, pivot point and RMS Error about centroid of pivot.
    :raises: TypeError, ValueError
    """
    if number_iterations < 1:
        raise ValueError("The number of iterations must be > 1")
    if error_threshold < 0:
        raise ValueError("The error threshold must be a positive distance.")
    if concensus_threshold < 0 or concensus_threshold > 1:
        raise ValueError("The concensus threshold must be [0-1] as percentage")
    if not isinstance(tracking_matrices, np.ndarray):
        raise TypeError("tracking_matrices is not a numpy array'")

    number_of_matrices = tracking_matrices.shape[0]
    population_of_indices = range(number_of_matrices)
    minimum_matrices_required = 3

    highest_number_of_inliers = -1
    best_model = None
    best_rms = -1

    for iter_counter in range(number_iterations):
        indexes = random.sample(population_of_indices,
                                minimum_matrices_required)
        sample = tracking_matrices[indexes]

        try:
            model, _ = pivot_calibration(sample)
        except ValueError:
            print("RANSAC, iteration " + str(iter_counter) + ", failed.")
            continue

        # Need to evaluate the number of inliers.
        # Slow, but it's written as a teaching exercise.
        world_point = model[3:6]
        number_of_inliers = 0
        inlier_indices = []
        for matrix_counter in range(number_of_matrices):
            offset = np.vstack((model[0:3], 1))
            transformed_point = tracking_matrices[matrix_counter] @ offset
            diff = world_point - transformed_point[0:3]
            norm = np.linalg.norm(diff)
            if norm < error_threshold:
                number_of_inliers = number_of_inliers + 1
                inlier_indices.append(matrix_counter)

        percentage_inliers = number_of_inliers / number_of_matrices

        # Keep the best model so far, based on the highest number of inliers.
        if percentage_inliers > concensus_threshold \
                and number_of_inliers > highest_number_of_inliers:
            highest_number_of_inliers = number_of_inliers
            inlier_matrices = tracking_matrices[inlier_indices]
            best_model, best_rms = pivot_calibration(inlier_matrices)

        # Early exit condition, as soon as we find model with enough fit.
        if percentage_inliers > concensus_threshold and early_exit:
            return best_model, best_rms

    if best_model is None:
        raise ValueError("Failed to find a model using RANSAC.")

    print("RANSAC Pivot, from " + str(number_of_matrices)
          + " matrices, used " + str(highest_number_of_inliers)
          + " matrices, with error threshold = " + str(error_threshold)
          + " and consensus threshold = " + str(concensus_threshold)
          )

    return best_model, best_rms
