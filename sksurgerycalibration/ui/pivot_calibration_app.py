# coding=utf-8

"""Basic Augmented Reality Demo BARD Application"""

from glob import glob
import numpy as np
from sksurgerycore.configuration.configuration_manager import \
        ConfigurationManager
from sksurgerycalibration.algorithms import pivot as p


def run_pivot_calibration(input_directory, config_file):
    """
    Performs Pivot Calibration using matrices stored in
    separate file input_dir
    """
    configuration_data = None
    if config_file is not None:
        configurer = ConfigurationManager(config_file)
        configuration_data = configurer.get_copy()

    input_files = input_directory + '/*'
    file_names = glob(input_files)
    arrays = [np.loadtxt(f) for f in file_names]

    matrices = np.concatenate(arrays)

    number_of_4x4_matrices = int(matrices.size / 16)

    matrices_4x4 = matrices.reshape((number_of_4x4_matrices, 4, 4))

    use_algebraic_one_step = True
    use_ransac = False
    pointer_offset = None
    pivot_location = None
    residual_error = None
    if configuration_data is not None:
        if configuration_data.get('ransac', False):
            use_ransac = True
            use_algebraic_one_step = False
    if use_algebraic_one_step:
        pointer_offset, pivot_location, residual_error = \
            p.pivot_calibration(matrices_4x4)

    if use_ransac:
        number_iterations = configuration_data.get('number_iterations', 10)
        error_threshold = configuration_data.get('error_threshold', 4)
        consensus_threshold = configuration_data.get('consensus_threshold',
                                                     0.25)
        early_exit = configuration_data.get('early_exit', False)
        pointer_offset, pivot_location, residual_error = \
            p.pivot_calibration_with_ransac(
                matrices_4x4, number_iterations, error_threshold,
                consensus_threshold, early_exit)

    print("Pointer Offset = ", pointer_offset.reshape((1, 3)))
    print("Pivot Location = ", pivot_location.reshape((1, 3)))
    print("Residual Error = ", residual_error)
