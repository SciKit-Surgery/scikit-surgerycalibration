# coding=utf-8

""" Functions to run pivot calibration. """

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
    configuration = None
    if config_file is not None:
        configurer = ConfigurationManager(config_file)
        configuration = configurer.get_copy()

    input_files = input_directory + '/*'
    file_names = glob(input_files)
    arrays = [np.loadtxt(f) for f in file_names]

    matrices = np.concatenate(arrays)

    number_of_4x4_matrices = int(matrices.size / 16)

    matrices_4x4 = matrices.reshape((number_of_4x4_matrices, 4, 4))

    pointer_offset, pivot_location, residual_error = \
            p.pivot_calibration(matrices_4x4, configuration)

    print("Pointer Offset = ", pointer_offset.reshape((1, 3)))
    print("Pivot Location = ", pivot_location.reshape((1, 3)))
    print("Residual Error = ", residual_error)
