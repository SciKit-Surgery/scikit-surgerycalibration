# -*- coding: utf-8 -*-

import os
from typing import Tuple
import numpy as np
import cv2


def get_calib_data(directory: str, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Generate filenames for calibration data in a given directory. """
    left_image = cv2.imread(
        os.path.join(directory, f'calib.left.images.{idx}.png')
    )

    right_image = cv2.imread(
        os.path.join(directory, f'calib.right.images.{idx}.png')
    )

    chessboard_tracking = np.loadtxt(
        os.path.join(directory, f'calib.calib_obj_tracking.{idx}.txt')
    )

    scope_tracking = np.loadtxt(
        os.path.join(directory, f'calib.device_tracking.{idx}.txt')
    )

    return left_image, right_image, chessboard_tracking, scope_tracking
