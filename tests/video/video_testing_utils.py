# -*- coding: utf-8 -*-

""" Various utilities to help testing. """

import os
import glob
import cv2


def load_left_right_pngs(dir_name, expected_number):
    """Given a dir_name, loads left/*.png and right/*.png"""
    left_images = []
    files = glob.glob(os.path.join(dir_name, "left", "*.png"))
    files.sort()
    for file in files:
        image = cv2.imread(file)
        left_images.append(image)
    assert len(left_images) == expected_number

    right_images = []
    files = glob.glob(os.path.join(dir_name, "right", "*.png"))
    files.sort()
    for file in files:
        image = cv2.imread(file)
        right_images.append(image)
    assert len(right_images) == expected_number

    return left_images, right_images
