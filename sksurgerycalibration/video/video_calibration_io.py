# -*- coding: utf-8 -*-

""" Various functions to help with IO. """

import os
import glob
from fnmatch import filter as file_filter


def _get_intrinsics_file_name(dir_name: str,
                              file_prefix: str):
    intrinsics_file = os.path.join(dir_name,
                                   file_prefix + ".intrinsics.txt")
    return intrinsics_file


def _get_distortion_file_name(dir_name: str,
                              file_prefix: str):
    dist_coeff_file = os.path.join(dir_name,
                                   file_prefix + ".distortion.txt")
    return dist_coeff_file


def _get_enumerated_file_name(dir_name: str,
                              file_prefix: str,
                              type_prefix: str,
                              view_number: str
                              ):
    # Keep in synch with _get_enumerated_file_glob
    file_name = \
        os.path.join(dir_name,
                     file_prefix
                     + "."
                     + type_prefix
                     + "."
                     + str(view_number) + ".txt")
    return file_name


def _get_enumerated_file_glob(dir_name: str,
                              file_prefix: str,
                              type_prefix: str
                              ):
    # Keep in synch with _get_enumerated_file_name
    file_glob = \
        os.path.join(dir_name,
                     file_prefix
                     + "."
                     + type_prefix
                     + "."
                     + "*" + ".txt")
    return file_glob


def _get_extrinsics_file_name(dir_name: str,
                              file_prefix: str,
                              view_number: int
                              ):
    extrinsics_file = _get_enumerated_file_name(dir_name,
                                                file_prefix,
                                                "extrinsics",
                                                view_number)
    return extrinsics_file


def _get_extrinsic_file_names(dir_name: str,
                              file_prefix: str):
    files = file_filter(os.listdir(dir_name),
                        file_prefix + ".extrinsics.*.txt")
    return files


def _get_left_prefix(file_prefix: str):
    left_prefix = "left"
    if file_prefix:
        left_prefix = file_prefix + "." + left_prefix
    return left_prefix


def _get_right_prefix(file_prefix: str):
    right_prefix = "right"
    if file_prefix:
        right_prefix = file_prefix + "." + right_prefix
    return right_prefix


def _get_l2r_file_name(dir_name: str,
                       file_prefix: str):
    l2r_file = os.path.join(dir_name,
                            file_prefix + ".l2r.txt")
    return l2r_file


def _get_images_file_name(dir_name: str,
                          file_prefix: str,
                          view_number: int
                          ):
    images_file = _get_enumerated_file_name(dir_name,
                                            file_prefix,
                                            "images",
                                            view_number)
    return images_file


def _get_ids_file_name(dir_name: str,
                       file_prefix: str,
                       view_number: int
                       ):
    ids_file = _get_enumerated_file_name(dir_name,
                                         file_prefix,
                                         "ids",
                                         view_number)
    return ids_file


def _get_objectpoints_file_name(dir_name: str,
                                file_prefix: str,
                                view_number: int
                                ):
    object_points_file = _get_enumerated_file_name(dir_name,
                                                   file_prefix,
                                                   "objectpoints",
                                                   view_number)
    return object_points_file


def _get_imagepoints_file_name(dir_name: str,
                               file_prefix: str,
                               view_number: int
                               ):
    image_points_file = _get_enumerated_file_name(dir_name,
                                                  file_prefix,
                                                  "imagepoints",
                                                  view_number)
    return image_points_file


def _get_device_tracking_file_name(dir_name: str,
                                   file_prefix: str,
                                   view_number: int
                                   ):
    device_tracking = _get_enumerated_file_name(dir_name,
                                                file_prefix,
                                                "device",
                                                view_number)
    return device_tracking


def _get_calibration_tracking_file_name(dir_name: str,
                                        file_prefix: str,
                                        view_number: int
                                        ):
    calibration_tracking = _get_enumerated_file_name(dir_name,
                                                     file_prefix,
                                                     "calibration",
                                                     view_number)
    return calibration_tracking


def _get_filenames_by_glob_expr(dir_name: str,
                                file_prefix: str,
                                type_prefix: str
                                ):

    file_glob = _get_enumerated_file_glob(dir_name,
                                          file_prefix,
                                          type_prefix)
    files = glob.glob(file_glob)
    files.sort()
    return files

