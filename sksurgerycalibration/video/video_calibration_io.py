# -*- coding: utf-8 -*-

""" Various functions to help with IO. Not intended for 3rd party clients. """

import os
import glob

# pylint: disable=missing-function-docstring, invalid-name


def get_calib_prefix(file_prefix: str):

    prefix = 'calib'
    if file_prefix:
        prefix = file_prefix
    return prefix


def get_left_prefix(file_prefix: str):

    left_prefix = "calib.left"
    if file_prefix:
        left_prefix = file_prefix + ".left"
    return left_prefix


def get_right_prefix(file_prefix: str):

    right_prefix = "calib.right"
    if file_prefix:
        right_prefix = file_prefix + ".right"
    return right_prefix


def get_intrinsics_file_name(dir_name: str,
                             file_prefix: str):

    intrinsics_file = os.path.join(dir_name,
                                   get_calib_prefix(file_prefix) +
                                   ".intrinsics.txt")
    return intrinsics_file


def get_distortion_file_name(dir_name: str,
                             file_prefix: str):

    dist_coeff_file = os.path.join(dir_name,
                                   get_calib_prefix(file_prefix) +
                                   ".distortion.txt")
    return dist_coeff_file


def get_enumerated_file_name(dir_name: str,
                             file_prefix: str,
                             type_prefix: str,
                             view_number: str,
                             extension_wth_dot: str
                             ):

    # Keep in synch with _get_enumerated_file_glob
    file_name = \
        os.path.join(dir_name,
                     get_calib_prefix(file_prefix)
                     + "."
                     + type_prefix
                     + "."
                     + str(view_number) + extension_wth_dot)
    return file_name


def get_enumerated_file_glob(dir_name: str,
                             file_prefix: str,
                             type_prefix: str,
                             extension_wth_dot: str
                             ):

    # Keep in synch with _get_enumerated_file_name
    file_glob = \
        os.path.join(dir_name,
                     get_calib_prefix(file_prefix)
                     + "."
                     + type_prefix
                     + "."
                     + "*" + extension_wth_dot)

    return file_glob


def get_extrinsics_file_name(dir_name: str,
                             file_prefix: str,
                             view_number: int
                             ):

    extrinsics_file = get_enumerated_file_name(dir_name,
                                               file_prefix,
                                               "extrinsics",
                                               view_number,
                                               ".txt")
    return extrinsics_file


def get_extrinsic_file_names(dir_name: str,
                             file_prefix: str):

    files = get_filenames_by_glob_expr(dir_name,
                                       file_prefix,
                                       "extrinsics",
                                       ".txt")
    return files


def get_l2r_file_name(dir_name: str,
                      file_prefix: str):
    l2r_file = os.path.join(dir_name,
                            get_calib_prefix(file_prefix) + ".l2r.txt")
    return l2r_file


def get_essential_matrix_file_name(dir_name: str,
                                   file_prefix: str):
    ess_file = os.path.join(dir_name,
                            get_calib_prefix(file_prefix) + ".essential.txt")
    return ess_file


def get_fundamental_matrix_file_name(dir_name: str,
                                     file_prefix: str):
    fun_file = os.path.join(dir_name,
                            get_calib_prefix(file_prefix) + ".fundamental.txt")
    return fun_file


def get_images_file_name(dir_name: str,
                         file_prefix: str,
                         view_number: int
                         ):
    images_file = get_enumerated_file_name(dir_name,
                                           file_prefix,
                                           "images",
                                           view_number,
                                           ".png")
    return images_file


def get_ids_file_name(dir_name: str,
                      file_prefix: str,
                      view_number: int
                      ):
    ids_file = get_enumerated_file_name(dir_name,
                                        file_prefix,
                                        "ids",
                                        view_number,
                                        ".txt")
    return ids_file


def get_objectpoints_file_name(dir_name: str,
                               file_prefix: str,
                               view_number: int
                               ):
    object_points_file = get_enumerated_file_name(dir_name,
                                                  file_prefix,
                                                  "object_points",
                                                  view_number,
                                                  ".txt")
    return object_points_file


def get_imagepoints_file_name(dir_name: str,
                              file_prefix: str,
                              view_number: int
                              ):
    image_points_file = get_enumerated_file_name(dir_name,
                                                 file_prefix,
                                                 "image_points",
                                                 view_number,
                                                 ".txt")
    return image_points_file


def get_device_tracking_file_name(dir_name: str,
                                  file_prefix: str,
                                  view_number: int
                                  ):
    device_tracking = get_enumerated_file_name(dir_name,
                                               file_prefix,
                                               "device_tracking",
                                               view_number,
                                               ".txt")
    return device_tracking


def get_calibration_tracking_file_name(dir_name: str,
                                       file_prefix: str,
                                       view_number: int
                                       ):
    calibration_tracking = get_enumerated_file_name(dir_name,
                                                    file_prefix,
                                                    "calib_obj_tracking",
                                                    view_number,
                                                    ".txt")
    return calibration_tracking


def get_filenames_by_glob_expr(dir_name: str,
                               file_prefix: str,
                               type_prefix: str,
                               extension_with_dot: str
                               ):

    file_glob = get_enumerated_file_glob(dir_name,
                                         file_prefix,
                                         type_prefix,
                                         extension_with_dot)
    files = glob.glob(file_glob)
    files.sort()
    return files
