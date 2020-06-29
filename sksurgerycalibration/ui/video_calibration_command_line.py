# coding=utf-8

""" Command line processing for video calibration app. """

import argparse
from sksurgerycalibration import __version__
from sksurgerycalibration.ui.video_calibration_app import run_video_calibration


def main(args=None):
    """Entry point for simple video calibration application. """

    parser = argparse.ArgumentParser(description='videocalibration')

    parser.add_argument("-c", "--config",
                        required=True,
                        type=str,
                        help="Configuration file containing the parameters "
                             "(see config/video_chessboard_conf.json "
                             "for example).")

    parser.add_argument("-s", "--save",
                        required=False,
                        type=str,
                        help="Directory to save to.")

    parser.add_argument("-p", "--prefix",
                        required=False,
                        type=str,
                        help="Filename prefix to save to.")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "-v", "--version",
        action='version',
        version='sksurgerycalibration version ' + friendly_version_string)

    args = parser.parse_args(args)

    run_video_calibration(args.config, args.save, args.prefix)
