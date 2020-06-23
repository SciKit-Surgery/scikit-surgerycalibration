# coding=utf-8

""" Command line processing for pivot calibration. """

import argparse
from sksurgerycalibration import __version__
from sksurgerycalibration.ui.pivot_calibration_app import run_pivot_calibration


def main(args=None):
    """Entry point for pivot calibration application. """

    parser = argparse.ArgumentParser(description='pivotcalibration')

    parser.add_argument("-i", "--input",
                        required=True,
                        type=str,
                        help="A directory containing tracking matrix files.")

    parser.add_argument("-c", "--config",
                        required=False,
                        type=str,
                        help="Configuration file containing the parameters")

    version_string = __version__
    friendly_version_string = version_string if version_string else 'unknown'
    parser.add_argument(
        "-v", "--version",
        action='version',
        version='sksurgerycalibration version ' + friendly_version_string)

    args = parser.parse_args(args)

    run_pivot_calibration(args.input, args.config)
