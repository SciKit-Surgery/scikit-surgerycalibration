#!/usr/bin/python
#  -*- coding: utf-8 -*-
import sys

from sksurgerycalibration.ui.pivot_calibration_command_line import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
