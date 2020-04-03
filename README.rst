scikit-surgery-calibration
===============================

.. image:: https://github.com/UCL/scikit-surgery-calibration /raw/master/project-icon.png
   :height: 128px
   :width: 128px
   :target: https://github.com/UCL/scikit-surgery-calibration 
   :alt: Logo

.. image:: https://github.com/UCL/scikit-surgery-calibration /badges/master/build.svg
   :target: https://github.com/UCL/scikit-surgery-calibration /pipelines
   :alt: GitLab-CI test status

.. image:: https://github.com/UCL/scikit-surgery-calibration /badges/master/coverage.svg
    :target: https://github.com/UCL/scikit-surgery-calibration /commits/master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgery-calibration /badge/?version=latest
    :target: http://scikit-surgery-calibration .readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Stephen Thompson

scikit-surgery-calibration is part of the `SNAPPY`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgery-calibration supports Python 2.7 and Python 3.6.

scikit-surgery-calibration is currently a demo project, which will add/multiply two numbers. Example usage:

::

    python sksurgerycalibration.py 5 8
    python sksurgerycalibration.py 3 6 --multiply

Please explore the project structure, and implement your own functionality.

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/scikit-surgery-calibration 


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    pip install pytest
    python -m pytest


Linting
^^^^^^^

This code conforms to the PEP8 standard. Pylint can be used to analyse the code:

::

    pip install pylint
    pylint --rcfile=tests/pylintrc sksurgerycalibration


Installing
----------

You can pip install directly from the repository as follows:

::

    pip install git+https://github.com/UCL/scikit-surgery-calibration 



Contributing
^^^^^^^^^^^^

Please see the `contributing guidelines`_.


Useful links
^^^^^^^^^^^^

* `Source code repository`_
* `Documentation`_


Licensing and copyright
-----------------------

Copyright 2020 University College London.
scikit-surgery-calibration is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/UCL/scikit-surgery-calibration 
.. _`Documentation`: https://scikit-surgery-calibration .readthedocs.io
.. _`SNAPPY`: https://weisslab.cs.ucl.ac.uk/WEISS/PlatformManagement/SNAPPY/wikis/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/scikit-surgery-calibration /blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/scikit-surgery-calibration /blob/master/LICENSE

