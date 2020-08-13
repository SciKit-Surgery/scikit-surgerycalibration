scikit-surgerycalibration
===============================

.. image:: https://github.com/UCL/scikit-surgerycalibration /raw/master/weiss_logo.png
   :height: 128px
   :width: 128px
   :target: https://github.com/UCL/scikit-surgerycalibration 
   :alt: Logo

|

.. image:: https://github.com/UCL/scikit-surgerycalibration/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/UCL/scikit-surgerycalibration/actions
   :alt: GitHub Actions CI statuss

.. image:: https://coveralls.io/repos/github/UCL/scikit-surgerycalibration/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/UCL/scikit-surgerycalibration?branch=master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgerycalibration /badge/?version=latest
    :target: http://scikit-surgerycalibration .readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status



Author: Stephen Thompson

scikit-surgerycalibration is part of the `SciKit-Surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgerycalibration is tested on Python 3.6-8.

scikit-surgerycalibration contains algorithms to perform calibrations useful during surgery, for example pointer calibration, ultrasound calibration, and camera calibration. 

Please explore the project structure, and request or implement your desired functionality.

.. features-start

Features
--------

* `Pivot Calibration <https://scikit-surgerycalibration.readthedocs.io/en/latest/module_ref.html#pivot-calibration>`_ for pivot calibration.
* `Calibration <https://scikit-surgerycalibration.readthedocs.io/en/latest/module_ref.html#video-calibration>`_ of mono or stereo tracked video data, calculating camera intrinsics and handeye transformation.

.. features-end

Developing
----------

Cloning
^^^^^^^

You can clone the repository using the following command:

::

    git clone https://github.com/UCL/scikit-surgerycalibration 


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

    pip install git+https://github.com/UCL/scikit-surgerycalibration 



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
scikit-surgerycalibration is released under the BSD-3 license. Please see the `license file`_ for details.


Acknowledgements
----------------

Supported by `Wellcome`_ and `EPSRC`_.


.. _`Wellcome EPSRC Centre for Interventional and Surgical Sciences`: http://www.ucl.ac.uk/weiss
.. _`source code repository`: https://github.com/UCL/scikit-surgerycalibration 
.. _`Documentation`: https://scikit-surgerycalibration.readthedocs.io
.. _`SciKit-Surgery`: https://github.com/UCL/scikit-surgery/wiki/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/UCL/scikit-surgerycalibration /blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/UCL/scikit-surgerycalibration /blob/master/LICENSE

