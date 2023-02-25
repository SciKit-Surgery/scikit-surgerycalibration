scikit-surgerycalibration
===============================

.. image:: https://github.com/SciKit-Surgery/scikit-surgerycalibration /raw/master/weiss_logo.png
   :height: 128px
   :width: 128px
   :target: https://github.com/SciKit-Surgery/scikit-surgerycalibration 
   :alt: Logo

|

.. image:: https://github.com/SciKit-Surgery/scikit-surgerycalibration/workflows/.github/workflows/ci.yml/badge.svg
   :target: https://github.com/SciKit-Surgery/scikit-surgerycalibration/actions
   :alt: GitHub Actions CI statuss

.. image:: https://coveralls.io/repos/github/SciKit-Surgery/scikit-surgerycalibration/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/SciKit-Surgery/scikit-surgerycalibration?branch=master
    :alt: Test coverage

.. image:: https://readthedocs.org/projects/scikit-surgerycalibration /badge/?version=latest
    :target: http://scikit-surgerycalibration .readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/Cite-SciKit--Surgery-informational
   :target: https://doi.org/10.1007/s11548-020-02180-5
   :alt: The SciKit-Surgery paper

.. image:: https://img.shields.io/twitter/follow/scikit_surgery?style=social
   :target: https://twitter.com/scikit_surgery?ref_src=twsrc%5Etfw
   :alt: Follow scikit_surgery on twitter


Author(s): Stephen Thompson; Contributor(s): Matt Clarkson, Thomas Dowrick and Miguel Xochicale

scikit-surgerycalibration is part of the `SciKit-Surgery`_ software project, developed at the `Wellcome EPSRC Centre for Interventional and Surgical Sciences`_, part of `University College London (UCL)`_.

scikit-surgerycalibration is tested on Python 3.7.

scikit-surgerycalibration contains algorithms to perform calibrations useful during surgery, for example pointer calibration, ultrasound calibration, and camera calibration. 

Please explore the project structure, and request or implement your desired functionality.

.. features-start

Features
--------

* `Pivot Calibration <https://scikit-surgerycalibration.readthedocs.io/en/latest/module_ref.html#pivot-calibration>`_ for pivot calibration.
* `Calibration <https://scikit-surgerycalibration.readthedocs.io/en/latest/module_ref.html#video-calibration>`_ of mono or stereo tracked video data, calculating camera intrinsics and handeye transformation.

.. features-end


Cloning
-------

You can clone the repository using the following command:
::

    git clone https://github.com/SciKit-Surgery/scikit-surgerycalibration
    git clone git@github.com:SciKit-Surgery/scikit-surgerycalibration.git # Alternatively, use password-protected SSH key.


Developing
----------

We recommend using `anaconda`_ or `miniconda`_ to create a python 3.7 environment,
then using `tox`_ to install all dependencies inside a dedicated `venv`_. We then use
github `actions`_ to run a matrix of builds for Windows, Linux and Mac and various python versions.

All library dependencies are specified via ``requirements-dev.txt`` which refers to ``requirements.txt``.

So, assuming either `anaconda`_ or `miniconda`_ is installed, and your current working directory is the root directory of this project:
::

    conda create --name scikit-surgery python=3.7
    conda activate scikit-surgery
    pip install tox
    tox

As the `tox`_ command runs, it will install all dependencies in a sub-directory ``.tox/py37`` (Linux/Mac) or ``.tox\py37`` (Windows).
`tox`_ will also run pytest and linting for you.

To run commands inside the same environment as `tox`_, you should:
::

    source .tox/py37/bin/activate

on Linux/Mac, or if you are Windows user:
::

    .tox\py37\Scripts\activate

Then you can run pytest, linting, or directly run python scripts, and
know that the environment was created correctly by `tox`_.


Generating documentation
^^^^^^^^^^^^^^^^^^^^^^^^

The simplest way is again using `tox`_.
::

    tox -e docs

then open ``docs/build/html/index.html`` in your browser.


Running tests
^^^^^^^^^^^^^

Pytest is used for running unit tests:
::

    python -m pytest
    pytest -v -s tests/algorithms/test_triangulate.py #example for individual tests


Linting
^^^^^^^
This code conforms to the PEP8 standard. Pylint can be used to analyse the code:
::

    pylint --rcfile=tests/pylintrc sksurgerycalibration


Installing
----------

You can pip install directly from the repository as follows:
::

    pip install git+https://github.com/SciKit-Surgery/scikit-surgerycalibration 



Contributing
------------

Please see the `contributing guidelines`_.


Useful links
------------

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
.. _`source code repository`: https://github.com/SciKit-Surgery/scikit-surgerycalibration 
.. _`Documentation`: https://scikit-surgerycalibration.readthedocs.io
.. _`SciKit-Surgery`: https://github.com/SciKit-Surgery/scikit-surgery/wiki/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`anaconda`: https://www.anaconda.com/
.. _`miniconda`: https://docs.conda.io/en/latest/miniconda.html
.. _`tox`: https://tox.wiki/en/latest/
.. _`venv`: https://docs.python.org/3/library/venv.html
.. _`actions`: https://github.com/SciKit-Surgery/scikit-surgerycalibration/actions
.. _`contributing guidelines`: https://github.com/SciKit-Surgery/scikit-surgerycalibration /blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/SciKit-Surgery/scikit-surgerycalibration /blob/master/LICENSE
