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

Developing
----------

Virtual environments
^^^^^^^
Virtualenv, venv, conda or pyenv can be used to create virtual environments to manage python packages.
You can use conda env by installing conda for your OS (`conda_installation`_) and the following yml file for its dependencies.
::
    ## Some useful commands to manage your conda env:
    ## LIST CONDA ENVS: conda list -n *VE # show list of installed packages
    ## UPDATE CONDA: conda update -n base -c defaults conda
    ## INSTALL CONDA EV: conda env create -f *VE.yml
    ## UPDATE CONDA ENV: conda env update -f *VE.yml --prune
    ## ACTIVATE CONDA ENV: conda activate *VE
    ## REMOVE CONDA ENV: conda remove -n *VE --all

    name: scikit-surgerycalibrationVE
    channels:
      - defaults
      - conda-forge
      - anaconda
    dependencies:
      - python=3.7
      - pip>=22.2.2
      - pip:
         - scikit-surgerycore
         - scikit-surgeryimage>=0.10.1
         - opencv-contrib-python-headless<4.6
         - tox>=3.26.0
         - pytest>=7.2.0
         - pylint>=2.15.9
         - jupyter
         - numpy>=1.21.6
         - scipy>=1.7.3
         - matplotlib

Cloning
^^^^^^^
You can clone the repository using the following command:
::

    git clone https://github.com/SciKit-Surgery/scikit-surgerycalibration
    git clone git@github.com:SciKit-Surgery/scikit-surgerycalibration.git # Alternatively, use password-protected SSH key.

Launching virtual env
^^^^^^^^^^^^^
Conda virtual environment is used to reproduce same python dependencies
::

    conda activate scikit-surgerycalibrationVE


Running tox
^^^^^^^^^^^^^
Tox is used to check package builds, docs, requirements and installs correctly under different environments.
::

    tox


Running tests
^^^^^^^^^^^^^
Pytest is used for running unit tests:
::

    python -m pytest
    pytest -v -s tests/algorithms/test_triangulate.py #for individual tests


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
.. _`source code repository`: https://github.com/SciKit-Surgery/scikit-surgerycalibration 
.. _`Documentation`: https://scikit-surgerycalibration.readthedocs.io
.. _`SciKit-Surgery`: https://github.com/SciKit-Surgery/scikit-surgery/wiki/home
.. _`University College London (UCL)`: http://www.ucl.ac.uk/
.. _`Wellcome`: https://wellcome.ac.uk/
.. _`EPSRC`: https://www.epsrc.ac.uk/
.. _`contributing guidelines`: https://github.com/SciKit-Surgery/scikit-surgerycalibration /blob/master/CONTRIBUTING.rst
.. _`license file`: https://github.com/SciKit-Surgery/scikit-surgerycalibration /blob/master/LICENSE
.. _`conda_installation` : https://conda.io/projects/conda/en/latest/user-guide/install/index.html