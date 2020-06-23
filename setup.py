# coding=utf-8
"""
Setup for scikit-surgerycalibration
"""

from setuptools import setup, find_packages
import versioneer

# Get the long description
with open('README.rst') as f:
    long_description = f.read()

setup(
    name='scikit-surgerycalibration',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description='scikit-surgerycalibration provides algorithms designed for the calibration of surgical instruments',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='https://github.com/UCL/scikit-surgerycalibration ',
    author='Stephen Thompson',
    author_email='YOUR-EMAIL@ucl.ac.uk',
    license='BSD-3 license',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Developers',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',


        'License :: OSI Approved :: BSD License',


        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3',

        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
    ],

    keywords='medical imaging',

    packages=find_packages(
        exclude=[
            'doc',
            'tests',
        ]
    ),

    install_requires=[
        'numpy',
        'ipykernel',
        'nbsphinx',
        'scipy',
        'opencv-contrib-python',
        'scikit-surgerycore',
        'scikit-surgeryimage',
        'scikit-surgeryopencvcpp',
    ],

    entry_points={
        'console_scripts': [
                'sksPivotCalibration=sksurgerycalibration.ui.pivot_calibration_command_line:main',
                'sksVideoCalibration=sksurgerycalibration.ui.video_calibration_command_line:main',
                ],
    },
)
