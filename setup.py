"""Slowest particle simulator on earth setup.

Notes for development installation
==================================
To install for development, using the commandline do:
    pip install -e /path/to/slowest_particle_simulator_on_earth

Notes for PyPI:
===============
1. cd to repository folder
2. ```python setup.py sdist upload -r pypitest```
3. Check everything looks fine on the test server.
4. ```python setup.py sdist upload -r pypi```
"""

import os
from setuptools import setup

# read the contents of README.md
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()

setup(name='slowest_particle_simulator_on_earth',
      version='0.0.2',
      description=('A very slow particle simulator for exploding nifti files.'),
      long_description=LONG_DESCRIPTION,
      long_description_content_type='text/markdown',
      url='https://github.com/ofgulban/slowest-particle-simulator-on-earth',
      author='Omer Faruk Gulban',
      author_email="farukgulban@gmail.com",
      license='BSD-3-clause',
      packages=['slowest_particle_simulator_on_earth'],
      install_requires=['numpy>=1.17', 'matplotlib>=3.1', 'nibabel>=2.2'],
      keywords=['mri', 'nifti', 'particle', 'voxel', 'simulation', 'explosion'],
      zip_safe=True,
      entry_points={
        'console_scripts': [
            'slowest_particle_simulator_on_earth = slowest_particle_simulator_on_earth.__main__:main',
            ]},
      )
