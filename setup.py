#!/usr/bin/env python

'''
    setup.py
    ----------

    Setup script

    :copyright: 2017, Juan David Adarve. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
'''

from setuptools import setup

setup(name='pltutil',
      version='0.1',
      description='Usefull stuff for matplotlib',
      url='https://github.com/jadarve/pltutil',
      author='Juan David Adarve',
      author_email='juanda0718@gmail.com',
      license='3-clause BSD',
      packages=['pltutil'],
      zip_safe=False,
      package_data=package_data)
