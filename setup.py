#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author            : Aaron Niskin <aaron@niskin.org>
# Date              : 2019-01-20
# Last Modified Date: 2019-01-20
# Last Modified By  : Aaron Niskin <aaron@niskin.org>
"""Play and solve sudoku programmatically"""

from setuptools import setup

def readme():
    """read the README file"""
    # pylint: disable=invalid-name
    with open('README.rst', 'r') as f:
        return f.read()

setup(name='sudso',
      version='0.1',
      description='Play and solve sudoku programmatically!',
      long_description=readme(),
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 2.7',
          'Topic :: Text Processing :: Linguistic',
      ],
      url='http://github.com/amniskin/sudso',
      author='Aaron Niskin',
      author_email='aaron@niskin.org',
      licence='MIT',
      packages=['sudso'],
      zip_safe=False)
