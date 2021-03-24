#!/usr/bin/env python

"""
Call `pip install -e .` to install package locally for testing.
"""

from setuptools import setup

# build command
setup(
    name="commsim",
    version="0.0.1",
    author="Jared Meek",
    packages=["commsim"],
    entry_points={
        'console_scripts': ['commsim = Species.__main__:main']
    }
)
