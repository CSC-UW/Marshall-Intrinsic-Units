#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# setup.py

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name="marshall2024",
    version="0.0.1",
    description="Code for A Mathematical Framework for Macro Units with Intrinsic Cause-Effect Power",
    author="Graham Findlay",
    url="http://github.com/CSC-UW/Marshall_et_al_2024",
    install_requires=["pyphi"],
    packages=["marshall2024"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
)
