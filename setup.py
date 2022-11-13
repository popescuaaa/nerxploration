#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    description="Named entity recognition exploration study",
    author="",
    author_email="",
    url="https://github.com/popescuaaa/nerxploration",
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
