# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poros",
    version="0.0.2",
    author="Feynman",
    author_email="diqiuzhuanzhuan@gmail.com",
    description="some useful code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diqiuzhuanzhuan/poros",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
