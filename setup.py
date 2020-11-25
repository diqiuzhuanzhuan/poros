# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import setuptools
import codecs


with codecs.open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poros",
    version="0.0.53",
    author="Feynman",
    author_email="diqiuzhuanzhuan@gmail.com",
    description="some useful code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diqiuzhuanzhuan/poros",
    packages=setuptools.find_packages(),
    install_requires=[
        "tensorflow>=2.2.0",
        "tensorflow-addons",
        "matplotlib",
        "seqeval",
        "tf2crf",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
