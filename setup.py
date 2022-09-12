import codecs
import os
import pathlib
from typing import List

from setuptools import find_packages, setup

ROOT = pathlib.Path(__file__).resolve().parent
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
README = os.path.join(ROOT, "docs", "index.md")

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
]


def get_requirements() -> List[str]:
    with codecs.open(REQUIREMENTS) as f:
        return f.read().splitlines()


def get_long_description() -> str:
    with codecs.open(README, "rt") as f:
        return f.read()


def get_version() -> str:
    return "0.1.0"


if __name__ == "__main__":

    setup(
        name="gsxform",
        version=get_version(),
        description="Wavelet scattering transforms on graphs via PyTorch",
        url="https://github.com/armaank/gsxform",
        author="Armaan Kohli",
        author_email="kohli@cooper.edu",
        license="BSD 3-Clause",
        classifiers=CLASSIFIERS,
        packages=find_packages(),
        install_requires=get_requirements(),
        long_description=get_long_description(),
        long_description_content_type="text/markdown",
        include_package_data=True,
    )
