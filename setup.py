import pathlib

from setuptools import setup

ROOT = pathlib.Path(__file__).resolve().parent

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.9",
    "License :: OSI Approved :: BSD 3-Clause",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
]


def get_version() -> str:
    return "0.0.0"


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
        packages=None,
        version=None,
    )
