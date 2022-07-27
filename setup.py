"""
Setup file for marian-tensorboard.
Use setup.cfg to configure your project.
"""
from setuptools import setup, find_packages
from distutils.util import convert_path


def get_version():
    """Returns package version"""
    variables = {}
    version_path = convert_path("src/marian_tensorboard/version.py")
    with open(version_path, "r", encoding="utf8") as version_file:
        exec(version_file.read(), variables)
    return variables["__version__"]


if __name__ == "__main__":
    setup(version=get_version())
