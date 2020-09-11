from setuptools import setup, find_packages
import os
from io import open as io_open

source_directory = os.path.abspath(os.path.dirname(__file__))
requirements_text = os.path.join(source_directory, 'requirements.txt')
with io_open(requirements_text, mode='r') as fd:
    install_requires = [i.strip() for i in fd.read().strip().split('\n')]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "sigmet",
    version = "0.9.0",
    author = "Arunav Gupta, Camille Dunning, Lucas Nguyen, Ka Ming Chan",
    author_email = "arunavg@ucsd.edu, adunning@ucsd.edu, kmc026@ucsd.edu",
    description = "A Python package to find and measure negative price shocks in financial time-series data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/agupta01/sigmet",
    packages = find_packages('sigmet'),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Development Status :: 4 - Beta",
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ],
    python_requires = '>=3.6',
    install_requires = install_requires,
    package_data={
        'sigmet': ['README.md', 'LICENSE', 'requirements.txt'],
    }
)

