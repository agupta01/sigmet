from setuptools import setup, find_packages
import os
from io import open as io_open

source_directory = os.path.abspath(os.path.dirname(__file__))
requirements_text = os.path.join(source_directory, 'requirements.txt')
with io_open(requirements_text, mode='r') as fd:
    install_requires = [i.strip() for i in fd.read().strip().split('\n')]

#here = path.abspath(path.dirname(__file__))
#with open(path.join(here, 'README.md'), encoding='utf-8') as f:
#    long_description = f.read()

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "sigmet",
    version = "0.0.1",
    author = "Zillow Team",
    author_email = "author@example.com",
    description = "A tool that measures shocks",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/agupta01/sigmet",
    packages = find_packages('sigmet'),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent (Linux in progress)",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: UCSD",
        "Topic :: Software Development :: Build Tools",
    ],
    python_requires = '>=3.6',
    install_requires = install_requires,
#    extras_require={  # Optional
#        'dev': ['check-manifest'],
#        'test': ['coverage'],
#    }
    package_data={
        'sigmet': ['README.md', 'LICENCE', 'requirements.txt'],
    }
)

