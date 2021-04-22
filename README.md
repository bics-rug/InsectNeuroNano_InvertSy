# InvertSy ![GitHub top language](https://img.shields.io/github/languages/top/InsectRobotics/InvertSy) [![GitHub license](https://img.shields.io/github/license/InsectRobotics/InvertSy)](https://github.com/InsectRobotics/InvertSy/blob/main/LICENSE) ![GitHub last-commit](https://img.shields.io/github/last-commit/InsectRobotics/InvertSy) [![Build Status](https://travis-ci.com/InsectRobotics/InvertSy.svg?token=tyo7V4GZ2Vq6iYPrXVLD&branch=main)](https://travis-ci.com/InsectRobotics/InvertSy)

This Python package implements environments such as the *sky* and an *AntWorld of
vegetation*, using  simple-to-install python packages, e.g. NumPy and SciPy. These
environments contain information that humans can or cannot detect but invertebrates
definitely can (e.g. polarised light in the sky). This package also contains some
examples of how to use the [InvertPy](https://github.com/InsectRobotics/InvertPy) package.


### invertsy.agent

### invertsy.env

### inversy.sim

## Environment

In order to be able to use this code, the required packages are listed below:
* [Python 3.8](https://www.python.org/downloads/release/python-380/)
* [NumPy](https://numpy.org/)  >= 1.20.1
* [SciPy](https://www.scipy.org/) >= 1.6.1
* [Matplotlib]() >= 3.3.4
* [InvertPy](https://github.com/InsectRobotics/InvertPy)

## Installation

In order to install the package and reproduce the results of the manuscript you need to clone
the code, navigate to the main directory of the project, install the dependencies and finally
the package itself. Here is an example code that installs the package:

```commandline
mkdit ~/src
cd ~/src
git clone https://github.com/InsectRobotics/InvertSy.git
cd InvertPy
pip install -r requirements.txt
pip install .
```
Note that the [pip](https://pypi.org/project/pip/) project is needed for the above installation.

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/InsectRobotics/InvertSy/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2021, Insect robotics Group, Institute of Perception,
Action and Behaviour, School of Informatics, the University of Edinburgh.
