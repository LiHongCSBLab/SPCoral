from importlib.metadata import version

__all__ = ["model", "plot", "pp", "analysis"]

from . import model
from . import plot
from . import pp
from . import analysis

# __version__ = version("spcoral")

"""
# Author: Heqi Wang(Lihong lab)
# File Name:SPCoral
# github:https://github.com/LiHongCSBLab/
# Description: A python package for registering and integrating cross-model spatial omics
"""

__author__ = "Heqi Wang(Lihong lab, SINH)"
__email__ = [
    "wangheqi2021@sinh.ac.cn"
    "lihong01@sibs.ac.cn"
]