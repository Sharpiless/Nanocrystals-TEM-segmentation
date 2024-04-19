"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.
"""

import sys, os, glob, pathlib, time
import numpy as np
from natsort import natsorted

from cellpose.gui import gui

import logging


# settings re-grouped a bit
def main():  
    gui.run()


if __name__ == "__main__":
    main()
