# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:58:07 2015

@author: Administrator
"""

from simulation import SimFiber, stim_num
from fitlnp import stress2response
import numpy as np
import matplotlib.pyplot as plt


class RaSim(SimFiber):

    def __init__(self, factor, level, control):
        self.factor = factor
        self.level = level
        self.control = control


if __name__ == '__main__':
    pass