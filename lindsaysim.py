# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:04:53 2015

@author: Administrator
"""

from simulation import SimFiber
import numpy as np
from scipy.io import savemat
from constants import DT

if __name__ == '__main__':
    control_list = ['Force', 'Displ']
    lindsaySimDict = {}
    for control in control_list:
        lindsaySimDict[control] = SimFiber('Lindsay', '', control)
    buffer_time = 0.175
    buffer_pts = int(buffer_time / DT)
    for control, lindsaySim in lindsaySimDict.items():
        for traces in lindsaySim.traces:
            for key, item in traces.items():
                if key != 'max_index' and key != 'time':
                    traces[key] = np.r_[item[0] * np.ones(buffer_pts), item]
                elif key == 'time':
                    traces[key] = np.arange(0, item[-1] + buffer_time, DT)
                elif key == 'max_index':
                    traces[key] = item + buffer_pts
    data = {}
    for control, lindsaySim in lindsaySimDict.items():
        data[control.lower() + '_control'] = lindsaySim.traces
    savemat('./pickles/lindsaysim.mat', data, do_compression=True)
