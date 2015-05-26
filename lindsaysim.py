# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:04:53 2015

@author: Administrator
"""

from simulation import SimFiber
from scipy.io import savemat


if __name__ == '__main__':
    lindsayForce = SimFiber('Lindsay', '', 'Force')
    lindsayDispl = SimFiber('Lindsay', '', 'Displ')
    data = {
        'force_control': lindsayForce.traces,
        'displ_control': lindsayDispl.traces}
    savemat('./pickles/lindsaysim.mat', data, do_compression=True)