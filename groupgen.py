# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 16:40:15 2015

@author: Administrator
"""

import numpy as np


def add_mc_uniform(mcnc_grouping, n_mc=None, n_group=None):
    # Determine n_mc and n_group if not given
    if n_mc is None:
        n_mc = int(mcnc_grouping.sum() * 0.522)
    if n_group is None:
        n_group = mcnc_grouping.size + 1
    assert mcnc_grouping.size <= n_group, 'New structure must have more'+\
        ' branches than the old structure'        
    # Create a new structure
    mcnc_grouping_new = np.zeros(n_group)
    mcnc_grouping_new[:mcnc_grouping.size] = mcnc_grouping
    # Start adding MC
    for i in range(n_mc):
        mcnc_grouping_new[int(np.random.rand()*n_group)] += 1
    return mcnc_grouping_new
    

if __name__ == '__main__':
    mcnc_grouping_4 = np.array([8, 5, 3, 1])
    mcnc_grouping_3 = np.array([7, 5, 2])
    mcnc_grouping_3_new = add_mc_uniform(mcnc_grouping_3)
    mcnc_grouping_4_new = add_mc_uniform(mcnc_grouping_4)
    mcnc_grouping_3_manual = np.array([8, 7, 4, 2])
    mcnc_grouping_4_manual = np.array([9, 8, 5, 2, 1])
