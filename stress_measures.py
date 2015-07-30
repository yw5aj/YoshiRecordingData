# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:29:12 2015

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from simulation import MAX_RADIUS, dist_key_list
from simulation import SimFiber


measure_list = ['msmaxprin', 'msmidprin', 'msminprin', 'msmises']


class StressMeasureFiber(SimFiber):

    def __init__(self, factor='SkinThick', level=2, control='Displ'):
        SimFiber.__init__(self, factor, level, control)
        self.get_stress_measures()
        return

    def get_stress_measures(self):
        self.get_dist(key_list=dist_key_list + measure_list)
        return


if __name__ == '__main__':
    stressMeasureFiber = StressMeasureFiber()
    # %% Plot distributions
    fig, axs = plt.subplots(1, 1)
    # Plot the surface pressure
    xscale = 1e3
    mscale = 1e-3
    axs.plot(stressMeasureFiber.dist[2]['mxold'][-1] * xscale,
             -stressMeasureFiber.dist[2]['msminprin'][-1] * mscale,
             label='Min principal (max compressive) stress')
    axs.plot(stressMeasureFiber.dist[2]['mxold'][-1] * xscale,
             -stressMeasureFiber.dist[2]['msmidprin'][-1] * mscale,
             label='Mid principal stress')
    axs.plot(stressMeasureFiber.dist[2]['mxold'][-1] * xscale,
             -stressMeasureFiber.dist[2]['msmaxprin'][-1] * mscale,
             label='Max principal stress')
    axs.plot(stressMeasureFiber.dist[2]['mxold'][-1] * xscale,
             stressMeasureFiber.dist[2]['msmises'][-1] * mscale,
             label='Von Mises stress')
    axs.plot(stressMeasureFiber.dist[2]['cxold'][-1] * xscale,
             stressMeasureFiber.dist[2]['cpress'][-1] * mscale,
             label='Contact pressure')
    axs.legend(loc=2)
    axs.set_xlim(0, MAX_RADIUS * xscale)
    axs.set_xlabel('Location (mm)')
    axs.set_ylabel('Stress (kPa)')
    fig.savefig('./plots/stress_measures.png', dpi=300)
    fig.savefig('./plots/stress_measures.pdf', dpi=300)
    plt.close(fig)
    # Print out the relative ratio
    total_int = stressMeasureFiber.dist[2]['msmaxprinint'] +\
        stressMeasureFiber.dist[2]['msmidprinint'] +\
        stressMeasureFiber.dist[2]['msminprinint']
    ratio_int = np.array([stressMeasureFiber.dist[2]['msminprinint'],
                          stressMeasureFiber.dist[2]['msmidprinint'],
                          stressMeasureFiber.dist[2]['msmaxprinint']]
                          ) / total_int
