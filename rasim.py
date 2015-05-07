# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:58:07 2015

@author: Administrator
"""

from constants import FIBER_TOT_NUM, DT, FIBER_MECH_ID
from simulation import SimFiber, level_num, stim_num
from fitlnp import stress2response
import numpy as np
import matplotlib.pyplot as plt
import pickle


factor_list = ['RaThick', 'RaInd']
control_list = ['Displ', 'Force']


class RaSim(SimFiber):

    def __init__(self, factor, level, control):
        self.factor = factor
        self.level = level
        self.control = control
        self.get_dist()
        self.load_traces()
        self.predict_lnp()

    def predict_lnp(self):
        # Load lnp parameters
        lnp_params = []
        for i in range(FIBER_TOT_NUM):
            lnp_params.append(np.loadtxt('./csvs/lnp_params_%d.csv' % i,
                                         delimiter=','))
        # Calculate firing rate
        self.lnp_response = []
        self.lnp_spikes = []
        for fiber_id in range(FIBER_TOT_NUM):
            lnp_response_fiber = []
            lnp_spikes_fiber = []
            lnp_params_fiber = lnp_params[fiber_id]
            for stim_id in range(stim_num):
                response = stress2response(lnp_params_fiber[:2],
                                           lnp_params_fiber[2:],
                                           self.traces[stim_id]['stress'],
                                           self.traces[stim_id]['time'])
                spikes = np.random.poisson(response * DT)
                # Merge two spikes into one
                spikes[spikes >= 1] = 1
                lnp_response_fiber.append(response)
                lnp_spikes_fiber.append(spikes)
            self.lnp_response.append(lnp_response_fiber)
            self.lnp_spikes.append(lnp_spikes_fiber)


if __name__ == '__main__':
    run_fiber = False
    if run_fiber:
        raSimList = [[[] for j in range(level_num)]
                     for i in range(len(factor_list))]
        for i, factor in enumerate(factor_list):
            for level in range(level_num):
                j = level
                for k, control in enumerate(control_list):
                    raSim = RaSim(factor, level, control)
                    raSimList[i][j].append(raSim)
                    print(factor + str(level) + control + ' is done.')
        with open('./pickles/raSimList.pkl', 'wb') as f:
            pickle.dump(raSimList, f)
    else:
        with open('./pickles/raSimList.pkl', 'rb') as f:
            raSimList = pickle.load(f)
    # %% Plot viscoelasticity change effect
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    for level in range(level_num):
        axs[0, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_spikes[fiber_id][3] - 2 * level +
            2 * level_num - 2,
            '-k')
        axs[1, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_response[fiber_id][3],
            '-k')
        axs[0, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].lnp_spikes[fiber_id][3] + 2 * level,
            '-k')
        axs[1, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].lnp_response[fiber_id][3],
            '-k')
    for axes in axs[0]:
        axes.set_ylim(-1, 2 * level_num)
        axes.axis('off')
    for axes in axs[1]:
        axes.set_xlabel('Time (s)')
    for axes in axs.ravel():
        axes.set_xlim(-.25, 5.5)
    axs[1, 0].set_ylabel('Expected firing rate (Hz)')
    axs[0, 0].set_title(r'Change in $G_\infty$ due to thickness variance')
    axs[0, 1].set_title(r'Change in $G_\infty$ due to individual differences')
    fig.suptitle('5-second ramp-and-hold displacement stimuli', fontsize=12)
    # %% Plot displacement vs. force control
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    for level in range(level_num):
        axs[0, 0].plot(raSimList[0][level][0].traces[3]['time'],
                       raSimList[0][level][0].lnp_response[fiber_id][3],
                       '-k')
        axs[1, 0].plot(raSimList[0][level][1].traces[3]['time'],
                       raSimList[0][level][1].lnp_response[fiber_id][3],
                       '-k')
        axs[0, 1].plot(raSimList[1][level][0].traces[3]['time'],
                       raSimList[1][level][0].lnp_response[fiber_id][3],
                       '-k')
        axs[1, 1].plot(raSimList[1][level][1].traces[3]['time'],
                       raSimList[1][level][1].lnp_response[fiber_id][3],
                       '-k')
    for axes in axs.ravel():
        axes.set_xlim(-.25, 5.5)
    # %% Plot impact of the stim id
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    level = 2
    for stim in range(stim_num):
        axs[0, 0].plot(raSimList[0][level][0].traces[stim]['time'],
                       raSimList[0][level][0].lnp_response[fiber_id][stim],
                       '-k')
        axs[0, 1].plot(raSimList[0][level][1].traces[stim]['time'],
                       raSimList[0][level][1].lnp_response[fiber_id][stim],
                       '-k')
        axs[1, 0].plot(
            raSimList[0][level][0].traces[stim]['time'],
            raSimList[0][level][0].lnp_response[fiber_id][stim] /
            raSimList[0][level][0].lnp_response[fiber_id][stim].max(),
            '-k')
        axs[1, 1].plot(
            raSimList[0][level][1].traces[stim]['time'],
            raSimList[0][level][1].lnp_response[fiber_id][stim] /
            raSimList[0][level][1].lnp_response[fiber_id][stim].max(),
            '-k')
    for axes in axs.ravel():
        axes.set_xlim(-.25, 5.5)
