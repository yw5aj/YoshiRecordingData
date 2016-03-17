# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:58:07 2015

@author: Administrator
"""

from constants import FIBER_TOT_NUM, DT, FIBER_MECH_ID, COLOR_LIST
from simulation import SimFiber, level_num, stim_num, control_list
from fitlnp import stress2response
import numpy as np
import matplotlib.pyplot as plt
import pickle


factor_list = ['RaThick', 'RaInd']


rathick_thick_array = np.loadtxt(
    'X:/YuxiangWang/AbaqusFolder/YoshiModel/csvs/simprop.csv',
    delimiter=',').T[0]
rathick_ginf_array = np.loadtxt(
    'X:/YuxiangWang/AbaqusFolder/YoshiModel/csvs/rathickg.csv',
    delimiter=',').T[-1]
raind_ginf_array = np.loadtxt(
    'X:/YuxiangWang/AbaqusFolder/YoshiModel/csvs/raindg.csv',
    delimiter=',').T[-1]


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
    # %% Plot viscoelasticity change effect, displ control
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for level in range(level_num):
        color = COLOR_LIST[level]
        label_thick = r'%d $\mu$m, $G_\infty$ = %.3f' % (
            rathick_thick_array[level], rathick_ginf_array[level])
        label_ind = r'%d $\mu$m, $G_\infty$ = %.3f' % (
            rathick_thick_array[2], raind_ginf_array[level])
        axs[0, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].traces[3]['displ'] * 1e3,
            '-k')
        axs[0, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].traces[3]['displ'] * 1e3,
            '-k')
        axs[1, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_response[fiber_id][3],
            '-', color=color, label=label_thick)
        axs[1, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].lnp_response[fiber_id][3],
            '-', color=color, label=label_ind)
        axs[2, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_response[fiber_id][3] /
            raSimList[0][level][0].lnp_response[fiber_id][3].max(),
            '-', color=color, label=label_thick)
        axs[2, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].lnp_response[fiber_id][3] /
            raSimList[1][level][0].lnp_response[fiber_id][3].max(),
            '-', color=color, label=label_ind)
    for axes in axs[-1]:
        axes.set_xlabel('Time (s)')
    for axes in axs.ravel():
        axes.set_xlim(-0.25, 5.5)
    axs[0, 0].set_ylabel('Displacement stimuli (mm)')
    axs[0, 1].set_ylabel('Displacement stimuli (mm)')
    axs[1, 0].set_ylabel('Expected response (Hz)')
    axs[1, 1].set_ylabel('Expected response (Hz)')
    axs[2, 0].set_ylabel('Normalized expected response')
    axs[2, 1].set_ylabel('Normalized expected response')
    axs[1, 0].legend(loc=1)
    axs[1, 1].legend(loc=1)
    axs[1, 1].set_ylim(0, 80)
    fig.tight_layout()
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.1, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.savefig('./plots/relaxadapt/visco_displ.png')
    plt.close(fig)
    # %% Plot viscoelasticity change effect, force control
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for level in range(level_num):
        color = COLOR_LIST[level]
        label_thick = r'%d $\mu$m, $G_\infty$ = %.3f' % (
            rathick_thick_array[level], rathick_ginf_array[level])
        label_ind = r'%d $\mu$m, $G_\infty$ = %.3f' % (
            rathick_thick_array[2], raind_ginf_array[level])
        axs[0, 0].plot(
            raSimList[0][level][1].traces[3]['time'],
            raSimList[0][level][1].traces[3]['force'] * 1e3,
            '-k')
        axs[0, 1].plot(
            raSimList[1][level][1].traces[3]['time'],
            raSimList[1][level][1].traces[3]['force'] * 1e3,
            '-k')
        axs[1, 0].plot(
            raSimList[0][level][1].traces[3]['time'],
            raSimList[0][level][1].lnp_response[fiber_id][3],
            '-', color=color, label=label_thick)
        axs[1, 1].plot(
            raSimList[1][level][1].traces[3]['time'],
            raSimList[1][level][1].lnp_response[fiber_id][3],
            '-', color=color, label=label_ind)
        axs[2, 0].plot(
            raSimList[0][level][1].traces[3]['time'],
            raSimList[0][level][1].lnp_response[fiber_id][3] /
            raSimList[0][level][1].lnp_response[fiber_id][3].max(),
            '-', color=color, label=label_thick)
        axs[2, 1].plot(
            raSimList[1][level][1].traces[3]['time'],
            raSimList[1][level][1].lnp_response[fiber_id][3] /
            raSimList[1][level][1].lnp_response[fiber_id][3].max(),
            '-', color=color, label=label_ind)
    for axes in axs[-1]:
        axes.set_xlabel('Time (s)')
    for axes in axs.ravel():
        axes.set_xlim(-0.25, 5.5)
    axs[0, 0].set_ylabel('Force stimuli (mN)')
    axs[0, 1].set_ylabel('Force stimuli (mN)')
    axs[1, 0].set_ylabel('Expected response (Hz)')
    axs[1, 1].set_ylabel('Expected response (Hz)')
    axs[2, 0].set_ylabel('Normalized expected response')
    axs[2, 1].set_ylabel('Normalized expected response')
    axs[1, 0].legend(loc=1)
    axs[1, 1].legend(loc=1)
    axs[0, 0].set_ylim(0, 5)
    axs[0, 1].set_ylim(0, 5)
    axs[1, 0].set_ylim(0, 80)
    axs[1, 1].set_ylim(0, 80)
    fig.tight_layout()
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.1, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.savefig('./plots/relaxadapt/visco_force.png')
    plt.close(fig)
    # %% Plot viscoelasticity change effect
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))
    for level in range(level_num):
        label_thick = r'''%d $\mu$m
            $G_\infty$=%.3f''' % (
            rathick_thick_array[level], rathick_ginf_array[level])
        label_ind = r'''%d $\mu$m
            $G_\infty$=%.3f''' % (
            rathick_thick_array[2], raind_ginf_array[level])
        yoffset = - 2 * level + 2 * level_num - 2
        axs[0, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_spikes[fiber_id][3] + yoffset,
            '-k')
        axs[0, 0].text(-.1, yoffset, label_thick, ha='right', va='bottom',
                       fontsize=6)
        axs[1, 0].plot(
            raSimList[0][level][0].traces[3]['time'],
            raSimList[0][level][0].lnp_response[fiber_id][3],
            '-k')
        yoffset = 2 * level
        axs[0, 1].plot(
            raSimList[1][level][0].traces[3]['time'],
            raSimList[1][level][0].lnp_spikes[fiber_id][3] + yoffset,
            '-k')
        axs[0, 1].text(-.1, yoffset, label_ind, ha='right', va='bottom',
                       fontsize=6)
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
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.savefig('./plots/relaxadapt/show_effect.png')
    plt.close(fig)
    # %% Plot effect of chaning stim
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(3, 2, figsize=(7, 6))
    for stim in range(1, stim_num):
        color = COLOR_LIST[stim]
        label = r'%d $\mu$m thickness, $G_\infty$ = %.3f' % (
            rathick_thick_array[2], rathick_ginf_array[2])
        axs[0, 0].plot(
            raSimList[0][2][0].traces[stim]['time'],
            raSimList[0][2][0].traces[stim]['displ'] * 1e3,
            '-k')
        axs[0, 1].plot(
            raSimList[0][2][1].traces[stim]['time'],
            raSimList[0][2][1].traces[stim]['force'] * 1e3,
            '-k')
        axs[1, 0].plot(
            raSimList[0][2][0].traces[stim]['time'],
            raSimList[0][2][0].lnp_response[fiber_id][stim],
            '-k')
        axs[1, 1].plot(
            raSimList[0][2][1].traces[stim]['time'],
            raSimList[0][2][1].lnp_response[fiber_id][stim],
            '-k')
        axs[2, 0].plot(
            raSimList[0][2][0].traces[stim]['time'],
            raSimList[0][2][0].lnp_response[fiber_id][stim] /
            raSimList[0][2][0].lnp_response[fiber_id][stim].max(),
            '-k')
        axs[2, 1].plot(
            raSimList[0][2][1].traces[stim]['time'],
            raSimList[0][2][1].lnp_response[fiber_id][stim] /
            raSimList[0][2][1].lnp_response[fiber_id][stim].max(),
            '-k')
    for axes in axs[-1]:
        axes.set_xlabel('Time (s)')
    for axes in axs.ravel():
        axes.set_xlim(-0.25, 5.5)
    axs[0, 0].set_ylabel('Displacement stimuli (mm)')
    axs[0, 1].set_ylabel('Force stimuli (mN)')
    axs[1, 0].set_ylabel('Expected response (Hz)')
    axs[1, 1].set_ylabel('Expected response (Hz)')
    axs[2, 0].set_ylabel('Normalized expected response')
    axs[2, 1].set_ylabel('Normalized expected response')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.11, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.suptitle(label, fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=.92)
    fig.savefig('./plots/relaxadapt/stim.png')
    plt.close(fig)

