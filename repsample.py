# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 22:27:13 2015

@author: Administrator
"""

# %% Set-up

import pickle

import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    SimFiber, fill_between_curves,
    control_list, stim_num, stim_plot_list, quantity_list,
    MAX_RADIUS, MAX_TIME, MAX_RATE_TIME, DT, FIBER_MECH_ID)


level_num = 10
level_plot_list = range(5)


class RepSample(SimFiber):

    def __init__(self, sample_id, control):
        self.factor = 'RepSample'
        self.level = sample_id
        self.control = control
        self.get_dist()
        self.load_traces()
        self.load_trans_params()
        self.get_predicted_fr()
        self.get_dist_fr()
        self.get_mi()
        self.get_line_fit()


if __name__ == '__main__':
    run_fiber = False
    fname = './pickles/repSample_list.pkl'
    if run_fiber:
        # Generate data
        repSample_list = [[] for j in range(level_num)]
        for level in range(level_num):
            j = level
            for k, control in enumerate(control_list):
                repSample = RepSample(level, control)
                repSample_list[j].append(repSample)
                print('RepSample%d%s done ...' % (j, control))
        # Store data
        with open(fname, 'wb') as f:
            pickle.dump(repSample_list, f)
    else:
        with open(fname, 'rb') as f:
            repSample_list = pickle.load(f)
    # %% Plot variance
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 6))
    for stim in stim_plot_list:
        if stim == 2:
            color = (0, 1, 0)
        elif stim == 1:
            color = (1, 0, 0)
        elif stim == 3:
            color = (0, 0, 1)
        displ_time_array_list = []
        displ_strain_array_list = []
        displ_sener_array_list = []
        force_time_array_list = []
        force_stress_array_list = []
        for level in range(5):
            repSampleDispl = repSample_list[level][0]
            repSampleForce = repSample_list[level][1]
            displ_time_array_list.append(
                repSampleDispl.traces[stim]['time'][::100])
            displ_strain_array_list.append(
                repSampleDispl.traces[stim]['strain'][::100])
            displ_sener_array_list.append(
                repSampleDispl.traces[stim]['sener'][::100] / 1e3)
            force_time_array_list.append(
                repSampleForce.traces[stim]['time'][::100])
            force_stress_array_list.append(
                repSampleForce.traces[stim]['stress'][::100] / 1e3)
        kwargs = dict(alpha=.25, color=color, label=stim)
        fill_between_curves(displ_time_array_list, displ_strain_array_list,
                            axs[0], **kwargs)
        fill_between_curves(displ_time_array_list, displ_sener_array_list,
                            axs[1], **kwargs)
        fill_between_curves(force_time_array_list, force_stress_array_list,
                            axs[2], **kwargs)
        axs[0].plot(repSample_list[0][0].traces[stim]['time'],
                    repSample_list[0][0].traces[stim]['strain'],
                    '-', color=color, label='Average skin')
        axs[1].plot(repSample_list[0][0].traces[stim]['time'],
                    repSample_list[0][0].traces[stim][
                        'sener'] / 1e3,
                    '-', color=color, label='Average skin')
        axs[2].plot(repSample_list[0][1].traces[stim]['time'],
                    repSample_list[0][1].traces[stim][
                        'stress'] / 1e3,
                    '-', color=color, label='Average skin')
    # Set x and y lim
    for axes in axs.ravel():
        axes.set_xlim(0, MAX_TIME)
    # Formatting labels
    # x-axis
    axs[-1].set_xlabel('Time (s)')
    # y-axis for the Stimulus magnitude over time
    axs[0].set_ylabel('Internal strain')
    axs[1].set_ylabel(r'Internal SED (kPa/$m^3$)')
    axs[2].set_ylabel('Internal stress (kPa)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/RepSample/simulated_variance.png', dpi=300)
    fig.savefig('./plots/RepSample/simulated_variance.pdf', dpi=300)
    plt.close(fig)
    # %% The small simulation figures - shape
    fig_rate, axs_rate = plt.subplots(2, 1, figsize=(3.5, 4))
    fig_geom, axs_geom = plt.subplots(2, 1, figsize=(3.5, 4))
    stim = stim_num // 2
    i, factor = 0, 'SkinThick'
    level = 0
    repSampleDispl = repSample_list[level][0]
    repSampleForce = repSample_list[level][1]
    axs_rate[0].plot(repSampleDispl.traces_rate[stim]['time'],
                     repSampleDispl.traces_rate[stim]['strain'],
                     '-k', label='Internal strain rate')
    axs_rate_0_twin = axs_rate[0].twinx()
    axs_rate_0_twin.plot(repSampleDispl.traces_rate[stim]['time'],
                         repSampleDispl.traces_rate[stim]['displ'] * 1e3,
                         '--k', label='Surface velocity')
    axs_rate[1].plot(repSampleForce.traces_rate[stim]['time'],
                     repSampleForce.traces_rate[stim]['stress'] * 1e-3,
                     '-k', label='Internal stress rate')
    axs_rate_1_twin = axs_rate[1].twinx()
    axs_rate_1_twin.plot(repSampleForce.traces_rate[stim]['time'],
                         repSampleForce.traces_rate[stim]['press'] * 1e-3,
                         '--k', label='Surface pressure')
    dist_displ = repSampleDispl.dist[stim]
    dist_force = repSampleForce.dist[stim]
    axs_geom[0].plot(dist_displ['mxold'][-1, :] * 1e3,
                     dist_displ['mstrain'][-1, :],
                     '-k', label='Internal strain')
    axs_geom_0_twin = axs_geom[0].twinx()
    axs_geom_0_twin.plot(dist_displ['cxold'][-1, :] * 1e3,
                         dist_displ['cy'][-1, :] * 1e-3,
                         '--k', label='Surface deflection')
    axs_geom[1].plot(dist_force['mxold'][-1, :] * 1e3,
                     dist_force['mstress'][-1, :] * 1e-3,
                     '-k', label='Internal stress')
    axs_geom_1_twin = axs_geom[1].twinx()
    axs_geom_1_twin.plot(dist_force['cxold'][-1, :] * 1e3,
                         dist_force['cpress'][-1, :] * 1e-3,
                         '--k', label='Surface pressure')
    # Set x and y lim
    for axes in axs_rate.ravel():
        axes.set_xlim(0, MAX_RATE_TIME)
    for axes in axs_geom.ravel():
        axes.set_xlim(0, MAX_RADIUS * 1e3)
    axs_rate[1].set_ylim(bottom=0)
    axs_rate_1_twin.set_ylim(bottom=0)
    # Formatting labels
    # x-axis
    axs_rate[-1].set_xlabel('Time (s)')
    axs_geom[-1].set_xlabel('Location (mm)')
    # y-axis
    axs_rate[0].set_ylabel(r'Internal strain rate (s$^{-1}$)')
    axs_rate_0_twin.set_ylabel(r'Surface velocity (mm/s)')
    axs_rate[1].set_ylabel(r'Internal stress rate (kPa/s)')
    axs_rate_1_twin.set_ylabel(r'Surface pressure rate (kPa/s)')
    axs_geom[0].set_ylabel('Internal strain')
    axs_geom_0_twin.set_ylabel(r'Surface deflection (mm)')
    axs_geom[1].set_ylabel('Internal stress (kPa)')
    axs_geom_1_twin.set_ylabel(r'Surface pressure (kPa)')
    # Add legends
    h1, l1 = axs_rate[0].get_legend_handles_labels()
    h2, l2 = axs_rate_0_twin.get_legend_handles_labels()
    axs_rate[0].legend(h1 + h2, l1 + l2, loc=3)
    h1, l1 = axs_rate[1].get_legend_handles_labels()
    h2, l2 = axs_rate_1_twin.get_legend_handles_labels()
    axs_rate[1].legend(h1 + h2, l1 + l2, loc=1)
    h1, l1 = axs_geom[0].get_legend_handles_labels()
    h2, l2 = axs_geom_0_twin.get_legend_handles_labels()
    axs_geom[0].legend(h1 + h2, l1 + l2, loc=3)
    h1, l1 = axs_geom[1].get_legend_handles_labels()
    h2, l2 = axs_geom_1_twin.get_legend_handles_labels()
    axs_geom[1].legend(h1 + h2, l1 + l2, loc=3)
    # Add panel labels
    for axes_id, axes in enumerate(axs_rate.ravel()):
        axes.text(-.175, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    for axes_id, axes in enumerate(axs_geom.ravel()):
        axes.text(-.175, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Save figure
    fig_rate.tight_layout()
    fig_geom.tight_layout()
    fig_rate.savefig('./plots/RepSample/simulated_shape_rate.png', dpi=300)
    fig_rate.savefig('./plots/RepSample/simulated_shape_rate.pdf', dpi=300)
    fig_geom.savefig('./plots/RepSample/simulated_shape_geom.png', dpi=300)
    fig_geom.savefig('./plots/RepSample/simulated_shape_geom.pdf', dpi=300)
    plt.close(fig_rate)
    plt.close(fig_geom)
    # %% The displ - force part of the encoding plot, filled
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots()
    x_array_list, y_array_list = [], []
    for level in level_plot_list:
        x_array_list.append(repSample_list[level][0].static_displ_exp)
        y_array_list.append(repSample_list[level][0].static_force_exp)
    fill_between_curves(x_array_list, y_array_list, axs, alpha=.25, color='k')
    repSample = repSample_list[0][0]
    axs.plot(
        repSample.static_displ_exp,
        repSample.static_force_exp,
        '-k', label='Average skin')
    # X and Y limits
    axs.set_ylim(0, 15)
    axs.set_xlim(.3, .8)
    # Axes and panel labels
    axs.set_xlabel(r'Static displacement (mm)')
    axs.set_ylabel('Static force (mN)')
    # Legend
    axs.legend(loc=2)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/RepSample/encoding_skin_filled.png', dpi=300)
    fig.savefig('./plots/RepSample/encoding_skin_filled.pdf', dpi=300)
    plt.close(fig)
    # %% Plot the encoding plot with only the samllest subset
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    color = 'k'
    for k, quantity in enumerate(quantity_list[-3:-1]):
        x_displ_array_list, x_force_array_list = [], []
        y_displ_array_list, y_force_array_list = [], []
        for level in level_plot_list:
            x_displ_array_list.append(
                repSample_list[level][0].static_displ_exp)
            y_displ_array_list.append(
                repSample_list[level][0].predicted_fr[
                    fiber_id][quantity].T[1])
            x_force_array_list.append(
                repSample_list[level][0].static_force_exp)
            y_force_array_list.append(
                repSample_list[level][0].predicted_fr[
                    fiber_id][quantity].T[1])
        fill_between_curves(
            x_displ_array_list, y_displ_array_list,
            axs[0, k], color=color, alpha=.25, label=factor)
        fill_between_curves(
            x_force_array_list, y_force_array_list,
            axs[1, k], color=color, alpha=.25, label=factor)
        # Plot median
        simFiber = repSample_list[0][0]
        axs[0, k].plot(
            simFiber.static_displ_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1],
            '-k', label='Average skin mechanics')
        axs[1, k].plot(
            simFiber.static_force_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1],
            '-k', label='Average skin mechanics')
    # X and Y limits
    for axes in axs[0:2].ravel():
        axes.set_ylim(0, 50)
    for axes in axs[0]:
        axes.set_xlim(.3, .8)
    for axes in axs[1]:
        axes.set_xlim(0, 12)
    # Axes and panel labels
    for i, axes in enumerate(axs[0, :].ravel()):
        axes.set_title('%s-based Model' % ['Stress', 'Strain'][i])
    for axes in axs[0]:
        axes.set_xlabel(r'Static displacement (mm)')
    for axes in axs[1]:
        axes.set_xlabel(r'Static force (mN)')
    for axes in axs[0:2, 0].ravel():
        axes.set_ylabel('Predicted static firing (Hz)')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.175, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Legends
    handels, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handels[0:1], ['Average skin'], loc=2)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/RepSample/encoding_neural_filled_grey.png', dpi=300)
    fig.savefig('./plots/RepSample/encoding_neural_filled_grey.pdf', dpi=300)
    plt.close(fig)
