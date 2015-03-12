# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:50:27 2015

@author: Administrator
"""

from constants import (FIBER_MECH_ID, MARKER_LIST, COLOR_LIST, DT, FIBER_RCV,
                       FIBER_FIT_ID_LIST)
from simulation import SimFiber, quantity_list, stim_num
from fitlif import LifModel
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

class HmstssFiber(SimFiber):

    def __init__(self, factor, level, control):
        SimFiber.__init__(self, factor, level, control)
        self.get_fr_fsl_distribution()

    def get_fr_fsl_distribution(self, trans_params=None):
        """
        Calculate the predicted static, dynamic firing rate and fsl.
        """
        # Determin whether this is a static call
        update_instance = False
        if trans_params is None:
            update_instance = True
            trans_params = self.trans_params
        fsl_dict_list, frs_dict_list, frd_dict_list = [], [], []
        for fiber_id in FIBER_FIT_ID_LIST:
            lifModel = LifModel(**FIBER_RCV[fiber_id])
            fsl_dict_list.append({})
            frs_dict_list.append({})
            frd_dict_list.append({})
            for quantity in quantity_list[-3:]:
                mquantity = 'm' + quantity
                num_mcnc = self.dist[0][mquantity].shape[1]
                # Initialize the arrays to store output
                fslmat = np.empty((stim_num, num_mcnc))
                frsmat = np.empty((stim_num, num_mcnc))
                frdmat = np.empty((stim_num, num_mcnc))
                for mcnc_id in range(num_mcnc):
                    quantity_dict_list = []
                    for stim_id, dist_dict in enumerate(self.dist):
                        max_index = self.traces[stim_id]['max_index']
                        time = dist_dict['time'].T[0]
                        time_fine = np.arange(0, time.max(), DT)
                        quantity_array = dist_dict[mquantity].T[mcnc_id]
                        quantity_array = np.interp(
                            time_fine, time, quantity_array)
                        quantity_dict_list.append({
                            'quantity_array': quantity_array,
                            'max_index': max_index})
                    frs, frd, fsl = lifModel.get_fr_fsl(
                        quantity_dict_list,
                        trans_params[fiber_id][quantity])
                    fslmat[:, mcnc_id] = fsl
                    frsmat[:, mcnc_id] = frs
                    frdmat[:, mcnc_id] = frd
                fsl_dict_list[-1][quantity] = fslmat
                frs_dict_list[-1][quantity] = frsmat
                frd_dict_list[-1][quantity] = frdmat
        if update_instance:
            for key in ['fsl', 'frs', 'frd']:
                varname = '%s_dict_list' % key
                setattr(self, varname, locals()[varname])
        return fsl_dict_list, frs_dict_list, frd_dict_list


if __name__ == '__main__':
    run_fiber = False
    if run_fiber:
        hmstssFiberDict = {
            'active': {
                'force': HmstssFiber('SkinThick', 1, 'Force'),
                'displ': HmstssFiber('SkinThick', 1, 'Displ')},
            'resting': {
                'force': HmstssFiber('SkinThick', 0, 'Force'),
                'displ': HmstssFiber('SkinThick', 0, 'Displ')}}
        with open('./pickles/hmstss.pkl', 'wb') as f:
            pickle.dump(hmstssFiberDict, f)
    else:
        with open('./pickles/hmstss.pkl', 'rb') as f:
            hmstssFiberDict = pickle.load(f)
    # %% Some local constants
    fiber_id = FIBER_MECH_ID
    resting_grouping_list = [[8, 5, 3, 1], [11, 7, 2], [5, 4, 3, 1], [7, 5, 2]]
    active_grouping_list = [[9, 8, 5, 2, 1], [13, 11, 4], [6, 5, 4, 2, 1, 1],
                            [8, 7, 4, 2]]
    grouping_df = pd.DataFrame(
        {'Resting': resting_grouping_list,
         'Active': active_grouping_list},
        index=['Group #%d' % (i+1) for i in range(len(active_grouping_list))])
    grouping_df = grouping_df[['Resting', 'Active']]
    typical_grouping_id_list = [0, 1]
    base_grouping = resting_grouping_list[0]
    # %% Function definitions

    def calculate_response_from_hc_grouping(
            grouping, skinphase, control, base_grouping=base_grouping,
            savedata=True, updatedata=False):
        """
        By default, will first detect whether data got saved, then save if no
        previous file exists.
        If savedata = False, then only do the calculations and return outputs.
        If savedata = True and updatedata = True, then overwrite.
        """
        hmstssFiber = hmstssFiberDict[skinphase][control]
        trans_params_fit = hmstssFiber.trans_params
        dr = grouping[0] / base_grouping[0]
        trans_params = copy.deepcopy(trans_params_fit)
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                trans_params[fiber_id][quantity][0] *= dr
                trans_params[fiber_id][quantity][1] *= dr
        if dr == 1:
            fsl_dict_list, frs_dict_list, frd_dict_list = (
                hmstssFiber.fsl_dict_list, hmstssFiber.frs_dict_list,
                hmstssFiber.frd_dict_list)
        else:
            fsl_dict_list, frs_dict_list, frd_dict_list = \
                hmstssFiber.get_fr_fsl_distribution(trans_params)
        stimuli = getattr(
            hmstssFiber, 'static_%s_exp' % hmstssFiber.control.lower())
        varlist = ['stimuli', 'fsl_dict_list', 'frs_dict_list',
                   'frd_dict_list']
        response_grouping = {}
        for varname in varlist:
            response_grouping[varname] = locals()[varname]
        # Deal with data saving
        if savedata:
            fname = './pickles/%s_%s_%d%d.pkl' % (
                skinphase, control, base_grouping[0], grouping[0])
            already_exist = os.path.isfile(fname)
            if already_exist and not updatedata:
                pass
            else:
                with open(fname, 'wb') as f:
                    pickle.dump(response_grouping, f)
        return response_grouping

    def get_response_from_hc_grouping(
            grouping, skinphase, quantity, control, coding,
            fiber_id=FIBER_MECH_ID, base_grouping=base_grouping):
        fname = './pickles/%s_%s_%d%d.pkl' % (
            skinphase, control, base_grouping[0], grouping[0])
        already_exist = os.path.isfile(fname)
        if already_exist:
            with open(fname, 'rb') as f:
                response_grouping = pickle.load(f)
        else:
            response_grouping = calculate_response_from_hc_grouping(
                grouping=grouping, skinphase=skinphase, control=control,
                base_grouping=base_grouping)
        stimuli = response_grouping['stimuli']
        response = response_grouping['%s_dict_list' % coding][
            fiber_id][quantity]
        return stimuli, response

    def plot_phase_single(
            grouping_dict, groupphase, skinphase, coding, control, quantity,
            axes, fiber_id=FIBER_MECH_ID, **kwargs):
        grouping = grouping_dict[groupphase]
        kwargs['label'] = '%s skin, %s grouping' % (
            skinphase.capitalize(),
            groupphase)
        if groupphase == 'resting' and skinphase == 'resting':
            kwargs['ls'] = '-'
        elif groupphase == 'resting' and skinphase == 'active':
            kwargs['ls'] = '-.'
        elif groupphase == 'active' and skinphase == 'resting':
            kwargs['ls'] = ':'
        elif groupphase == 'active' and skinphase == 'active':
            kwargs['ls'] = '--'
        stimuli, response = get_response_from_hc_grouping(
            grouping, skinphase, quantity, control, coding, fiber_id)
        response = response.T[0]
        axes.plot(stimuli, response, **kwargs)
        return axes
    # %% Single fiber, rate coding
    fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.5))
    for grouping_id, resting_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list[grouping_id]
        grouping_dict = dict(resting=resting_grouping, active=active_grouping)
        marker = MARKER_LIST[grouping_id]
        color = COLOR_LIST[grouping_id]
        kwargs = dict(marker=marker, color=color, mfc=color, mec=color, ms=3)
        # Here, only use stress as quantity, force controlled
        quantity = 'stress'
        control = 'force'
        hc_list = [(('resting', 'resting'), ('active', 'resting')),
                   (('resting', 'resting'), ('resting', 'active')),
                   (('resting', 'resting'), ('active', 'active'))]
        for row, hc_tuple in enumerate(hc_list):
            for col, coding in enumerate(['frs', 'frd']):
                for hc in hc_tuple:
                    plot_phase_single(
                        grouping_dict, hc[0], hc[1], coding, control, quantity,
                        axs[row, col], **kwargs)
    # Formatting
    for axes in axs[-1]:
        axes.set_xlabel('Force (mN)')
    for i, axes in enumerate(axs.ravel()):
        if i % 2 == 0:
            axes.set_ylim(0, 60)
            axes.set_ylabel(r'$\mathit{r_s}$ (Hz)')
        else:
            axes.set_ylim(0, 105)
            axes.set_ylabel(r'$\mathit{r_d}$ (Hz)')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.35, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    axs[0, 0].set_title('Static hold')
    axs[0, 1].set_title('Dynamic ramp')
    fig.tight_layout()
    fig.savefig('./plots/hmstss_single_fiber.png')
    plt.close(fig)
    # %% Old plots
    fig1, axs1 = plt.subplots(2, 3, figsize=(7.5, 5))
    fig2, axs2 = plt.subplots(3, 1, figsize=(3.27, 7.5))
    for grouping_id, resting_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list[grouping_id]
        grouping_dict = dict(resting=resting_grouping, active=active_grouping)
        marker = MARKER_LIST[grouping_id]
        color = COLOR_LIST[grouping_id]
        kwargs = dict(marker=marker, color=color, mfc=color, mec=color)
        for i, quantity in enumerate(quantity_list[-3:]):
            # Do fig1, for two typical cases
            if grouping_id in typical_grouping_id_list:
                j = typical_grouping_id_list.index(grouping_id)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'resting', 'frs', 'force', 'stress',
                    axs1[j, i], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'active', 'frs', 'force', 'stress',
                    axs1[j, i], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'active', 'active', 'frs', 'force', 'stress',
                    axs1[j, i], **kwargs)
            # Do fig2, for multiple sensitivities
            if quantity == 'stress':
                plot_phase_single(
                    grouping_dict,
                    'active', 'active', 'frs', 'force', 'stress',
                    axs2[2], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'resting', 'frs', 'force', 'stress',
                    axs2[2], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'active', 'resting', 'frs', 'force', 'stress',
                    axs2[0], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'resting', 'frs', 'force', 'stress',
                    axs2[0], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'active', 'frs', 'force', 'stress',
                    axs2[1], **kwargs)
                plot_phase_single(
                    grouping_dict,
                    'resting', 'resting', 'frs', 'force', 'stress',
                    axs2[1], **kwargs)
    # Add legends for fig 2
    handles, labels = axs2[0].get_legend_handles_labels()
    handle = [[] for i in range(3)]
    label = [[] for i in range(3)]
    import matplotlib.lines as mlines
    handle[0] = [mlines.Line2D([], [], ls=h.get_linestyle(), c=h.get_c())
                 for h in handles[:2]] +\
                [mlines.Line2D([], [], ls='None', marker=h.get_marker(),
                               mec=h.get_mec(), mfc=h.get_mfc())
                 for h in handles[1::2]]
    label[0] = labels[:2] + [
        'Fiber #%d' % (i + 1)
        for i in range(len(resting_grouping_list))]
    handles, labels = axs2[1].get_legend_handles_labels()
    handle[1] = [mlines.Line2D([], [], ls=h.get_linestyle(), c=h.get_c())
                 for h in handles[:2]]
    label[1] = labels[:2]
    handles, labels = axs2[2].get_legend_handles_labels()
    handle[2] = [mlines.Line2D([], [], ls=h.get_linestyle(), c=h.get_c())
                 for h in handles[:2]]
    label[2] = labels[:2]
    for axes_id, axes in enumerate(axs2):
        axes.legend(handle[axes_id], label[axes_id], loc=2)
        axes.set_ylim(0, 70)
    # Add legends for fig 1
    for axes_id, axes in enumerate(axs1[0]):
        if axes_id == 0:
            axes.legend(loc=2)
        axes.set_ylim(0, 60)
        axes.text(.1, .5, 'Group #%d' % (typical_grouping_id_list[0] + 1),
                  transform=axes.transAxes, fontsize=8)
    for axes_id, axes in enumerate(axs1[1]):
        if axes_id == 0:
            axes.legend(loc=2)
        axes.set_ylim(0, 80)
        axes.text(.1, .5, 'Group #%d' % (typical_grouping_id_list[1] + 1),
                  transform=axes.transAxes, fontsize=8)
    # Add labels
    for axes in np.r_[axs1.T[0], axs2]:
        axes.set_ylabel('Predicted mean firing (Hz)')
    for axes in np.r_[axs1[-1], [axs2[-1]]]:
        axes.set_xlabel('Force (mN)')
    # Add titles
    for axes_id, axes in enumerate(axs1[0]):
        axes.set_title('%s-based model' % ['Stress', 'Strain', 'SED'][axes_id])
    axs2[0].set_title('Both skin and grouping change')
    axs2[1].set_title('Only grouping changes')
    axs2[2].set_title('Only skin changes')
    # Format and save
    for axes_id, axes in enumerate(axs1.ravel()):
        axes.text(-.14, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    for axes_id, axes in enumerate(axs2.ravel()):
        axes.text(-.125, 1.05, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig1.tight_layout()
    fig2.tight_layout()
    fig1.savefig('./plots/grouping_typical.png')
    fig2.savefig('./plots/grouping_all.png')
    plt.close(fig1)
    plt.close(fig2)
