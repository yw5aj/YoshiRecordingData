# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:50:27 2015

@author: Administrator
"""

from constants import FIBER_MECH_ID, MARKER_LIST, COLOR_LIST, DT, FIBER_RCV
from simulation import SimFiber, quantity_list, stim_num
from fitlif import LifModel
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class HmstssFiber(SimFiber):

    def __init__(self, factor, level, control):
        SimFiber.__init__(self, factor, level, control)

    def get_fr_fsl_distribution(self, trans_params=None):
        # Determin whether this is a static call
        update_instance = False
        if trans_params is None:
            update_instance = True
            trans_params = self.trans_params
        # Calculated predicted fr and fsl
        # For now, only based on stress, static & dynamic
        # Fix the fiber_id to 2
        fiber_id = 2
        lifModel = LifModel(**FIBER_RCV[fiber_id])
        quantity = 'mstress'
        num_mcnc = self.dist[0][quantity].shape[1]
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
                quantity_array = dist_dict[quantity].T[mcnc_id]
                quantity_array = np.interp(time_fine, time, quantity_array)
                quantity_dict_list.append({
                    'quantity_array': quantity_array,
                    'max_index': max_index})
            frs, frd, fsl = lifModel.get_fr_fsl(
                quantity_dict_list, trans_params[fiber_id][quantity[1:]])
            fslmat[:, mcnc_id] = fsl
            frsmat[:, mcnc_id] = frs
            frdmat[:, mcnc_id] = frd
        if update_instance:
            self.fslmat, self.frsmat, self.frdmat = fslmat, frsmat, frdmat


if __name__ == '__main__':
    hmstssFiberList = dict(
        active=HmstssFiber('SkinThick', 1, 'Force'),
        resting=HmstssFiber('SkinThick', 0, 'Force'))
    # %% See how Lesniak model would say!
    fiber_id = FIBER_MECH_ID
    control = 'Force'
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
    # Convenenience function to get fiber response from different grouping

    def get_fr_from_hc_grouping(
            grouping, skinphase, quantity, fiber_id=fiber_id,
            base_grouping=base_grouping):
        hmstssFiber = hmstssFiberList[skinphase]
        trans_params_fit = hmstssFiber.trans_params
        dr = grouping[0] / base_grouping[0]
        trans_params = copy.deepcopy(trans_params_fit)
        trans_params[fiber_id][quantity][0] *= dr
        trans_params[fiber_id][quantity][1] *= dr
        predicted_fr = hmstssFiber.get_predicted_fr(trans_params=trans_params)
        static_force_exp = hmstssFiber.static_force_exp
        static_fr = predicted_fr[fiber_id][quantity][:, 1]
        return static_force_exp, static_fr
    fig1, axs1 = plt.subplots(2, 3, figsize=(7.5, 5))
    fig2, axs2 = plt.subplots(3, 1, figsize=(3.27, 7.5))
    for grouping_id, resting_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list[grouping_id]
        grouping = dict(resting=resting_grouping, active=active_grouping)
        marker = MARKER_LIST[grouping_id]
        color = COLOR_LIST[grouping_id]
        kwargs = dict(marker=marker, color=color, mfc=color, mec=color)
        for i, quantity in enumerate(quantity_list[-3:]):
            # Plot different responses, from, say resting
            def plot_phase(groupphase, skinphase, axes, **kwargs):
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
                force, response = get_fr_from_hc_grouping(grouping[groupphase],
                                                          skinphase, quantity)
                axes.plot(force, response, **kwargs)
                return axes
            # Do fig1, for two typical cases
            if grouping_id in typical_grouping_id_list:
                j = typical_grouping_id_list.index(grouping_id)
                plot_phase('resting', 'resting', axs1[j, i], **kwargs)
                plot_phase('resting', 'active', axs1[j, i], **kwargs)
                plot_phase('active', 'active', axs1[j, i], **kwargs)
            # Do fig2, for multiple sensitivities
            if quantity == 'stress':
                plot_phase('active', 'active', axs2[0], **kwargs)
                plot_phase('resting', 'resting', axs2[0], **kwargs)
                plot_phase('active', 'resting', axs2[1], **kwargs)
                plot_phase('resting', 'resting', axs2[1], **kwargs)
                plot_phase('resting', 'active', axs2[2], **kwargs)
                plot_phase('resting', 'resting', axs2[2], **kwargs)
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
