# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:50:27 2015

@author: Administrator
"""

from constants import (FIBER_HMSTSS_ID, FIBER_RCV, MS)
from simulation import SimFiber, stim_num
from fitlif import LifModel
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os


fiber_hmstss_use = FIBER_HMSTSS_ID
fiber_id = fiber_hmstss_use
level_num = 10


class HmstssFiber(SimFiber):

    def __init__(self, level):
        self.factor = 'Hmstss'
        self.control = 'Force'
        self.level = level
        self.get_dist()
        self.load_traces()
        self.load_trans_params()
        self.get_predicted_fr()

    def get_predicted_fr(self, trans_params=None):
        """
        Returns the predicted fr based on the instance's stress/strain/sener.
        1) If `trans_params is None`:
            Perform calculation on the instance's current `self.trans_params`
            and save data to `self.predicted_fr`;
        2) Otherwise:
            Sepcify `trans_params` and return the `predicted_fr` w/o
            updating the instance's properties.
            Note: do `copy.deepcopy()` before modifying the old `trans_params`!
        """
        # Determine whether a static call or not
        update_instance = False
        if trans_params is None:
            update_instance = True
            trans_params = self.trans_params
        # Calculate predicted fr
        quantity = 'stress'
        # Get the quantity_dict_list for input
        quantity_dict_list = [{
            'quantity_array': self.traces[i][quantity],
            'max_index': self.traces[i]['max_index']}
            for i in range(stim_num)]
        # Calculate
        lifModel = LifModel(**FIBER_RCV[fiber_id])
        predicted_fr =\
            lifModel.trans_param_to_predicted_fr(
                quantity_dict_list, trans_params[fiber_id][quantity])
        # Update instance if needed
        if update_instance:
            self.predicted_fr = predicted_fr
        return predicted_fr


def load_fiber():
    fname = './pickles/hmstss.pkl'
    already_exist = os.path.isfile(fname)
    if already_exist:
        with open(fname, 'rb') as f:
            hmstssFiberList = pickle.load(f)
    else:
        hmstssFiberList = []
        for level in range(level_num):
            hmstssFiberList.append(HmstssFiber(level))
        with open(fname, 'wb') as f:
            pickle.dump(hmstssFiberList, f)
    return hmstssFiberList


if __name__ == '__main__':
    hmstssFiberList = load_fiber()
    # %% Some local constants
    thickness_array = np.linspace(125, 433, level_num)
    median_dict = {'resting': 3, 'active': 6}
    resting_grouping_list = [[8, 5, 3, 1], [11, 7, 2], [6, 4, 2], [13, 5]]
    active_grouping_list_list = [
        [[9, 8, 5, 2, 2], [11, 6, 5, 2, 1, 1], [10, 8, 4, 3, 1]],
        [[13, 9, 6, 2], [14, 8, 6, 2], [15, 8, 3, 2, 2]],
        [[7, 5, 3, 2, 2], [8, 6, 4, 1], [7, 5, 4, 3]],
        [[15, 8, 3, 2], [16, 9, 3], [14, 7, 4, 3]]]
    grouping_table = np.empty((len(resting_grouping_list),
                               len(active_grouping_list_list[0]) + 1),
                              dtype='object')
    for i in range(grouping_table.shape[0]):
        grouping_table[i, 0] = resting_grouping_list[i]
        for j in range(grouping_table.shape[1] - 1):
            grouping_table[i, j + 1] = active_grouping_list_list[i][j]
    grouping_df = pd.DataFrame(grouping_table,
                               index=['Fiber #%d' % (i + 1) for i in range(4)],
                               columns=['Resting'] + ['Active'] * 3)
    grouping_df.to_csv('./csvs/grouping.csv')

    def validate_grouping():
        """
        Validate number of MC counts.
        """
        for row in grouping_df.iterrows():
            for group in row[1]:
                print(sum(group))
    base_grouping = resting_grouping_list[0]
    stimuli = hmstssFiberList[0].static_force_exp
    # %% Function definitions

    def get_response_from_hc_grouping(
            grouping, skinlevel, base_grouping=base_grouping):
        quantity = 'stress'
        fiber_id = fiber_hmstss_use
        fname = './pickles/hmstss%d_%d%d.pkl' % (
            skinlevel, base_grouping[0], grouping[0])
        already_exist = os.path.isfile(fname)
        if already_exist:
            with open(fname, 'rb') as f:
                response = pickle.load(f)
        else:
            hmstssFiber = hmstssFiberList[skinlevel]
            trans_params_fit = hmstssFiber.trans_params
            dr = grouping[0] / base_grouping[0]
            trans_params = copy.deepcopy(trans_params_fit)
            trans_params[fiber_id][quantity][0] *= dr
            trans_params[fiber_id][quantity][1] *= dr
            if dr == 1:
                response = hmstssFiber.predicted_fr.T[1]
            else:
                response = hmstssFiber.get_predicted_fr(trans_params).T[1]
            # Deal with data saving
            with open(fname, 'wb') as f:
                pickle.dump(response, f)
        return response

    def get_sts(response, st=False):
        if st is False:
            start_idx = 0
        else:
            start_idx = response.nonzero()[0][0]
        sts = np.polyfit(stimuli[start_idx:], response[start_idx:], 1)[0]
        return sts

    def get_sts_change(base_response, response):
        return get_sts(response) - get_sts(base_response)

    def get_percent(base_response, response, overall=False):
        if overall:  # Calculate overall change
            percent = (response.sum() / base_response.sum() - 1)
        else:
            percent = (response[-1] / base_response[-1]) - 1
        return percent

    def plot_phase(grouping, skinlevel, axes,
                   fiber_id=fiber_hmstss_use, **kwargs):
        response = get_response_from_hc_grouping(grouping, skinlevel)
        axes.plot(stimuli, response, **kwargs)
        return axes
    # %% Table for first computational experiment, change skin thickness
    t42 = np.empty((level_num, len(resting_grouping_list)))
    t43 = np.empty((level_num, len(resting_grouping_list)))
    for grouping_id, base_grouping in enumerate(resting_grouping_list):
        base_response = get_response_from_hc_grouping(base_grouping, 0)
        for skinlevel in range(level_num):
            response = get_response_from_hc_grouping(base_grouping, skinlevel)
            t42[skinlevel, grouping_id] = get_percent(base_response, response)
            t43[skinlevel, grouping_id] = get_sts(response)
    # Get column and row labels
    columns = [str(grouping) for grouping in resting_grouping_list]
    rows = ['%d μm' % thickness for thickness in thickness_array]
    t42_df = pd.DataFrame(t42, columns=columns, index=rows)
    t43_df = pd.DataFrame(t43, columns=columns, index=rows)
    t42_df.to_csv('./csvs/t42.csv')
    t43_df.to_csv('./csvs/t43.csv')
    # %% Figure for first computational experiment, change skin thickness
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for grouping_id, base_grouping in enumerate(resting_grouping_list):
        plot_phase(base_grouping, 0, axs.ravel()[grouping_id],
                   ls='-', marker='s', color='k', ms=MS,
                   label='%d μm' % thickness_array[0])
        plot_phase(base_grouping, -1, axs.ravel()[grouping_id],
                   ls='-', marker='o', color='k', ms=MS,
                   label='%d μm' % thickness_array[-1])
        axs.ravel()[grouping_id].set_title(
            'Organ = %s' % str(base_grouping))
    axs[0, 0].legend(loc=2)
    for axes in axs[-1]:
        axes.set_xlabel('Force (mN)')
    for axes in axs.T[0]:
        axes.set_ylabel('Static firing (Hz)')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.1, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.savefig('./plots/lesniak_f41.png')
    plt.close(fig)
    # %% Table for third computational experiment, both skin and neuron changes
    t46 = np.empty((len(resting_grouping_list), 2))
    t47 = np.empty((len(resting_grouping_list), 2))
    columns = ['Skin constant', 'Skin changes']
    rows = []
    for grouping_id, base_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list_list[grouping_id][0]
        base_response = get_response_from_hc_grouping(
            base_grouping, median_dict['resting'])
        response_skin_constant = get_response_from_hc_grouping(
            active_grouping, median_dict['resting'])
        response_skin_changes = get_response_from_hc_grouping(
            active_grouping, median_dict['active'])
        t46[grouping_id, 0] = get_percent(
            base_response, response_skin_constant)
        t46[grouping_id, 1] = get_percent(
            base_response, response_skin_changes)
        t47[grouping_id, 0] = get_sts_change(
            base_response, response_skin_constant)
        t47[grouping_id, 1] = get_sts_change(
            base_response, response_skin_changes)
        rows.append('%s -> %s' % (str(base_grouping), str(active_grouping)))
    t46_df = pd.DataFrame(t46, columns=columns, index=rows)
    t47_df = pd.DataFrame(t47, columns=columns, index=rows)
    t46_df.to_csv('./csvs/t46.csv')
    t47_df.to_csv('./csvs/t47.csv')
    # %% Plot for thrid computational experiment
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for grouping_id, base_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list_list[grouping_id][0]
        plot_phase(
            base_grouping, median_dict['resting'], axs.ravel()[grouping_id],
            ls='-', marker='s', color='k', ms=MS, label='Rest organ')
        plot_phase(
            active_grouping, median_dict['resting'], axs.ravel()[grouping_id],
            ls='-', marker='o', color='k', ms=MS,
            label='Active organ, skin constant')
        plot_phase(
            active_grouping, median_dict['active'], axs.ravel()[grouping_id],
            ls='--', marker='^', color='k', ms=MS,
            label='Active organ, skin changes')
        axs.ravel()[grouping_id].set_title(
            'Resting = %s, active = %s' % (base_grouping, active_grouping),
            fontsize=6)
    for axes in axs[-1]:
        axes.set_xlabel('Force (mN)')
    for axes in axs.T[0]:
        axes.set_ylabel('Static firing (Hz)')
    axs[0, 0].legend(loc=2, fontsize=6)
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.1, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.savefig('./plots/lesniak_f43.png')
    plt.close(fig)
    # %% Table for second computational experiment
    mda = median_dict['active']
    mdr = 0
    skin_change_table = np.empty((len(resting_grouping_list) * len(
        active_grouping_list_list[0]), 1))
    t44 = np.empty((len(resting_grouping_list) * len(
        active_grouping_list_list[0]), 2))
    t45 = np.empty((len(resting_grouping_list) * len(
        active_grouping_list_list[0]), 2))
    active_organ_list = []
    resting_organ_list = []
    for i, base_grouping in enumerate(resting_grouping_list):
        active_grouping_list = active_grouping_list_list[i]
        for j, active_grouping in enumerate(active_grouping_list):
            active_organ_list.append(active_grouping)
            resting_organ_list.append(base_grouping)
            base_response = get_response_from_hc_grouping(
                base_grouping, mdr)
            response_skin_constant = get_response_from_hc_grouping(
                active_grouping, mdr)

            def get_skin_number():
                score = np.empty(level_num)
                for i in range(level_num):
                    response_skin_changes = get_response_from_hc_grouping(
                        active_grouping, i)
                    score[i] = get_percent(base_response,
                                           response_skin_changes)
                skin_number = np.abs(score).argmin()
                return skin_number
            skin_change_table[3 * i + j] = get_skin_number()
            response_skin_changes = get_response_from_hc_grouping(
                active_grouping, skin_change_table[3 * i + j])
            t44[3 * i + j, 0] = get_percent(
                base_response, response_skin_constant)
            t44[3 * i + j, 1] = get_percent(
                base_response, response_skin_changes)
            t45[3 * i + j, 0] = get_sts_change(
                base_response, response_skin_constant)
            t45[3 * i + j, 1] = get_sts_change(
                base_response, response_skin_changes)
    changed_to = thickness_array[skin_change_table.astype('int')].astype('int')
    changed_to_um = ['%d μm' % thickness for thickness in changed_to]
    columns = ['Resting organ', 'Active organ', 'Skin constant',
               'Skin changes', 'Skin thickness changed to']
    t44_df = pd.DataFrame(np.c_[resting_organ_list, active_organ_list,
                                t44, changed_to_um], columns=columns)
    t45_df = pd.DataFrame(np.c_[resting_organ_list, active_organ_list,
                                t45, changed_to_um], columns=columns)
    t44_df.to_csv('./csvs/t44.csv')
    t45_df.to_csv('./csvs/t45.csv')
    # %% Plot for second computational experiment
    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    for grouping_id, base_grouping in enumerate(resting_grouping_list):
        active_grouping = active_grouping_list_list[grouping_id][0]
        plot_phase(
            base_grouping, mdr, axs.ravel()[grouping_id],
            ls='-', marker='s', color='k', ms=MS, label='Rest organ')
        plot_phase(
            active_grouping, mdr, axs.ravel()[grouping_id],
            ls='-', marker='o', color='k', ms=MS,
            label='Active organ, skin constant')
        plot_phase(
            active_grouping, skin_change_table[3 * grouping_id],
            axs.ravel()[grouping_id],
            ls='--', marker='^', color='k', ms=MS,
            label='Active organ, skin changes')
        axs.ravel()[grouping_id].set_title(
            'Resting = %s, active = %s' % (base_grouping, active_grouping),
            fontsize=6)
    for axes in axs[-1]:
        axes.set_xlabel('Force (mN)')
    for axes in axs.T[0]:
        axes.set_ylabel('Static firing (Hz)')
    axs[0, 0].legend(loc=2, fontsize=6)
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.1, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.savefig('./plots/lesniak_f42.png')
    plt.close(fig)
    # %% Increase heminode for the slide
    fig, axs = plt.subplots()
    mdr = median_dict['resting']
    g = [[11, 7, 2], [15, 10, 5], [13, 9, 4, 2, 2]]
    for i in range(len(g)):
        plot_phase(g[i], mdr, axs, ls=['-', '--', ':'][i],
                   marker=['s', '^', 'o'][i], color='k',
                   ms=MS, label=str(g[i]))
    axs.legend(loc=2)
    axs.set_xlabel('Force (mN)')
    axs.set_ylabel('Static firing (Hz)')
    fig.savefig('./plots/hmstss_heminode.png')
    plt.close(fig)
