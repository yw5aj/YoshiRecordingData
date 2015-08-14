# -*- coding: utf-8 -*-
"""
Created on Sun May  4 22:38:40 2014

@author: Yuxiang Wang
"""

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.stats import pearsonr
import pickle
from constants import (
    DT, FIBER_TOT_NUM, MARKER_LIST, COLOR_LIST, MS, FIBER_MECH_ID,
    FIBER_FIT_ID_LIST, LS_LIST, EVAL_DISPL, EVAL_FORCE, FIBER_RCV)
from fitlif import LifModel
import copy


BASE_CSV_PATH = 'X:/WorkFolder/AbaqusFolder/YoshiModel/csvs/'
factor_list = ['SkinThick', 'SkinAlpha', 'SkinGinf', 'SylgardThick',
               'SylgardC10']
factor_display_list = ['skin thickness', 'skin modulus',
                       'skin viscoelasticity', 'substrate thickness',
                       'substrate modulus']
level_num = 5
control_list = ['Displ', 'Force']
surface_list = ['displ', 'press']
quantity_list = ['displ', 'force', 'press', 'stress', 'strain', 'sener']
quantile_label_list = ['Min', 'Lower-quartile', 'Median',
                       'Upper-quartile', 'Max']
phase_list = ['dynamic', 'static']
percentage_label_list = ['%d%%' % i for i in range(50, 175, 25)]
displcoeff = np.loadtxt('./csvs/displcoeff.csv', delimiter=',')
stim_num = 6
AREA = np.pi * 1e-3**2 / 4
MAX_RADIUS = .6e-3
MAX_TIME = 5.
MAX_RATE_TIME = .3
stim_plot_list = [1, 2, 3]  # Stims to be plotted
level_plot_list = range(level_num)[1:-1]
dist_key_list = ['cpress', 'cxnew', 'cxold', 'cy', 'msener',
                 'mstrain', 'mstress', 'mxnew', 'mxold', 'my',
                 'time']
stim_in_geom_plot = 4


class SimFiber:

    def __init__(self, factor, level, control):
        """
        Class for simulated model from Abaqus.

        Parameters
        ----------
        factor : str
            Which part of the mechanics was changed. Options are 'SkinThick',
            'SkinAlpha', 'SylgardThick', 'SylgardC10'
        level : int
            An int in [0, 5]. Corresponds to min., lower-quartile, median,
            upper-quartile and max.
        control : str
            The applied control in simulation, either force or displacement,
            noted by 'Force' or 'Displ'.
        """
        self.factor = factor
        self.level = level
        self.control = control
        self.get_dist()
        self.load_traces()
        self.load_trans_params()
        self.get_predicted_fr()
        self.get_dist_fr()
        self.get_mi()
        self.get_line_fit()
        return

    def get_dist(self, key_list=dist_key_list):
        fpath = BASE_CSV_PATH
        self.dist = [{} for i in range(stim_num)]
        for stim in range(stim_num)[1:]:
            for key in key_list:
                self.dist[stim][key] = np.loadtxt(
                    fpath + self.factor + str(self.level) + str(stim - 1) +
                    self.control + '_'+key+'.csv', delimiter=',')
                # Change unit to experiment ones
                if 'y' in key:
                    self.dist[stim][key] = displcoeff[0] + displcoeff[1] *\
                        self.dist[stim][key]*(-1e6)
            argsort = self.dist[stim]['cxold'][-1].argsort()
            # Sort order in x
            for key in key_list:
                # Calculate integration over area
                if key.startswith('c'):
                    self.dist[stim][key] = (self.dist[stim][key].T[argsort]).T
            # Propagate time
            self.dist[stim]['time'] = np.tile(
                self.dist[stim]['time'][:, np.newaxis],
                self.dist[stim]['cxold'].shape[1])
            # Calculate integration over area
            for key in key_list:
                if 'time' not in key:
                    def get_field(r):
                        return np.interp(r, self.dist[stim][key[0]+'xold'][
                            -1], self.dist[stim][key][-1])
                    self.dist[stim][key+'int'] = dblquad(
                        lambda r, theta: get_field(r) * r,
                        0, 2 * np.pi,
                        lambda r: 0,
                        lambda r: MAX_RADIUS
                        )[0]
        for key, value in self.dist[1].items():
            if type(value) is float:
                self.dist[0][key] = 0.
            if type(value) is np.ndarray and not key == 'time':
                self.dist[0][key] = np.zeros_like(value)
            elif key == 'time':
                self.dist[0][key] = value
        return

    def load_trans_params(self):
        self.trans_params = []
        for fiber_id in range(FIBER_TOT_NUM):
            with open('./pickles/trans_params_%d.pkl' % fiber_id, 'rb') as f:
                self.trans_params.append(pickle.load(f))
        return

    def load_traces(self):
        fpath = BASE_CSV_PATH
        # Use stim_num - 1 to leave space for the zero-stim trace
        fname_list = [self.factor + str(self.level) + str(stim) +
                      self.control + '.csv' for stim in range(stim_num-1)]
        self.traces = [{} for i in range(stim_num)]
        self.traces_rate = [{} for i in range(stim_num)]
        # Read the non-zero output from FEM
        for i, fname in enumerate(fname_list):
            # Get all quantities
            time, force, displ, stress, strain, sener = np.loadtxt(
                fpath+fname, delimiter=',').T
            press = self.dist[i+1]['cpress'][:, 0]
            # Save absolute quantities
            fine_time = np.arange(0, time.max(), DT)
            self.traces[i+1]['time'] = fine_time
            for quantity in quantity_list:
                self.traces[i+1][quantity] = np.interp(fine_time, time,
                                                       locals()[quantity])
            self.traces[i+1]['max_index'] = self.traces[i+1]['force'].argmax()
            # Save rate quantities
            fine_time = np.arange(0, time[-1], DT)
            self.traces_rate[i+1]['time'] = fine_time
            for quantity in quantity_list:
                self.traces_rate[i+1][quantity] = np.interp(
                    fine_time, time[:-1],
                    np.diff(locals()[quantity])/np.diff(time))
            self.traces_rate[i+1]['max_index'] = self.traces[i+1]['max_index']
        # Fill the zero-stim trace
        self.traces[0]['max_index'] = self.traces[1]['max_index']
        self.traces[0]['time'] = self.traces[1]['time']
        self.traces_rate[0]['max_index'] = self.traces_rate[1]['max_index']
        self.traces_rate[0]['time'] = self.traces_rate[1]['time']
        for quantity in quantity_list:
            self.traces[0][quantity] = np.zeros_like(self.traces[0]['time'])
            self.traces_rate[0][quantity] = np.zeros_like(
                self.traces_rate[0]['time'])
        # Scale the displ
        for i in range(stim_num):
            self.traces[i]['displ'] = displcoeff[0] * 1e-6 +\
                displcoeff[1] * self.traces[i]['displ']
        # Get the FEM and corresponding displ / force
        self.static_displ_exp = np.array(
            [self.traces[i]['displ'][-1] for i in range(stim_num)]) * 1e3
        self.dynamic_displ_exp = np.array(
            [self.traces[i]['displ'][self.traces[i]['max_index']]
             for i in range(stim_num)]) * 1e3
        self.static_force_fem = np.array(
            [self.traces[i]['force'][-1] for i in range(stim_num)])
        self.dynamic_force_fem = np.array(
            [self.traces[i]['force'].max() for i in range(stim_num)])
        self.static_force_exp = self.static_force_fem * 1e3
        self.dynamic_force_exp = self.dynamic_force_fem * 1e3
        # Get the avg displ / force rate
        self.displ_rate_exp = np.array(
            [self.dynamic_displ_exp[i] / simFiber.traces[i]['max_index'] / DT
             for i in range(stim_num)])
        self.force_rate_exp = np.array(
            [self.dynamic_force_exp[i] / simFiber.traces[i]['max_index'] / DT
             for i in range(stim_num)])
        return

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
        predicted_fr = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                # Get the quantity_dict_list for input
                quantity_dict_list = [{
                    'quantity_array': self.traces[i][quantity],
                    'max_index': self.traces[i]['max_index']}
                    for i in range(stim_num)]
                # Calculate
                lifModel = LifModel(**FIBER_RCV[fiber_id])
                predicted_fr[fiber_id][quantity] =\
                    lifModel.trans_param_to_predicted_fr(
                        quantity_dict_list, trans_params[fiber_id][quantity])
        # Update instance if needed
        if update_instance:
            self.predicted_fr = predicted_fr
        return predicted_fr

    def get_dist_fr(self, trans_params=None):
        # Determine whether a static call or not
        update_instance = False
        if trans_params is None:
            update_instance = True
            trans_params = self.trans_params
        # Calculate predicted fr
        total_elem_num = self.dist[-1]['mstress'].shape[1]
        dist_fr = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                dist_fr[fiber_id][quantity] = np.empty((
                    stim_num, 2, total_elem_num))
                for elem in range(total_elem_num):
                    # Get the quantity_dict_list for input
                    quantity_dict_list = [{
                        'quantity_array':
                            np.interp(np.arange(0, self.dist[i]['time'].max(),
                                                DT),
                                      self.dist[i]['time'][:, 0],
                                      self.dist[i]['m' + quantity][:, elem]),
                        'max_index': self.traces[i]['max_index']}
                        for i in range(stim_num)]
                    # Calculate
                    lifModel = LifModel(**FIBER_RCV[fiber_id])
                    dist_fr[fiber_id][quantity][:, :, elem] =\
                        lifModel.trans_param_to_predicted_fr(
                            quantity_dict_list,
                            trans_params[fiber_id][quantity])[:, 1:]
        # Update instance if needed
        if update_instance:
            self.dist_fr = dist_fr
        return dist_fr

    def get_mi(self):
        def calculate_m(fr_array):
            rmax = fr_array.max()
            rmin = fr_array.min()
            if rmax != rmin:
#                max_idx = fr_array.argmax()
#                rmin = fr_array[:max_idx].min()
                rmin = fr_array[0]
                m = (rmax - rmin) / (rmax + rmin)
            else:
                m = 0
            return m
        mi = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                mi[fiber_id][quantity] = np.empty((stim_num, 2))
                for stim in range(stim_num):
                    for phase in range(2):
                        mi[fiber_id][quantity][stim, phase] = calculate_m(
                            self.dist_fr[fiber_id][quantity][stim, phase, :])
        self.mi = mi
        return mi

    def get_predicted_spike_array(self, trans_params=None):
        # Determine whether a static call or not
        update_instance = False
        if trans_params is None:
            update_instance = True
            trans_params = self.trans_params
        # Calculate predicted spike array
        predicted_spike_array = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                # Get the quantity_dict_list for input
                quantity_dict_list = [{
                    'quantity_array': self.traces[i][quantity],
                    'max_index': self.traces[i]['max_index']}
                    for i in range(stim_num)]
                # Calculate
                lifModel = LifModel(**FIBER_RCV[fiber_id])
                predicted_spike_array[fiber_id][quantity] =\
                    lifModel.trans_param_to_predicted_spike_array(
                        quantity_dict_list, trans_params[fiber_id][quantity])
        # Update instance if needed
        if update_instance:
            self.predicted_spike_array = predicted_spike_array
        return predicted_spike_array

    def get_line_fit(self):
        self.line_fit = [{} for i in range(FIBER_TOT_NUM)]
        self.line_fit_median_predict = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                self.line_fit[fiber_id][quantity] = {
                    'displ_dynamic': np.polyfit(
                        self.dynamic_displ_exp,
                        self.predicted_fr[fiber_id][quantity][:, 2], 1),
                    'displ_static': np.polyfit(
                        self.static_displ_exp,
                        self.predicted_fr[fiber_id][quantity][:, 1], 1),
                    'force_dynamic': np.polyfit(
                        self.dynamic_force_exp,
                        self.predicted_fr[fiber_id][quantity][:, 2], 1),
                    'force_static': np.polyfit(
                        self.static_force_exp,
                        self.predicted_fr[fiber_id][quantity][:, 1], 1)}
                self.line_fit_median_predict[fiber_id][quantity] = {
                    key: np.polyval(
                        self.line_fit[fiber_id][quantity][key],
                        globals()['EVAL_' + key[:5].upper()])
                    for key in iter(self.line_fit[fiber_id][quantity])}
        return

    def plot_predicted_fr(self, axs, fiber_id, **kwargs):
        if self.control is 'Displ':
            for i, quantity in enumerate(quantity_list[-3:]):
                axs[i, 1].plot(
                    self.static_displ_exp,
                    self.predicted_fr[fiber_id][quantity][:, 1],
                    **kwargs)
                axs[i, 0].plot(
                    self.static_displ_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2],
                    **kwargs)
        if self.control is 'Force':
            for i, quantity in enumerate(quantity_list[-3:]):
                axs[i, 1].plot(
                    self.static_force_exp,
                    self.predicted_fr[fiber_id][quantity][:, 1],
                    **kwargs)
                axs[i, 0].plot(
                    self.static_force_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2],
                    **kwargs)
        return


if __name__ == '__main__':
    run_fiber = False
    # Load experiment data
    binned_exp_list = []
    for i in range(FIBER_TOT_NUM):
        with open('./pickles/binned_exp_%d.pkl' % i, 'rb') as f:
            binned_exp_list.append(pickle.load(f))
    fname = './pickles/simFiberList.pkl'
    if run_fiber:
        # Generate data
        simFiberList = [[[] for j in
                        range(level_num)] for i in range(len(factor_list))]
        for i, factor in enumerate(factor_list):
            for level in range(level_num):
                j = level
                for k, control in enumerate(control_list):
                    simFiber = SimFiber(factor, level, control)
                    simFiberList[i][j].append(simFiber)
                    print(factor+str(level)+control+' is done.')
        # Store data
        with open(fname, 'wb') as f:
            pickle.dump(simFiberList, f)
    else:
        with open(fname, 'rb') as f:
            simFiberList = pickle.load(f)
    # %% Generate table for integration
    spatial_table = np.empty([6, 3])
    for i, factor in enumerate(factor_list[:3]):
        for j, control in enumerate(control_list):
            for k, quantity in enumerate(quantity_list[-3:]):
                iqr = np.abs(simFiberList[i][3][j].dist[
                    2]['m%sint' % quantity] - simFiberList[i][1][j].dist[
                    2]['m%sint' % quantity])
                distance = .5 * np.abs(simFiberList[i][2][j].dist[
                    3]['m%sint' % quantity] - simFiberList[i][2][j].dist[
                    1]['m%sint' % quantity])
                spatial_table[3*j+k, i] = iqr / distance
    spatial_table_sum = spatial_table.sum(axis=1)
    np.savetxt('./csvs/spatial_table.csv', spatial_table, delimiter=',')
    # %% Calculate Pearson correlation coefficients
    """
    Three rows stand for three quantities, two columns for two controls.
    Ignore the effects for each factors since it wouldn't matter.
    """
    spatial_pearsonr_table = np.empty([3, 2])
    spatial_pearsonp_table = np.empty_like(spatial_pearsonr_table)
    for i, quantity in enumerate(quantity_list[-3:]):
        for j, control in enumerate(control_list):
            dist = simFiberList[0][2][j].dist[3]
            xcoord = np.linspace(0, MAX_RADIUS, 100)
            # Get surface data
            if control == 'Force':
                surface_quantity = 'cpress'
            elif control == 'Displ':
                surface_quantity = 'cy'
            surface_data = np.interp(xcoord, dist['cxold'][-1],
                                     dist[surface_quantity][-1])
            # Get mcnc data
            mcnc_data = np.interp(xcoord, dist['mxold'][-1],
                                  dist['m'+quantity][-1])
            # Calculate correlation
            spatial_pearsonr_table[i, j], spatial_pearsonp_table[i, j] = \
                pearsonr(surface_data, mcnc_data)
    np.savetxt('./csvs/spatial_r2_table.csv', spatial_pearsonr_table**2,
               delimiter=',')
    # %% Plot distribution
    fig, axs = plt.subplots(5, 2, figsize=(6.83, 9.19), sharex=True)
    mquantity_list = ['mstress', 'mstrain', 'msener']
    cquantity_list = ['cy', 'cpress']
    for i, factor in enumerate(factor_list[:3]):
        for j, control in enumerate(control_list):
            for level in level_plot_list:
                for stim in stim_plot_list:
                    alpha = 1. - .65 * abs(level - 2)
                    if stim == 2:
                        color = (0, 0, 0, alpha)
                    elif stim == 1:
                        color = (1, 0, 0, alpha)
                    elif stim == 3:
                        color = (0, 0, 1, alpha)
                    ls = LS_LIST[i]
                    dist = simFiberList[i][level][j].dist[stim]
                    xscale = 1e3
                    for row, cquantity in enumerate(cquantity_list):
                        # Scaling the axes
                        if 'y' in cquantity:
                            cscale = 1e-3
                        elif 'ress' in cquantity:
                            cscale = 1e-3
                        # Plotting
                        axs[row, j].plot(
                            dist['cxold'][-1, :] * xscale,
                            dist[cquantity][-1, :] * cscale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
                    for row, mquantity in enumerate(mquantity_list):
                        # Scaling the axes
                        if 'ress' in mquantity or 'sener' in mquantity:
                            mscale = 1e-3
                        else:
                            mscale = 1
                        # Plotting
                        axs[row+2, j].plot(
                            dist['mxold'][-1, :] * xscale,
                            dist[mquantity][-1, :] * mscale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
    # Set x and y lim
    for axes in axs.ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Location (mm)')
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel(r'Surface pressure (kPa)')
    axs[2, 0].set_ylabel('Internal stress (kPa)')
    axs[3, 0].set_ylabel('Internal strain')
    axs[4, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        if axes_id // 2 in [0]:
            xloc = -.135
        elif axes_id // 2 in [1, 2]:
            xloc = -0.105
        elif axes_id // 2 in [3, 4]:
            xloc = -.12
        axes.text(xloc, 1.1, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(level_plot_list)//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                level_plot_list)],
        [factor_display[5:].capitalize()
         for factor_display in factor_display_list[:3]], loc=3)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=3)
    # Add subtitles
    axs[0, 0].set_title('Deflection controlled')
    axs[0, 1].set_title('Pressure controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/spatial_distribution.png', dpi=300)
    plt.close(fig)
    # %% Calculating iqr for all temporal traces
    # Calculate iqrs

    def calculate_iqr(simFiberLevelList, end_time=MAX_TIME):

        def integrate(simFiber, quantity, stim):
            time = simFiber.traces[stim]['time']
            trace = simFiber.traces[stim][quantity]
            integration = quad(
                lambda t: np.interp(t, time, trace),
                time[0], end_time)[0]
            return integration
        iqr_dict, distance_dict = {}, {}
        for quantity in quantity_list[-3:]:
            iqr_dict[quantity] = \
                np.abs(integrate(simFiberLevelList[3], quantity, 2) -
                       integrate(simFiberLevelList[1], quantity, 2))
            distance_dict[quantity] = \
                .5 * np.abs(integrate(simFiberLevelList[2], quantity, 1) -
                            integrate(simFiberLevelList[2], quantity, 3))
        return iqr_dict, distance_dict
    temporal_table = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            iqr_dict, distance_dict = calculate_iqr(
                [simFiberList[i][j][k] for j in range(level_num)])
            for row, quantity in enumerate(quantity_list[-3:]):
                temporal_table[3*k+row, i] = \
                    iqr_dict[quantity] / distance_dict[quantity]
    temporal_table_sum = temporal_table.sum(axis=1)
    np.savetxt('./csvs/temporal_table.csv', temporal_table, delimiter=',')
    # %% Calculate Pearson correlation coefficients
    temporal_pearsonr_table = np.empty([3, 2])
    temporal_pearsonp_table = np.empty_like(temporal_pearsonr_table)
    for i, quantity in enumerate(quantity_list[-3:]):
        for j, control in enumerate(control_list):
            trace = simFiberList[0][2][j].traces[3]
            end_index = (trace['time'] > MAX_TIME).nonzero()[0][0]
            surface = 'press' if control == 'Force' else 'displ'
            xdata = trace[surface][:end_index]
            ydata = trace[quantity][:end_index]
            temporal_pearsonr_table[i, j], temporal_pearsonp_table[i, j] = \
                pearsonr(xdata, ydata)
    np.savetxt('./csvs/temporal_r2_table.csv', temporal_pearsonr_table**2,
               delimiter=',')
    # %% Plot temporal traces
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(5, 2, figsize=(6.83, 9.19), sharex=True)
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            control = control.lower()
            for level in level_plot_list:
                for stim in stim_plot_list:
                    alpha = 1. - .65 * abs(level - 2)
                    if stim == 2:
                        color = (0, 0, 0, alpha)
                    elif stim == 1:
                        color = (1, 0, 0, alpha)
                    elif stim == 3:
                        color = (0, 0, 1, alpha)
                    ls = LS_LIST[i]
                    simFiber = simFiberList[i][level][k]
                    for row, surface in enumerate(surface_list):
                        sscale = 1e3 if surface == 'displ' else 1e-3
                        axs[row, k].plot(
                            simFiber.traces[stim]['time'],
                            simFiber.traces[stim][surface] * sscale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
                    for row, quantity in enumerate(quantity_list[-3:]):
                        scale = 1 if quantity is 'strain' else 1e-3
                        axes = axs[row+2, k]
                        axes.plot(
                            simFiber.traces[stim]['time'],
                            simFiber.traces[stim][quantity]*scale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel(r'Surface pressure (kPa)')
    axs[2, 0].set_ylabel('Internal stress (kPa)')
    axs[3, 0].set_ylabel('Internal strain')
    axs[4, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # Formatting
    for axes_id, axes in enumerate(axs.ravel()):
        if axes_id // 2 in [0, 3, 4]:
            xloc = -.135
        else:
            xloc = -0.12
        axes.text(xloc, 1.1, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
        axes.set_xlim(-.0, MAX_TIME)
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(level_plot_list)//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                level_plot_list)],
        [factor_display[5:].capitalize()
         for factor_display in factor_display_list[:3]], loc=4)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=4)
    # Add subtitles
    axs[0, 0].set_title('Deflection controlled')
    axs[0, 1].set_title('Pressure controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/temporal_distribution.png', dpi=300)
    plt.close(fig)
    # %% Calculating iqr for all Stimulus rate over time traces
    # Calculate iqrs

    def calculate_rate_iqr(simFiberLevelList, end_time=MAX_RATE_TIME):

        def integrate_rate(simFiber, quantity, stim):
            time = simFiber.traces[stim]['time'][:-1]
            dt = time[1] - time[0]
            trace = np.diff(simFiber.traces[stim][quantity])/dt
            integration = quad(
                lambda t: np.interp(t, time, trace),
                time[0], time[-1])[0]
            return integration
        iqr_dict, distance_dict = {}, {}
        for quantity in quantity_list[-3:]:
            iqr_dict[quantity] = \
                np.abs(integrate_rate(simFiberLevelList[3], quantity, 2) -
                       integrate_rate(simFiberLevelList[1], quantity, 2))
            distance_dict[quantity] = \
                .5 * np.abs(integrate_rate(simFiberLevelList[2], quantity, 1) -
                            integrate_rate(simFiberLevelList[2], quantity, 3))
        return iqr_dict, distance_dict
    temporal_rate_table = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            iqr_dict, distance_dict = calculate_rate_iqr(
                [simFiberList[i][j][k] for j in range(level_num)])
            for row, quantity in enumerate(quantity_list[-3:]):
                temporal_rate_table[3*k+row, i] = \
                    iqr_dict[quantity] / distance_dict[quantity]
    temporal_rate_table_sum = temporal_table.sum(axis=1)
    np.savetxt('./csvs/temporal_rate_table.csv', temporal_rate_table,
               delimiter=',')
    # %% Plot temporal trace rate
    # Calculate Pearson correlation coefficients
    temporal_rate_pearsonr_table = np.empty([3, 2])
    temporal_rate_pearsonp_table = np.empty_like(temporal_rate_pearsonr_table)
    for i, quantity in enumerate(quantity_list[-3:]):
        for j, control in enumerate(control_list):
            trace = simFiberList[0][2][j].traces[3]
            end_index = (trace['time'] > MAX_RATE_TIME).nonzero()[0][0]
            surface = 'press' if control == 'Force' else 'displ'
            xdata = np.diff(trace[surface][:end_index])
            ydata = np.diff(trace[quantity][:end_index])
            temporal_rate_pearsonr_table[i, j], \
                temporal_rate_pearsonp_table[i, j] = pearsonr(xdata, ydata)
    np.savetxt('./csvs/temporal_rate_r2_table.csv',
               temporal_rate_pearsonr_table**2, delimiter=',')
    # %% Plot Stimulus rate over time traces
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots(5, 2, figsize=(6.83, 9.19), sharex=True)
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            control = control.lower()
            for level in level_plot_list:
                for stim in stim_plot_list:
                    alpha = 1. - .65 * abs(level - 2)
                    if stim == 2:
                        color = (0, 0, 0, alpha)
                    elif stim == 1:
                        color = (1, 0, 0, alpha)
                    elif stim == 3:
                        color = (0, 0, 1, alpha)
                    ls = LS_LIST[i]
                    simFiber = simFiberList[i][level][k]
                    for row, surface in enumerate(surface_list):
                        sscale = 1e3 if surface == 'displ' else 1e-3
                        axs[row, k].plot(
                            simFiber.traces_rate[stim]['time'],
                            simFiber.traces_rate[stim][surface] * sscale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
                    for row, quantity in enumerate(quantity_list[-3:]):
                        scale = 1 if quantity is 'strain' else 1e-3
                        axes = axs[row+2, k]
                        axes.plot(
                            simFiber.traces_rate[stim]['time'],
                            simFiber.traces_rate[stim][quantity] * scale,
                            ls=ls, color=color,
                            label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(r'Surface velocity (mm/s)')
    axs[1, 0].set_ylabel(r'Surface pressure rate (kPa/s)')
    axs[2, 0].set_ylabel(r'Internal stress rate (kPa/s)')
    axs[3, 0].set_ylabel(r'Internal strain rate (s$^{-1}$)')
    axs[4, 0].set_ylabel(r'Internal SED rate (kPa$\cdot m^3$/s)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        if axes_id // 2 in [0]:
            xloc = -.135
        elif axes_id // 2 in [1, 2]:
            xloc = -0.105
        elif axes_id // 2 in [3, 4]:
            xloc = -.12
        axes.text(xloc, 1.1, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
        axes.set_xlim(-.0, MAX_RATE_TIME)
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list) * (len(level_plot_list) // 2) +
                len(stim_plot_list) // 2::len(stim_plot_list) * len(
                level_plot_list)],
        [factor_display[5:].capitalize()
         for factor_display in factor_display_list[:3]],
        loc=1)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=1)
    # Add subtitles
    axs[0, 0].set_title('Deflection controlled')
    axs[0, 1].set_title('Pressure controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/temporal_rate_distribution.png', dpi=300)
    plt.close(fig)
    # %% The function to calculate the supra-threshold sensitivity

    def get_sensitivity_function(x, y, supra_threshold=False):
        if y.any():
            start_index = 0
            if supra_threshold:
                start_index = y.nonzero()[0][0]
            slope = np.polyfit(
                x[start_index:], y[start_index:], 1)[0]
        else:
            slope = 0
        return slope
    # %% Calculate all the IQRs and compare force vs. displ
    fiber_id = FIBER_MECH_ID

    def get_slope_iqr(simFiberLevelList, quantity):
        """
        Returns
        -------
        slope_iqr : list
            The 1st element for displ., 2nd for force.
        """
        slope_list_displ = [get_sensitivity_function(
            simFiber.static_displ_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1],
            supra_threshold=False)
            for simFiber in simFiberLevelList]
        slope_list_force = [get_sensitivity_function(
            simFiber.static_force_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1],
            supra_threshold=False)
            for simFiber in simFiberLevelList]
        slope_iqr = []
        slope_iqr.append(np.abs(
            (slope_list_displ[3] - slope_list_displ[1]) / slope_list_displ[2]))
        slope_iqr.append(np.abs(
            (slope_list_force[3] - slope_list_force[1]) / slope_list_force[2]))
        return slope_iqr
    sim_table = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, quantity in enumerate(quantity_list[-3:]):
            simFiberLevelList = [simFiberList[i][level][0] for level in
                                 range(level_num)]
            slope_iqr = get_slope_iqr(simFiberLevelList, quantity)
            sim_table[k, i] = slope_iqr[0]
            sim_table[3+k, i] = slope_iqr[1]
    sim_table_sum = sim_table.sum(axis=1)
    np.savetxt('./csvs/sim_table.csv', sim_table, delimiter=',')
    # %% Calculate all the IQRs and compare force vs. displ rate
    fiber_id = FIBER_MECH_ID

    def get_slope_iqr_rate(simFiberLevelList, quantity):
        """
        Returns
        -------
        slope_iqr : list
            The 1st element for displ., 2nd for force.
        """

        slope_list_displ = [get_sensitivity_function(
            simFiber.displ_rate_exp,
            simFiber.predicted_fr[fiber_id][quantity][:, 2],
            supra_threshold=False)
            for simFiber in simFiberLevelList]
        slope_list_force = [get_sensitivity_function(
            simFiber.force_rate_exp,
            simFiber.predicted_fr[fiber_id][quantity][:, 2],
            supra_threshold=False)
            for simFiber in simFiberLevelList]
        slope_iqr = []
        slope_iqr.append(np.abs(
            (slope_list_displ[3] - slope_list_displ[1]) / slope_list_displ[2]))
        slope_iqr.append(np.abs(
            (slope_list_force[3] - slope_list_force[1]) / slope_list_force[2]))
        return slope_iqr
    sim_table_rate = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, quantity in enumerate(quantity_list[-3:]):
            simFiberLevelList = [simFiberList[i][level][0] for level in
                                 range(level_num)]
            slope_iqr = get_slope_iqr_rate(simFiberLevelList, quantity)
            sim_table_rate[k, i] = slope_iqr[0]
            sim_table_rate[3+k, i] = slope_iqr[1]
    sim_table_rate_sum = sim_table_rate.sum(axis=1)
    np.savetxt('./csvs/sim_table_rate.csv', sim_table_rate, delimiter=',')
    # %% Calculate all the IQRs and compare force vs. displ geometry
    fiber_id = FIBER_MECH_ID
    stim = stim_in_geom_plot

    def get_dr_iqr_geometry(simFiberLevelListDispl, simFiberLevelListForce,
                            quantity):
        """
        Returns
        -------
        dr_iqr : list
            The 1st element for displ., 2nd for force.
        """

        def get_dr(fr_coarse, x_coarse, res=0.1):
            x_fine = np.arange(0, 1, res)
            fr_fine = np.interp(x_fine, x_coarse, fr_coarse)
            max_index = fr_fine.argmax()
            dr = fr_fine[max_index] - fr_fine[max_index - 1] +\
                fr_fine[max_index] - fr_fine[max_index + 1]
            return dr

        dr_displ_list = [
            get_dr(simFiber.dist_fr[fiber_id][quantity][stim, 0, :],
                   simFiber.dist[-1]['mxold'][0] * 1e-3)
            for simFiber in simFiberLevelListDispl]
        dr_force_list = [
            get_dr(simFiber.dist_fr[fiber_id][quantity][stim, 0, :],
                   simFiber.dist[-1]['mxold'][0] * 1e-3)
            for simFiber in simFiberLevelListForce]
        dr_iqr = []
        dr_iqr.append(np.abs(
            (dr_displ_list[3] - dr_displ_list[1]) / dr_displ_list[2]))
        dr_iqr.append(np.abs(
            (dr_force_list[3] - dr_force_list[1]) / dr_force_list[2]))
        return dr_iqr
    sim_table_geometry = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, quantity in enumerate(quantity_list[-3:]):
            simFiberLevelListDispl = [
                simFiberList[i][level][0] for level in range(level_num)]
            simFiberLevelListForce = [
                simFiberList[i][level][1] for level in range(level_num)]
            dr_iqr = get_dr_iqr_geometry(
                simFiberLevelListDispl, simFiberLevelListForce, quantity)
            sim_table_geometry[k, i] = dr_iqr[0]
            sim_table_geometry[3+k, i] = dr_iqr[1]
    sim_table_geometry_sum = sim_table_geometry.sum(axis=1)
    np.savetxt('./csvs/sim_table_geometry.csv', sim_table_geometry,
               delimiter=',')
    # %% Start plotting
    # Factors explaining the force-alignment - static
    for fiber_id in FIBER_FIT_ID_LIST:
        fig, axs = plt.subplots(3, 3, figsize=(6.83, 6))
        for i, factor in enumerate(factor_list[:3]):
            for k, quantity in enumerate(quantity_list[-3:]):
                # for level in level_plot_list:
                for level in range(level_num):
                    alpha = 1. - .4 * abs(level-2)
                    color = (0, 0, 0, alpha)
                    fmt = LS_LIST[i]
                    label = quantile_label_list[level]
                    simFiber = simFiberList[i][level][0]
                    axs[0, k].plot(
                        simFiber.static_displ_exp,
                        simFiber.static_force_exp,
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    axs[1, k].plot(
                        simFiber.static_displ_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[1],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    axs[2, k].plot(
                        simFiber.static_force_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[1],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
        # X and Y limits
        for axes in axs[0:, :].ravel():
            axes.set_ylim(0, 15)
        for axes in axs[1:, :].ravel():
            axes.set_ylim(0, 50)
        for axes in axs[:2, :].ravel():
            axes.set_xlim(.35, .75)
            axes.set_xticks(np.arange(.35, .85, .1))
        for axes in axs[2, :].ravel():
            axes.set_xlim(0, 15)
        # Axes and panel labels
        for i, axes in enumerate(axs[0, :].ravel()):
            axes.set_title('%s-based Model' % ['Stress', 'Strain', 'SED'][i])
        for axes in axs[:2, :].ravel():
            axes.set_xlabel(r'Static displacement (mm)')
        for axes in axs[2, :].ravel():
            axes.set_xlabel('Static force (mN)')
        for axes in axs[0, :1].ravel():
            axes.set_ylabel('Static force (mN)')
        for axes in axs[1:, 0].ravel():
            axes.set_ylabel('Predicted mean firing (Hz)')
        for axes_id, axes in enumerate(axs.ravel()):
            axes.text(-.175, 1.1, chr(65+axes_id), transform=axes.transAxes,
                      fontsize=12, fontweight='bold', va='top')
        # Legend
        # The line type labels
        handles, labels = axs[0, 0].get_legend_handles_labels()
        axs[0, 0].legend(handles[2::5], ['Thickness', 'Modulus', 'Visco.'],
                         loc=2)
        # The 5 quantile labels
        axs[0, 1].legend(handles[:3], ['Extreme', 'Quartile',
                         'Median'], loc=2)
        # Save
        fig.tight_layout()
        fig.savefig('./plots/sim_compare_variance_%d.png' % fiber_id, dpi=300)
        fig.savefig('./plots/sim_compare_variance_%d.pdf' % fiber_id, dpi=300)
        plt.close(fig)
    # %% Plot the other two fibers, stress&force, strain&displ
    fig, axs = plt.subplots(2, 3, figsize=(6.83, 4))
    for i, factor in enumerate(factor_list[:3]):
        for level in range(level_num):
            alpha = 1. - .4 * abs(level-2)
            color = (0, 0, 0, alpha)
            fmt = LS_LIST[i]
            label = quantile_label_list[level]
            axs[0, 0].plot(
                simFiberList[i][level][0].static_displ_exp,
                simFiberList[i][level][0].predicted_fr[2]['strain'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
            axs[1, 0].plot(
                simFiberList[i][level][0].static_force_exp,
                simFiberList[i][level][0].predicted_fr[2]['stress'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
            axs[0, 1].plot(
                simFiberList[i][level][0].static_displ_exp,
                simFiberList[i][level][0].predicted_fr[0]['strain'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
            axs[1, 1].plot(
                simFiberList[i][level][0].static_force_exp,
                simFiberList[i][level][0].predicted_fr[0]['stress'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
            axs[0, 2].plot(
                simFiberList[i][level][0].static_displ_exp,
                simFiberList[i][level][0].predicted_fr[1]['strain'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
            axs[1, 2].plot(
                simFiberList[i][level][0].static_force_exp,
                simFiberList[i][level][0].predicted_fr[1]['stress'].T[1],
                color=color, mec=color, ms=MS, ls=fmt, label=label)
    # Formatting
    for axes in axs[0, :].ravel():
        axes.set_xlabel(r'Static displacement (mm)')
    for axes in axs[1, :].ravel():
        axes.set_xlabel(r'Static force (mN)')
        axes.set_xlim(0, 15)
    for row, axes in enumerate(axs[:, 0].ravel()):
        axes.set_ylabel('Predicted mean firing (Hz)\n%s-based model' %
                        ['Strain', 'Stress'][row])
    for axes in axs.ravel():
        axes.set_ylim(0, 50)
    for i, axes in enumerate(axs[0, :].ravel()):
        axes.set_title('Fiber #%d' % (i+1))
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.175, 1.1, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Legend
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[1, 1].legend(handles[2::5], ['Thickness', 'Modulus', 'Visco.'],
                     loc=4)
    # The 5 quantile labels
    axs[1, 2].legend(handles[:3], ['Extreme', 'Quartile',
                     'Median'], loc=4)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/sim_compare_variance_other_fibers.png', dpi=300)
    fig.savefig('./plots/sim_compare_variance_other_fibers.pdf', dpi=300)
    plt.close(fig)
    # %% Plot the other two groupings, stress&force, strain&displ
    fig, axs = plt.subplots(2, 3, figsize=(6.83, 4))
    grouping_elif_list = [[8, 5, 3, 1], [10, 5, 1, 1], [6, 5, 3, 3]]
    base_grouping = grouping_elif_list[0]

    def get_fr_from_grouping(simFiber, grouping, quantity,
                             fiber_id=FIBER_MECH_ID,
                             base_grouping=base_grouping):
        trans_params_fit = simFiber.trans_params
        dr = grouping[0] / base_grouping[0]
        trans_params = copy.deepcopy(trans_params_fit)
        trans_params[fiber_id][quantity][0] *= dr
        trans_params[fiber_id][quantity][1] *= dr
        predicted_fr = simFiber.get_predicted_fr(trans_params=trans_params)
        static_fr = predicted_fr[fiber_id][quantity][:, 1]
        return static_fr
    for i, factor in enumerate(factor_list[:3]):
        for level in range(level_num):
            alpha = 1. - .4 * abs(level-2)
            color = (0, 0, 0, alpha)
            fmt = LS_LIST[i]
            label = quantile_label_list[level]
            simFiber = simFiberList[i][level][0]
            for col, grouping in enumerate(grouping_elif_list):
                for row, quantity in enumerate(['strain', 'stress']):
                    response = get_fr_from_grouping(
                        simFiber, grouping, quantity)
                    if row == 0:
                        stimuli = simFiber.static_displ_exp
                    elif row == 1:
                        stimuli = simFiber.static_force_exp
                    axs[row, col].plot(stimuli, response, color=color,
                                       mec=color,
                                       ms=MS, ls=fmt, label=label)
    # Formatting
    for axes in axs[0, :].ravel():
        axes.set_xlabel(r'Static displacement (mm)')
    for axes in axs[1, :].ravel():
        axes.set_xlabel(r'Static force (mN)')
        axes.set_xlim(0, 15)
    for row, axes in enumerate(axs[:, 0].ravel()):
        axes.set_ylabel('Predicted mean firing (Hz)\n%s-based model' %
                        ['Strain', 'Stress'][row])
    for axes in axs.ravel():
        axes.set_ylim(0, 50)
    for i, axes in enumerate(axs[0, :].ravel()):
        axes.set_title('Grouping: %s' % str(grouping_elif_list[i]))
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.175, 1.1, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Legend
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[1, 1].legend(handles[2::5], ['Thickness', 'Modulus', 'Visco.'],
                     loc=4)
    # The 5 quantile labels
    axs[1, 2].legend(handles[:3], ['Extreme', 'Quartile',
                     'Median'], loc=2)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/sim_compare_variance_groupings.png', dpi=300)
    fig.savefig('./plots/sim_compare_variance_groupings.pdf', dpi=300)
    plt.close(fig)
    # %% Calculate values for displacement vs. mcnc displacement
    spatial_y_table = np.empty([3])
    for i, factor in enumerate(factor_list[:3]):
        j = 0
        control = 'Displ'
        quantity = 'y'
        iqr = np.abs(simFiberList[i][3][j].dist[
            2]['m%sint' % quantity] - simFiberList[i][1][j].dist[
            2]['m%sint' % quantity])
        distance = .5 * np.abs(simFiberList[i][2][j].dist[
            3]['m%sint' % quantity] - simFiberList[i][2][j].dist[
            1]['m%sint' % quantity])
        spatial_y_table[i] = iqr / distance
    spatial_y_table_sum = spatial_y_table.sum()
    # Calculate Pearson correlation coefficients
    dist = simFiberList[0][2][0].dist[3]
    # Calculate shape R^2
    xcoord = np.linspace(0, MAX_RADIUS, 100)
    # Get surface data
    surface_quantity = 'cy'
    surface_data = np.interp(xcoord, dist['cxold'][-1],
                             dist[surface_quantity][-1])
    # Get mcnc data
    mcnc_data = np.interp(xcoord, dist['mxold'][-1],
                          dist['my'][-1])
    # Calculate correlation
    spatial_y_pearsonr, spatial_y_pearsonp = pearsonr(surface_data, mcnc_data)
    # %% Plot the displacement at MCNC
    fig, axs = plt.subplots(2, 3, figsize=(7, 3.5))
    for i, factor in enumerate(factor_list[:3]):
        for level in level_plot_list:
            for stim in stim_plot_list:
                alpha = 1. - .65 * abs(level - 2)
                if stim == 2:
                    color = (0, 0, 0, alpha)
                elif stim == 1:
                    color = (1, 0, 0, alpha)
                elif stim == 3:
                    color = (0, 0, 1, alpha)
                ls = LS_LIST[i]
                kwargs = dict(ls=ls, color=color,
                              label=quantile_label_list[level])
                # First column, Stimulus magnitude over time
                dist = simFiberList[i][level][0].dist[stim]
                axs[0, 0].plot(
                    dist['time'][:, 0],
                    dist['cy'][:, 0] * 1e-3,
                    **kwargs)
                axs[1, 0].plot(
                    dist['time'][:, 0],
                    dist['my'][:, 0] * 1e-3,
                    **kwargs)
                # Second column, Stimulus rate over time
                axs[0, 1].plot(
                    dist['time'][:, 0],
                    np.r_[0, np.diff(dist['cy'][:, 0]) /
                          np.diff(dist['time'][:, 0])] * 1e-3,
                    **kwargs)
                axs[1, 1].plot(
                    dist['time'][:, 0],
                    np.r_[0, np.diff(dist['my'][:, 0]) /
                          np.diff(dist['time'][:, 0])] * 1e-3,
                    **kwargs)
                # Third column, Stimulus magnitude over space
                xscale = 1e3
                axs[0, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cy'][-1, :] * 1e-3,
                    **kwargs)
                axs[1, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['my'][-1, :] * 1e-3,
                    **kwargs)
    # Set x and y lim
    for axes in axs[:, 0].ravel():
        axes.set_xlim(0, MAX_TIME)
    for axes in axs[:, 1].ravel():
        axes.set_xlim(0, MAX_RATE_TIME)
        axes.set_ylim(0, 1.8)
    for axes in axs[:, 2].ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    # x-axis
    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Time (s)')
    axs[-1, 2].set_xlabel('Location (mm)')
    # y-axis for the Stimulus magnitude over time
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel('Internal displacement (mm)')
    # y-axis for the Stimulus rate over time
    axs[0, 1].set_ylabel(r'Surface velocity (mm/s)')
    axs[1, 1].set_ylabel(r'Internal velocity (mm/s)')
    # y-axis for the Stimulus magnitude over space
    axs[0, 2].set_ylabel(r'Surface deflection (mm)')
    axs[1, 2].set_ylabel('Internal displacement (mm)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.34, 1.13, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(level_plot_list)//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                level_plot_list)],
        [factor_display[5:].capitalize()
         for factor_display in factor_display_list[:3]], loc=4, fontsize=6)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=1, fontsize=6)
    # Add subtitles
    axs[0, 0].set_title('Stimulus magnitude over time', fontsize=8)
    axs[0, 1].set_title('Stimulus rate over time', fontsize=8)
    axs[0, 2].set_title('Stimulus magnitude over space', fontsize=8)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/paper_internal_disp.png', dpi=300)
    fig.savefig('./plots/paper_internal_disp.pdf', dpi=300)
    plt.close(fig)
    # %% The huge mech table for JN paper
    jn_mech_table = np.empty((4, 15))
    # Fill iqr ratio data
    jn_mech_table[0, 0:3] = temporal_table[1]
    jn_mech_table[0, 5:8] = temporal_rate_table[1]
    jn_mech_table[0, 10:13] = spatial_table[1]
    jn_mech_table[1, 0:3] = temporal_table[2]
    jn_mech_table[1, 5:8] = temporal_rate_table[2]
    jn_mech_table[1, 10:13] = spatial_table[2]
    jn_mech_table[2, 0:3] = temporal_table[3]
    jn_mech_table[2, 5:8] = temporal_rate_table[3]
    jn_mech_table[2, 10:13] = spatial_table[3]
    jn_mech_table[3, 0:3] = temporal_table[5]
    jn_mech_table[3, 5:8] = temporal_rate_table[5]
    jn_mech_table[3, 10:13] = spatial_table[5]
    # Sum the iqr ratios
    for i in range(3, 14, 5):
        jn_mech_table[:, i] = jn_mech_table[:, i-3:i].sum(axis=1)
    # Fill r2 data
    jn_mech_table[0, 4] = temporal_pearsonr_table[1, 0] ** 2
    jn_mech_table[1, 4] = temporal_pearsonr_table[2, 0] ** 2
    jn_mech_table[2, 4] = temporal_pearsonr_table[0, 1] ** 2
    jn_mech_table[3, 4] = temporal_pearsonr_table[2, 1] ** 2
    jn_mech_table[0, 9] = temporal_rate_pearsonr_table[1, 0] ** 2
    jn_mech_table[1, 9] = temporal_rate_pearsonr_table[2, 0] ** 2
    jn_mech_table[2, 9] = temporal_rate_pearsonr_table[0, 1] ** 2
    jn_mech_table[3, 9] = temporal_rate_pearsonr_table[2, 1] ** 2
    jn_mech_table[0, 14] = spatial_pearsonr_table[1, 0] ** 2
    jn_mech_table[1, 14] = spatial_pearsonr_table[2, 0] ** 2
    jn_mech_table[2, 14] = spatial_pearsonr_table[0, 1] ** 2
    jn_mech_table[3, 14] = spatial_pearsonr_table[2, 1] ** 2
    # Convert to pandas dataframe before I forget
    columns = []
    [columns.extend(['T', 'M', 'V', 'Sum', 'R2'])
        for i in range(3)]
    index = ['Strain', 'SED', 'Stress', 'SED']
    jn_mech_table_df = pd.DataFrame(data=jn_mech_table,
                                    columns=columns, index=index)
    jn_mech_table_df.to_csv('./csvs/jn_mech_table.csv')
    # %% The table for simulations
    jn_sim_table = np.empty((2, 12))
    for control_id in range(2):
        for quantity_id in range(3):
            jn_sim_table[control_id, quantity_id * 4:quantity_id * 4 + 3] =\
                sim_table[3 * control_id + quantity_id]
            jn_sim_table[control_id, quantity_id * 4 + 3] = sim_table[
                3 * control_id + quantity_id].sum()
    columns = []
    [columns.extend(['T', 'M', 'V', 'Sum']) for i in range(3)]
    index = ['Displacement', 'Force']
    df_jn_sim_table = pd.DataFrame(jn_sim_table,
                                   columns=columns, index=index)
    df_jn_sim_table.to_csv('./csvs/jn_sim_table.csv')
    # %% The huge simulation figure in JN paper
    fig, axs = plt.subplots(6, 3, figsize=(7, 8.75))
    for i, factor in enumerate(factor_list[:3]):
        for level in level_plot_list:
            for stim in stim_plot_list:
                alpha = 1. - .65 * abs(level - 2)
                if stim == 2:
                    color = (0, 0, 0, alpha)
                elif stim == 1:
                    color = (1, 0, 0, alpha)
                elif stim == 3:
                    color = (0, 0, 1, alpha)
                ls = LS_LIST[i]
                kwargs = dict(ls=ls, color=color,
                              label=quantile_label_list[level])
                # First column, Stimulus magnitude over time
                simFiber = simFiberList[i][level][0]
                axs[0, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['strain'],
                    **kwargs)
                axs[2, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['sener'] * 1e-3,
                    **kwargs)
                simFiber = simFiberList[i][level][1]
                axs[3, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['press'] * 1e-3,
                    **kwargs)
                axs[4, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['stress'] * 1e-3,
                    **kwargs)
                axs[5, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['sener'] * 1e-3,
                    **kwargs)
                # Second column, Stimulus rate over time
                simFiber = simFiberList[i][level][0]
                axs[0, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['strain'],
                    **kwargs)
                axs[2, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['sener'] * 1e-3,
                    **kwargs)
                simFiber = simFiberList[i][level][1]
                axs[3, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['press'] * 1e-3,
                    **kwargs)
                axs[4, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['stress'] * 1e-3,
                    **kwargs)
                axs[5, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['sener'] * 1e-3,
                    **kwargs)
                # Third column, Stimulus magnitude over space
                xscale = 1e3
                dist = simFiberList[i][level][0].dist[stim]
                axs[0, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cy'][-1, :] * 1e-3,
                    **kwargs)
                axs[1, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['mstrain'][-1, :],
                    **kwargs)
                axs[2, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['msener'][-1, :] * 1e-3,
                    **kwargs)
                dist = simFiberList[i][level][1].dist[stim]
                axs[3, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cpress'][-1, :] * 1e-3,
                    **kwargs)
                axs[4, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['mstress'][-1, :] * 1e-3,
                    **kwargs)
                axs[5, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['msener'][-1, :] * 1e-3,
                    **kwargs)
    # Set x and y lim
    for axes in axs[:, 0].ravel():
        axes.set_xlim(0, MAX_TIME)
    for axes in axs[:, 1].ravel():
        axes.set_xlim(0, MAX_RATE_TIME)
    for axes in axs[:, 2].ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    # x-axis
    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Time (s)')
    axs[-1, 2].set_xlabel('Location (mm)')
    # y-axis for the Stimulus magnitude over time
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel('Internal strain')
    axs[2, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    axs[3, 0].set_ylabel(r'Surface pressure (kPa)')
    axs[4, 0].set_ylabel('Internal stress (kPa)')
    axs[5, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # y-axis for the Stimulus rate over time
    axs[0, 1].set_ylabel(r'Surface velocity (mm/s)')
    axs[1, 1].set_ylabel(r'Internal strain rate (s$^{-1}$)')
    axs[2, 1].set_ylabel(r'Internal SED rate (kPa$\cdot m^3$/s)')
    axs[3, 1].set_ylabel(r'Surface pressure rate (kPa/s)')
    axs[4, 1].set_ylabel(r'Internal stress rate (kPa/s)')
    axs[5, 1].set_ylabel(r'Internal SED rate (kPa$\cdot m^3$/s)')
    # y-axis for the Stimulus magnitude over space
    axs[0, 2].set_ylabel(r'Surface deflection (mm)')
    axs[1, 2].set_ylabel('Internal strain')
    axs[2, 2].set_ylabel(r'Internal SED (kPa/$m^3$)')
    axs[3, 2].set_ylabel(r'Surface pressure (kPa)')
    axs[4, 2].set_ylabel('Internal stress (kPa)')
    axs[5, 2].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.375, 1.13, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].set_ylim(.225, .625)
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(level_plot_list)//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                level_plot_list)],
        [factor_display[5:].capitalize()
         for factor_display in factor_display_list[:3]], loc=4, fontsize=6)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=1, fontsize=6)
    # Add subtitles
    axs[0, 0].set_title('Stimulus magnitude over time', fontsize=8)
    axs[0, 1].set_title('Stimulus rate over time', fontsize=8)
    axs[0, 2].set_title('Stimulus magnitude over space', fontsize=8)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/paper_simulation.png', dpi=300)
    fig.savefig('./plots/paper_simulation.pdf', dpi=300)
    # Add x labels to all for presentation use
    for row in axs:
        for col, axes in enumerate(row):
            xlabel = ['Time (s)', 'Time (s)', 'Location (mm)'][col]
            axes.set_xlabel(xlabel)
    axs[0, 0].set_ylim(.2, .625)
    fig.savefig('./plots/paper_simulation_prez.png', dpi=300)
    plt.close(fig)
    # %% The figure for substrate simulations
    fig, axs = plt.subplots(6, 3, figsize=(7, 9.19))
    for i, factor in enumerate(factor_list[-2:]):
        i = i + 3
        for level in range(level_num):
            for stim in stim_plot_list:
                alpha = 1. - .3 * abs(level - 2)
                if stim == 2:
                    color = (0, 0, 0, alpha)
                elif stim == 1:
                    color = (1, 0, 0, alpha)
                elif stim == 3:
                    color = (0, 0, 1, alpha)
                ls = LS_LIST[i - 3]
                kwargs = dict(ls=ls, color=color,
                              label=quantile_label_list[level])
                # First column, Stimulus magnitude over time
                simFiber = simFiberList[i][level][0]
                axs[0, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['strain'],
                    **kwargs)
                axs[2, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['sener'] * 1e-3,
                    **kwargs)
                simFiber = simFiberList[i][level][1]
                axs[3, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['press'] * 1e-3,
                    **kwargs)
                axs[4, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['stress'] * 1e-3,
                    **kwargs)
                axs[5, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['sener'] * 1e-3,
                    **kwargs)
                # Second column, Stimulus rate over time
                simFiber = simFiberList[i][level][0]
                axs[0, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['strain'],
                    **kwargs)
                axs[2, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['sener'] * 1e-3,
                    **kwargs)
                simFiber = simFiberList[i][level][1]
                axs[3, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['press'] * 1e-3,
                    **kwargs)
                axs[4, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['stress'] * 1e-3,
                    **kwargs)
                axs[5, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['sener'] * 1e-3,
                    **kwargs)
                # Third column, Stimulus magnitude over space
                xscale = 1e3
                dist = simFiberList[i][level][0].dist[stim]
                axs[0, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cy'][-1, :] * 1e-3,
                    **kwargs)
                axs[1, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['mstrain'][-1, :],
                    **kwargs)
                axs[2, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['msener'][-1, :] * 1e-3,
                    **kwargs)
                dist = simFiberList[i][level][1].dist[stim]
                axs[3, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cpress'][-1, :] * 1e-3,
                    **kwargs)
                axs[4, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['mstress'][-1, :] * 1e-3,
                    **kwargs)
                axs[5, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['msener'][-1, :] * 1e-3,
                    **kwargs)
    # Set x and y lim
    for axes in axs[:, 0].ravel():
        axes.set_xlim(0, MAX_TIME)
    for axes in axs[:, 1].ravel():
        axes.set_xlim(0, MAX_RATE_TIME)
    for axes in axs[:, 2].ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    # x-axis
    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Time (s)')
    axs[-1, 2].set_xlabel('Location (mm)')
    # y-axis for the Stimulus magnitude over time
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel('Internal strain')
    axs[2, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    axs[3, 0].set_ylabel(r'Surface pressure (kPa)')
    axs[4, 0].set_ylabel('Internal stress (kPa)')
    axs[5, 0].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # y-axis for the Stimulus rate over time
    axs[0, 1].set_ylabel(r'Surface velocity (mm/s)')
    axs[1, 1].set_ylabel(r'Internal strain rate (s$^{-1}$)')
    axs[2, 1].set_ylabel(r'Internal SED rate (kPa$\cdot m^3$/s)')
    axs[3, 1].set_ylabel(r'Surface pressure rate (kPa/s)')
    axs[4, 1].set_ylabel(r'Internal stress rate (kPa/s)')
    axs[5, 1].set_ylabel(r'Internal SED rate (kPa$\cdot m^3$/s)')
    # y-axis for the Stimulus magnitude over space
    axs[0, 2].set_ylabel(r'Surface deflection (mm)')
    axs[1, 2].set_ylabel('Internal strain')
    axs[2, 2].set_ylabel(r'Internal SED (kPa/$m^3$)')
    axs[3, 2].set_ylabel(r'Surface pressure (kPa)')
    axs[4, 2].set_ylabel('Internal stress (kPa)')
    axs[5, 2].set_ylabel(r'Internal SED (kPa/$m^3$)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.38, 1.13, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    axs[0, 1].set_ylim(0, 0.8)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(range(level_num))//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                range(level_num))],
        [factor_display.capitalize()
         for factor_display in factor_display_list[-2:]], loc=4)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(range(level_num))+1:3], [
        r'100±50%', r'100±25%', r'100%'], loc=1)
    # Add subtitles
    axs[0, 0].set_title('Stimulus magnitude over time', fontsize=8)
    axs[0, 1].set_title('Stimulus rate over time', fontsize=8)
    axs[0, 2].set_title('Stimulus magnitude over space', fontsize=8)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/paper_substrate_full.png', dpi=300)
    fig.savefig('./plots/paper_substrate_full.pdf', dpi=300)
    plt.close(fig)
    # %% The figure for substrate simulations
    fig, axs = plt.subplots(2, 3, figsize=(7, 3.5))
    for i, factor in enumerate(factor_list[-2:]):
        i = i + 3
        for level in range(level_num):
            for stim in stim_plot_list:
                alpha = 1. - .3 * abs(level - 2)
                if stim == 2:
                    color = (0, 0, 0, alpha)
                elif stim == 1:
                    color = (1, 0, 0, alpha)
                elif stim == 3:
                    color = (0, 0, 1, alpha)
                ls = LS_LIST[i - 3]
                kwargs = dict(ls=ls, color=color,
                              label=quantile_label_list[level])
                # First column, Stimulus magnitude over time
                simFiber = simFiberList[i][level][0]
                axs[0, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 0].plot(
                    simFiber.traces[stim]['time'],
                    simFiber.traces[stim]['strain'],
                    **kwargs)
                # Second column, Stimulus rate over time
                simFiber = simFiberList[i][level][0]
                axs[0, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['displ'] * 1e3,
                    **kwargs)
                axs[1, 1].plot(
                    simFiber.traces_rate[stim]['time'],
                    simFiber.traces_rate[stim]['strain'],
                    **kwargs)
                # Third column, Stimulus magnitude over space
                xscale = 1e3
                dist = simFiberList[i][level][0].dist[stim]
                axs[0, 2].plot(
                    dist['cxold'][-1, :] * xscale,
                    dist['cy'][-1, :] * 1e-3,
                    **kwargs)
                axs[1, 2].plot(
                    dist['mxold'][-1, :] * xscale,
                    dist['mstrain'][-1, :],
                    **kwargs)
    # Set x and y lim
    for axes in axs[:, 0].ravel():
        axes.set_xlim(0, MAX_TIME)
    for axes in axs[:, 1].ravel():
        axes.set_xlim(0, MAX_RATE_TIME)
    for axes in axs[:, 2].ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    # x-axis
    axs[-1, 0].set_xlabel('Time (s)')
    axs[-1, 1].set_xlabel('Time (s)')
    axs[-1, 2].set_xlabel('Location (mm)')
    # y-axis for the Stimulus magnitude over time
    axs[0, 0].set_ylabel(r'Surface deflection (mm)')
    axs[1, 0].set_ylabel('Internal strain')
    # y-axis for the Stimulus rate over time
    axs[0, 1].set_ylabel(r'Surface velocity (mm/s)')
    axs[1, 1].set_ylabel(r'Internal strain rate (s$^{-1}$)')
    # y-axis for the Stimulus magnitude over space
    axs[0, 2].set_ylabel(r'Surface deflection (mm)')
    axs[1, 2].set_ylabel('Internal strain')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.3, 1.15, chr(65+axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    axs[0, 1].set_ylim(0, 0.8)
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(
        handles[len(stim_plot_list)*(len(range(level_num))//2) +
                len(stim_plot_list)//2::len(stim_plot_list)*len(
                range(level_num))],
        [factor_display.capitalize()
         for factor_display in factor_display_list[-2:]], loc=4)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(range(level_num))+1:3], [
        r'100±50%', r'100±25%', r'100%'], loc=1)
    # Add subtitles
    axs[0, 0].set_title('Stimulus magnitude over time', fontsize=8)
    axs[0, 1].set_title('Stimulus rate over time', fontsize=8)
    axs[0, 2].set_title('Stimulus magnitude over space', fontsize=8)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/paper_substrate_short.png', dpi=300)
    fig.savefig('./plots/paper_substrate_short.pdf', dpi=300)
    plt.close(fig)
    # %% The displ - force part of the encoding plot
    fiber_id = FIBER_MECH_ID
    fig, axs = plt.subplots()
    for i, factor in enumerate(factor_list[:3]):
        for level in range(level_num):
            alpha = 1. - .4 * abs(level-2)
            color = (0, 0, 0, alpha)
            fmt = LS_LIST[i]
            label = quantile_label_list[level]
            simFiber = simFiberList[i][level][0]
            axs.plot(
                simFiber.static_displ_exp,
                simFiber.static_force_exp,
                color=color, mec=color, ms=MS,
                ls=fmt, label=label)
    # X and Y limits
    axs.set_ylim(0, 15)
    axs.set_xlim(.3, .8)
    # Axes and panel labels
    axs.set_xlabel(r'Static displacement (mm)')
    axs.set_xlabel('Static force (mN)')
    # Legend
    handles, labels = axs.get_legend_handles_labels()
    axs.legend(handles[2::5] + handles[:3],
               ['Thickness', 'Modulus', 'Viscoelasticity'] + [
                   'Extreme', 'Quartile', 'Median'],
               loc=2)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/encoding_skin.png', dpi=300)
    fig.savefig('./plots/encoding_skin.pdf', dpi=300)
    plt.close(fig)
    # %% Plot the encoding plot with all three features
    stim = stim_in_geom_plot
    for fiber_id in [FIBER_MECH_ID]:
        fig, axs = plt.subplots(6, 3, figsize=(7, 9.19))
        for i, factor in enumerate(factor_list[:3]):
            for k, quantity in enumerate(quantity_list[-3:]):
                # for level in level_plot_list:
                for level in range(level_num):
                    alpha = 1. - .4 * abs(level-2)
                    color = (0, 0, 0, alpha)
                    fmt = LS_LIST[i]
                    label = quantile_label_list[level]
                    simFiber = simFiberList[i][level][0]
                    simFiberForce = simFiberList[i][level][1]
                    # Static
                    axs[0, k].plot(
                        simFiber.static_displ_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[1],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    axs[1, k].plot(
                        simFiber.static_force_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[1],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    # Dynamic
                    axs[2, k].plot(
                        simFiber.displ_rate_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[2],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    axs[3, k].plot(
                        simFiber.force_rate_exp,
                        simFiber.predicted_fr[fiber_id][quantity].T[2],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    # Spatial distribution
                    axs[4, k].plot(
                        simFiber.dist[stim]['mxold'][0] * 1e3,
                        simFiber.dist_fr[fiber_id][quantity][stim, 0, :],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
                    axs[5, k].plot(
                        simFiberForce.dist[stim]['mxold'][0] * 1e3,
                        simFiberForce.dist_fr[fiber_id][quantity][stim, 0, :],
                        color=color, mec=color, ms=MS,
                        ls=fmt, label=label)
        # X and Y limits
        for axes in axs[0:2, :].ravel():
            axes.set_ylim(0, 50)
        for axes in axs[2:4].ravel():
            axes.set_ylim(0, 150)
        for axes in axs[4:6].ravel():
            axes.set_ylim(0, 80)
        for axes in axs[0]:
            axes.set_xlim(.3, .8)
        for axes in axs[1]:
            axes.set_xlim(0, 15)
        for axes in axs[3]:
            axes.set_xlim(0, 45)
        for axes in axs[4:6, :].ravel():
            axes.set_xlim(0., MAX_RADIUS * 1e3)
        # Axes and panel labels
        for i, axes in enumerate(axs[0, :].ravel()):
            axes.set_title('%s-based Model' % ['Stress', 'Strain', 'SED'][i])
        for axes in axs[4, :]:
            axes.set_title('Tip displacement = %.2f mm' %
                           simFiber.static_displ_exp[stim])
        for axes in axs[5, :]:
            axes.set_title('Tip force = %.2f mN' %
                           simFiberForce.static_force_exp[stim])
        for axes in axs[0]:
            axes.set_xlabel(r'Static displacement (mm)')
        for axes in axs[1]:
            axes.set_xlabel(r'Static force (mN)')
        for axes in axs[2]:
            axes.set_xlabel(r'Mean velocity (mm/s)')
        for axes in axs[3]:
            axes.set_xlabel(r'Mean force rate (mN/s)')
        for axes in axs[4:6, :].ravel():
            axes.set_xlabel('Location (mm)')
        for axes in axs[0:2, 0].ravel():
            axes.set_ylabel('Predicted static firing (Hz)')
        for axes in axs[2:4, 0].ravel():
            axes.set_ylabel('Predicted dynamic firing (Hz)')
        for axes in axs[4:6, 0].ravel():
            axes.set_ylabel('Predicted static firing (Hz)')
        for axes_id, axes in enumerate(axs.ravel()):
            axes.text(-.11, 1.25, chr(65+axes_id), transform=axes.transAxes,
                      fontsize=12, fontweight='bold', va='top')
        # Legend
        # The line type labels
        handles, labels = axs[0, 0].get_legend_handles_labels()
        axs[0, 1].legend(handles[2::5], ['Thickness', 'Modulus', 'Visco.'],
                         loc=2, fontsize=6)
        # The 5 quantile labels
        axs[0, 2].legend(handles[:3], ['Extreme', 'Quartile',
                         'Median'], loc=2, fontsize=6)
        # Save
        fig.tight_layout()
        fig.savefig('./plots/encoding_neural.png', dpi=300)
        fig.savefig('./plots/encoding_neural.pdf', dpi=300)
        plt.close(fig)
    # %% Make the table for encoding neural spatial part
    jn_sim_table = np.empty((8, 12))
    for row in range(8):
        control_id = row % 2
        for quantity_id in range(3):
            if row <= 1:
                jn_sim_table[row, quantity_id * 4:quantity_id * 4 + 3] =\
                    sim_table[3 * control_id + quantity_id]
                jn_sim_table[row, quantity_id * 4 + 3] = sim_table[
                    3 * control_id + quantity_id].sum()
            elif row <= 3:
                jn_sim_table[row, quantity_id * 4:quantity_id * 4 + 3] =\
                    sim_table_rate[3 * control_id + quantity_id]
                jn_sim_table[row, quantity_id * 4 + 3] = sim_table_rate[
                    3 * control_id + quantity_id].sum()
            elif row <= 5:
                jn_sim_table[row, quantity_id * 4:quantity_id * 4 + 3] =\
                    sim_table_geometry[3 * control_id + quantity_id]
                jn_sim_table[row, quantity_id * 4 + 3] = sim_table_geometry[
                    3 * control_id + quantity_id].sum()
            elif row <= 7:
                jn_sim_table[row] = jn_sim_table[row % 2:6:2].sum(axis=0)
    columns = []
    [columns.extend(['T', 'M', 'V', 'Sum']) for i in range(3)]
    index = []
    [index.extend(['Displacement', 'Force']) for i in range(4)]
    df_jn_sim_table = pd.DataFrame(jn_sim_table,
                                   columns=columns, index=index)
    df_jn_sim_table.to_csv('./csvs/jn_sim_table_three_aspects.csv')
