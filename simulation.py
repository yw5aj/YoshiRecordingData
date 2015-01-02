# -*- coding: utf-8 -*-
"""
Created on Sun May  4 22:38:40 2014

@author: Yuxiang Wang
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad
from scipy.stats import pearsonr
import pickle
from constants import (DT, FIBER_TOT_NUM, MARKER_LIST, COLOR_LIST, MS,
    FIBER_MECH_ID, FIBER_FIT_ID_LIST, EVAL_DISPL, EVAL_FORCE, STATIC_START,
    STATIC_END, LS_LIST)
from fitlif import trans_param_to_predicted_fr



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
stim_plot_list = [1, 2, 3] # Stims to be plotted
level_plot_list = range(level_num)[1:-1]

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
        self.get_line_fit()
        return

    def get_dist(self):
        fpath = BASE_CSV_PATH
        self.dist = [{} for i in range(stim_num)]
        key_list = ['cpress', 'cxnew', 'cxold', 'cy', 'msener',
                    'mstrain', 'mstress', 'mxnew', 'mxold', 'my',
                    'time']
        for stim in range(stim_num)[1:]:
            for key in key_list:
                self.dist[stim][key] = np.loadtxt(
                    fpath+self.factor+str(self.level)+str(stim-1)+self.control
                    +'_'+key+'.csv', delimiter=',')
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
                self.dist[stim]['time'][:, np.newaxis], self.dist[stim][
                'cxnew'].shape[1])
            # Calculate integration over area
            for key in key_list:
                if 'x' not in key and 'time' not in key:
                    def get_field(r):
                        return np.interp(r, self.dist[stim][key[0]+'xnew'][
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
        fname_list = [self.factor + str(self.level) + str(stim) +\
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
                self.traces_rate[i+1][quantity] = np.interp(fine_time,
                    time[:-1], np.diff(locals()[quantity])/np.diff(time))
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
            self.traces[i]['displ'] = displcoeff[0] * 1e-6 + displcoeff[1
                ] * self.traces[i]['displ']
        # Get the FEM and corresponding displ / force
        self.static_displ_exp = np.array([self.traces[i]['displ'][-1]
            for i in range(stim_num)]) * 1e3
        self.static_force_fem = np.array([self.traces[i]['force'][-1]
            for i in range(stim_num)])
        self.static_force_exp = self.static_force_fem * 1e3
        return

    def get_predicted_fr(self):
        self.predicted_fr = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                # Get the quantity_dict_list
                quantity_dict_list = [{
                    'quantity_array': self.traces[i][quantity],
                    'max_index': self.traces[i]['max_index']}
                    for i in range(stim_num)]
                self.predicted_fr[fiber_id][quantity] =\
                    trans_param_to_predicted_fr(quantity_dict_list,
                    self.trans_params[fiber_id][quantity])
        return

    def get_line_fit(self):
        self.line_fit = [{} for i in range(FIBER_TOT_NUM)]
        self.line_fit_median_predict = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list[-3:]:
                self.line_fit[fiber_id][quantity] = {
                    'displ_dynamic': np.polyfit(self.static_displ_exp,
                        self.predicted_fr[fiber_id][quantity][:, 2], 1),
                    'displ_static': np.polyfit(self.static_displ_exp,
                        self.predicted_fr[fiber_id][quantity][:, 1], 1),
                    'force_dynamic': np.polyfit(self.static_force_exp,
                        self.predicted_fr[fiber_id][quantity][:, 2], 1),
                    'force_static': np.polyfit(self.static_force_exp,
                        self.predicted_fr[fiber_id][quantity][:, 1], 1),
                    }
                self.line_fit_median_predict[fiber_id][quantity] = {
                    key: np.polyval(self.line_fit[fiber_id][quantity][key]
                        , globals()['EVAL_' + key[:5].upper()])
                        for key in iter(self.line_fit[fiber_id][quantity])
                    }
        return

    def plot_predicted_fr(self, axs, fiber_id, **kwargs):
        if self.control is 'Displ':
            for i, quantity in enumerate(quantity_list[-3:]):
                axs[i, 1].plot(self.static_displ_exp,
                    self.predicted_fr[fiber_id][quantity][:, 1], **kwargs)
                axs[i, 0].plot(self.static_displ_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2], **kwargs)
        if self.control is 'Force':
            for i, quantity in enumerate(quantity_list[-3:]):
                axs[i, 1].plot(self.static_force_exp,
                    self.predicted_fr[fiber_id][quantity][:, 1], **kwargs)
                axs[i, 0].plot(self.static_force_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2], **kwargs)
        return


if __name__ == '__main__':
    # Load experiment data
    binned_exp_list = []
    for i in range(FIBER_TOT_NUM):
        with open('./pickles/binned_exp_%d.pkl' % i, 'rb') as f:
            binned_exp_list.append(pickle.load(f))
    # Generate data
    simFiberList = [[[] for j in
        range(level_num)] for i in range(len(factor_list))]
    for i, factor in enumerate(factor_list[:3]):
        for level in range(level_num):
            j = level
            for k, control in enumerate(control_list):
                simFiber = SimFiber(factor, level, control)
                simFiberList[i][j].append(simFiber)
                print(factor+str(level)+control+' is done.')
    # %% Generate table for integration
    spatial_table = np.empty([6, 3])
    for i, factor in enumerate(factor_list[:3]):
        for j, control in enumerate(control_list):
            for k, quantity in enumerate(quantity_list[-3:]):
                iqr = np.abs(simFiberList[i][3][j].dist[
                    2]['m%sint'%quantity] - simFiberList[i][1][j].dist[
                    2]['m%sint'%quantity])
                distance = .5 * np.abs(simFiberList[i][2][j].dist[
                    3]['m%sint'%quantity] - simFiberList[i][2][j].dist[
                    1]['m%sint'%quantity])
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
            surface_data = np.interp(xcoord, dist['cxnew'][-1],
                                     dist[surface_quantity][-1])
            # Get mcnc data
            mcnc_data = np.interp(xcoord, dist['mxnew'][-1],
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
                        axs[row, j].plot(dist['cxnew'][-1, :] * xscale,
                            dist[cquantity][-1, :] * cscale,
                            ls=ls, c=color, label=quantile_label_list[level])
                    for row, mquantity in enumerate(mquantity_list):
                        # Scaling the axes
                        if 'ress' in mquantity or 'sener' in mquantity:
                            mscale = 1e-3
                        else:
                            mscale = 1
                        # Plotting
                        axs[row+2, j].plot(dist['mxnew'][-1, :] * xscale,
                            dist[mquantity][-1, :] * mscale,
                            ls=ls, c=color, label=quantile_label_list[level])
    # Set x and y lim
    for axes in axs.ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Location (mm)')
    axs[0, 0].set_ylabel(r'Surface deformation (mm)')
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
    axs[0, 0].legend(handles[len(stim_plot_list)*(len(level_plot_list)//2)
        +len(stim_plot_list)//2::len(stim_plot_list)*len(
        level_plot_list)], [factor_display[5:].capitalize()
        for factor_display in factor_display_list[:3]], loc=3)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=3)
    # Add subtitles
    axs[0, 0].set_title('Deformation controlled')
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
                np.abs(integrate(simFiberLevelList[3], quantity, 2)
                - integrate(simFiberLevelList[1], quantity, 2)
                )
            distance_dict[quantity] = \
                .5 * np.abs(integrate(simFiberLevelList[2], quantity, 1)
                - integrate(simFiberLevelList[2], quantity, 3))
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
    fiber_id = FIBER_FIT_ID_LIST[0]
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
                            ls=ls, c=color, label=quantile_label_list[level])
                    for row, quantity in enumerate(quantity_list[-3:]):
                        scale = 1 if quantity is 'strain' else 1e-3
                        axes = axs[row+2, k]
                        axes.plot(
                            simFiber.traces[stim]['time'],
                            simFiber.traces[stim][quantity]*scale,
                            ls=ls, c=color, label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(r'Surface deformation (mm)')
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
    axs[0, 0].legend(handles[len(stim_plot_list)*(len(level_plot_list)//2)+
        len(stim_plot_list)//2::len(stim_plot_list)*len(
        level_plot_list)], [factor_display[5:].capitalize()
        for factor_display in factor_display_list[:3]], loc=4)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=4)
    # Add subtitles
    axs[0, 0].set_title('Deformation controlled')
    axs[0, 1].set_title('Pressure controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/temporal_distribution.png', dpi=300)
    plt.close(fig)
    # %% Calculating iqr for all temporal rate traces
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
                np.abs(integrate_rate(simFiberLevelList[3], quantity, 2)
                - integrate_rate(simFiberLevelList[1], quantity, 2)
                )
            distance_dict[quantity] = \
                .5 * np.abs(integrate_rate(simFiberLevelList[2], quantity, 1)
                - integrate_rate(simFiberLevelList[2], quantity, 3))
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
    # %% Plot temporal rate traces
    fiber_id = FIBER_FIT_ID_LIST[0]
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
                            ls=ls, c=color, label=quantile_label_list[level])
                    for row, quantity in enumerate(quantity_list[-3:]):
                        scale = 1 if quantity is 'strain' else 1e-3
                        axes = axs[row+2, k]
                        axes.plot(
                            simFiber.traces_rate[stim]['time'],
                            simFiber.traces_rate[stim][quantity] * scale,
                            ls=ls, c=color, label=quantile_label_list[level])
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
    axs[0, 0].legend(handles[len(stim_plot_list)*(len(level_plot_list)//2)
        +len(stim_plot_list)//2::len(stim_plot_list)*len(
        level_plot_list)], [factor_display[5:].capitalize()
        for factor_display in factor_display_list[:3]], loc=1)
    # The 5 quantile labels
    axs[0, 1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=1)
    # Add subtitles
    axs[0, 0].set_title('Deformation controlled')
    axs[0, 1].set_title('Pressure controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/temporal_rate_distribution.png', dpi=300)
    plt.close(fig)
    # %% Plot all simulations together
    fiber_id = FIBER_FIT_ID_LIST[0]
    # Calculate all the IQRs and compare force vs. displ
    def get_slope_iqr(simFiberLevelList, quantity):
        """
        Returns
        -------
        slope_iqr : list
            The 1st element for displ., 2nd for force.
        """
        slope_list_displ = [np.polyfit(
            simFiber.static_displ_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1], 1)[0]
            for simFiber in simFiberLevelList]
        slope_list_force = [np.polyfit(
            simFiber.static_force_exp,
            simFiber.predicted_fr[fiber_id][quantity].T[1], 1)[0]
            for simFiber in simFiberLevelList]
        slope_iqr = []
        slope_iqr.append(np.abs(
            (slope_list_displ[3] - slope_list_displ[1])/slope_list_displ[2]))
        slope_iqr.append(np.abs(
            (slope_list_force[3] - slope_list_force[1])/slope_list_force[2]))
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
    # %% Start plotting
    # Factors explaining the force-alignment - static
    fig, axs = plt.subplots(3, 3, figsize=(6.83, 6))
    for i, factor in enumerate(factor_list[:3]):
        for k, quantity in enumerate(quantity_list[-3:]):
#            for level in level_plot_list:
            for level in range(level_num):
                alpha = 1. - .4 * abs(level-2)
                color = (0, 0, 0, alpha)
                fmt = LS_LIST[i]
                label = quantile_label_list[level]
                simFiber = simFiberList[i][level][0]
                axs[0, k].plot(
                    simFiber.static_displ_exp,
                    simFiber.static_force_exp,
                    c=color, mec=color, ms=MS,
                    ls=fmt, label=label)
                axs[1, k].plot(
                    simFiber.static_displ_exp,
                    simFiber.predicted_fr[fiber_id][quantity].T[1],
                    c=color, mec=color, ms=MS,
                    ls=fmt, label=label)
                axs[2, k].plot(
                    simFiber.static_force_exp,
                    simFiber.predicted_fr[fiber_id][quantity].T[1],
                    c=color, mec=color, ms=MS,
                    ls=fmt, label=label)
    # X and Y limits
    for axes in axs[0:, :].ravel():
        axes.set_ylim(0, 10)
    for axes in axs[1:, :].ravel():
        axes.set_ylim(0, 45)
    for axes in axs[:2, :].ravel():
        axes.set_xlim(.3, .7)
        axes.set_xticks(np.arange(.3, .8, .1))
    for axes in axs[2, :].ravel():
        axes.set_xlim(0, 10)
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
    axs[0, 0].legend(handles[2::5], ['Thickness', 'Modulus', 'Visco.'], loc=2)
#    axs[0, 0].legend(handles[2::5], [factor_display[5:
#        ].capitalize() for factor_display in factor_display_list[:3]], loc=2)
    # The 5 quantile labels
    axs[0, 1].legend(handles[:3], ['Extreme', 'Quartile',
        'Median'], loc=2)
    # Save
    fig.tight_layout()
    fig.savefig('./plots/sim_compare_variance.png', dpi=300)
    plt.close(fig)
    # %% Calculate values for displacement vs. mcnc displacement
    spatial_y_table = np.empty([3])
    for i, factor in enumerate(factor_list[:3]):
        j = 0
        control = 'Displ'
        quantity = 'y'
        iqr = np.abs(simFiberList[i][3][j].dist[
            2]['m%sint'%quantity] - simFiberList[i][1][j].dist[
            2]['m%sint'%quantity])
        distance = .5 * np.abs(simFiberList[i][2][j].dist[
            3]['m%sint'%quantity] - simFiberList[i][2][j].dist[
            1]['m%sint'%quantity])
        spatial_y_table[i] = iqr / distance
    spatial_y_table_sum = spatial_y_table.sum()
    # Calculate Pearson correlation coefficients
    dist = simFiberList[0][2][0].dist[3]
    # Calculate shape R^2
    xcoord = np.linspace(0, MAX_RADIUS, 100)
    # Get surface data
    surface_quantity = 'cy'
    surface_data = np.interp(xcoord, dist['cxnew'][-1],
                             dist[surface_quantity][-1])
    # Get mcnc data
    mcnc_data = np.interp(xcoord, dist['mxnew'][-1],
                          dist['my'][-1])
    # Calculate correlation
    spatial_y_pearsonr, spatial_y_pearsonp = pearsonr(surface_data, mcnc_data)
    # %% Plot distribution
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 4), sharex=True)
    for i, factor in enumerate(factor_list[:3]):
        for level in level_plot_list:
            j = 0
            control = 'Displ'
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
                cquantity = 'cy'
                cscale = 1e-3
                axs[0].plot(dist['cxnew'][-1, :] * xscale,
                    dist[cquantity][-1, :] * cscale,
                    ls=ls, c=color, label=quantile_label_list[level])
                # Scaling the axes
                mquantity = 'my'
                mscale = 1e-3
                # Plotting
                axs[1].plot(dist['mxnew'][-1, :] * xscale,
                    dist[mquantity][-1, :] * mscale,
                    ls=ls, c=color, label=quantile_label_list[level])
    # Set x and y lim
    for axes in axs.ravel():
        axes.set_xlim(0, MAX_RADIUS*1e3)
    # Formatting labels
    axs[1].set_xlabel('Location (mm)')
    axs[0].set_ylabel(r'Surface deformation (mm)')
    axs[1].set_ylabel(r'Internal displacement (mm)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[len(stim_plot_list)*(len(level_plot_list)//2)
        +len(stim_plot_list)//2::len(stim_plot_list)*len(
        level_plot_list)], [factor_display[5:].capitalize()
        for factor_display in factor_display_list[:3]], loc=3)
    # The 5 quantile labels
    axs[1].legend(handles[1:3*len(level_plot_list)+1:3], [
        'Quartile', 'Median'], loc=3)
    # Add subtitles
    axs[0].set_title('Deformation controlled')
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/spatial_cy_my.png', dpi=300)
    plt.close(fig)