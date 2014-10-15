# -*- coding: utf-8 -*-
"""
Created on Sun May  4 22:38:40 2014

@author: Yuxiang Wang
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import dblquad
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
quantity_list = ['force', 'displ', 'stress', 'strain', 'sener']
quantile_label_list = ['Min', 'LQ', 'Med', 'UQ', 'Max']
phase_list = ['dynamic', 'static']
percentage_label_list = ['%d%%' % i for i in range(50, 175, 25)]
displcoeff = np.loadtxt('./csvs/displcoeff.csv', delimiter=',')
stim_num = 6
AREA = np.pi * 1e-3**2 / 4
MAX_RADIUS = 1e-3


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
            An int in [0, 4]. Corresponds to min., lower-quartile, median, 
            upper-quartile and max.
        control : str
            The applied control in simulation, either force or displacement,
            noted by 'Force' or 'Displ'.
        """
        self.factor = factor
        self.level = level
        self.control = control
        self.load_traces()
        self.get_traces_mean()
        self.load_trans_params()
        self.get_predicted_fr()
        self.get_line_fit()
        self.get_dist()
        return
    
    def get_dist(self):
        fpath = BASE_CSV_PATH
        # Use the 2nd magnitude as example
        stim = 2
        self.dist = {}        
        key_list = ['cpress', 'cxnew', 'cxold', 'cy', 'msener',
                    'mstrain', 'mstress', 'mxnew', 'mxold',
                    'time']
        for key in key_list:
            self.dist[key] = np.loadtxt(
                fpath+self.factor+str(self.level)+str(stim)+self.control+'_'
                +key+'.csv', delimiter=',')
        argsort = self.dist['cxold'][-1].argsort()
        # Sort order in x
        for key in key_list:
            # Calculate integration over area            
            if key.startswith('c'):
                self.dist[key] = (self.dist[key].T[argsort]).T
        # Propagate time
        self.dist['time'] = np.tile(self.dist['time'][:, np.newaxis],
                                    self.dist['cxnew'].shape[1])
        # Calculate integration over area
        for key in key_list:
            if 'x' not in key and 'y' not in key and 'time' not in key:
                def get_field(r):
                    return np.interp(r, self.dist[key[0]+'xnew'][-1],
                                     self.dist[key][-1])
                self.dist[key+'int'] = dblquad(
                    lambda r, theta: get_field(r) * r,
                    0, 2 * np.pi,
                    lambda r: 0,
                    lambda r: MAX_RADIUS
                    )[0]
        return
    
    def get_traces_mean(self):
        def get_mean(quantity_array, max_index):
#            dynamic_window = range(int(max_index/DT))
            static_window = np.arange(int(STATIC_START/DT), int(STATIC_END/DT))
#            dynamic_mean = quantity_array[dynamic_window].mean()
            static_mean = quantity_array[static_window].mean()
            return static_mean
        self.traces_mean = {quantity: np.asarray([get_mean(traces[quantity],
            traces['max_index']) for traces in self.traces]) for quantity in
            quantity_list}
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
        # Read the non-zero output from FEM
        for i, fname in enumerate(fname_list):
            time, force, displ, stress, strain, sener = np.loadtxt(
                fpath+fname, delimiter=',').T
            fine_time = np.arange(0, time.max(), DT)
            self.traces[i+1]['time'] = fine_time
            self.traces[i+1]['force'] = np.interp(fine_time, time, force)
            self.traces[i+1]['displ'] = np.interp(fine_time, time, displ)
            self.traces[i+1]['stress'] = np.interp(fine_time, time, stress)
            self.traces[i+1]['strain'] = np.interp(fine_time, time, strain)
            self.traces[i+1]['sener'] = np.interp(fine_time, time, sener)
            self.traces[i+1]['max_index'] = self.traces[i+1]['force'].argmax()
        # Fill the zero-stim trace
        self.traces[0]['max_index'] = self.traces[1]['max_index']
        self.traces[0]['time'] = self.traces[1]['time']
        self.traces[0]['force'] = np.zeros_like(self.traces[0]['time'])
        self.traces[0]['displ'] = np.zeros_like(self.traces[0]['time'])
        self.traces[0]['stress'] = np.zeros_like(self.traces[0]['time'])
        self.traces[0]['strain'] = np.zeros_like(self.traces[0]['time'])
        self.traces[0]['sener'] = np.zeros_like(self.traces[0]['time'])
        # Get the FEM and corresponding displ / force
        self.static_displ_fem = np.array([self.traces[i]['displ'][-1]
            for i in range(stim_num)])
        self.static_displ_exp = displcoeff[0] + self.static_displ_fem *\
            1e6 * displcoeff[1]
        self.static_force_fem = np.array([self.traces[i]['force'][-1]
            for i in range(stim_num)])
        self.static_force_exp = self.static_force_fem * 1e3
        return
    
    def get_predicted_fr(self):
        self.predicted_fr = [{} for i in range(FIBER_TOT_NUM)]
        for fiber_id in FIBER_FIT_ID_LIST:
            for quantity in quantity_list:
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
            for quantity in quantity_list:
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
            for i, quantity in enumerate(quantity_list):
                axs[i, 1].plot(self.static_displ_exp, 
                    self.predicted_fr[fiber_id][quantity][:, 1], **kwargs)
                axs[i, 0].plot(self.static_displ_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2], **kwargs)
        if self.control is 'Force':
            for i, quantity in enumerate(quantity_list):
                axs[i, 1].plot(self.static_force_exp, 
                    self.predicted_fr[fiber_id][quantity][:, 1], **kwargs)
                axs[i, 0].plot(self.static_force_exp,
                    self.predicted_fr[fiber_id][quantity][:, 2], **kwargs)
        return


def fit_sim_cluster(simFiberLevelList, fiber_id, quantity):
    displ_list = []
    force_list = []
    dynamic_fr_list = []
    static_fr_list = []
    for simFiber in simFiberLevelList:
        displ_list.extend(simFiber.static_displ_exp)
        force_list.extend(simFiber.static_force_exp)
        dynamic_fr_list.extend(simFiber.predicted_fr[fiber_id][quantity][:, 2])
        static_fr_list.extend(simFiber.predicted_fr[fiber_id][quantity][:, 1])
    # Fitting
    displ_static_fit_param = np.polyfit(displ_list, static_fr_list, 1)
    force_static_fit_param = np.polyfit(force_list, static_fr_list, 1)
    displ_dynamic_fit_param = np.polyfit(displ_list, dynamic_fr_list, 1)
    force_dynamic_fit_param = np.polyfit(force_list, dynamic_fr_list, 1)
    # Get predictions
    displ_static_fit_predict = np.polyval(displ_static_fit_param, displ_list)
    force_static_fit_predict = np.polyval(force_static_fit_param, force_list)
    displ_dynamic_fit_predict = np.polyval(displ_dynamic_fit_param, displ_list)
    force_dynamic_fit_predict = np.polyval(force_dynamic_fit_param, force_list)
    # Get resvar
    resvar = {
        'displ_static':  (displ_static_fit_predict -
            np.asarray(static_fr_list)).var(),
        'force_static':  (force_static_fit_predict -
            np.asarray(static_fr_list)).var(),
        'displ_dynamic':  (displ_dynamic_fit_predict -
            np.asarray(dynamic_fr_list)).var(),
        'force_dynamic':  (force_dynamic_fit_predict -
            np.asarray(dynamic_fr_list)).var(),
        }
    # Get prediction
    prediction = {
        'displ': np.asarray(displ_list),
        'force': np.asarray(force_list),
        'displ_static': displ_static_fit_predict,
        'force_static': force_static_fit_predict,
        'displ_dynamic': displ_dynamic_fit_predict,
        'force_dynamic': force_dynamic_fit_predict,
        }
    return resvar, prediction


if __name__ == '__main__':
    # Load experiment data
    binned_exp_list = []
    for i in range(FIBER_TOT_NUM):
        with open('./pickles/binned_exp_%d.pkl' % i, 'rb') as f:
            binned_exp_list.append(pickle.load(f))
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
    #%% 
    # Switches for the script
    plot_exp_flag = False                
    """
    # Plotting the big figure
    fig_list = [[] for i in range(len(factor_list))]
    axs_list = [[] for i in range(len(factor_list))]                
    for fiber_id in FIBER_FIT_ID_LIST:
        for i, factor in enumerate(factor_list):
            fig, axs = plt.subplots(len(quantity_list), 4, figsize=(6.83,
                9.19))
            fig_list.append(fig)
            axs_list.append(axs)
            # Plot experiment
            def plot_exp(axs):
                color = '.8'
                # Plot experiment
                for row in range(axs.shape[0]):
                    for col in range(axs.shape[1]):
                        axes = axs[row, col]
                        phase = ['dynamic', 'static'][col % 2]
                        control = ['displ', 'force'][col // 2]
                        [axes.errorbar(binned_exp[control+'_mean'],
                            binned_exp[phase+'_fr_mean'],
                            binned_exp[phase+'_fr_std'],
                            fmt=':'+MARKER_LIST[i],
                            c=color, mec=color, ms=MS) for i, binned_exp in
                            enumerate(binned_exp_list)]
                return axs
            if plot_exp_flag:
                plot_exp(axs)
            # Plot simulation
            for j, level in enumerate(range(level_num)):
                for k, control in enumerate(control_list):
                    marker = MARKER_LIST[fiber_id]
                    color = str(.6 - .15 * level)
                    if i < 2:
                        level_label_list = quantile_label_list
                    else:
                        level_label_list = percentage_label_list
                    simFiberList[i][j][k].plot_predicted_fr(axs[:, 2*k:2*k+2], 
                        fiber_id, marker=marker, c=color, mec=color, ms=MS,
                        label=level_label_list[level])
            # Set x and y lim
            ymin_array = np.empty_like(axs, dtype=np.float)
            ymax_array = np.empty_like(axs, dtype=np.float)
            for row, axs_row in enumerate(axs):
                for col, axes in enumerate(axs_row):
                    # X-ticks
                    if col < 2:
                        axes.set_xlim(300, 650)
                    else:
                        axes.set_xlim(-0.5, 8.5)
                    # Y-lim record
                    ymin_array[row, col] = axes.get_ylim()[0]                        
                    ymax_array[row, col] = axes.get_ylim()[1]
            for row, axs_row in enumerate(axs):
                for col, axes in enumerate(axs_row):
                    ymin = ymin_array[row, col%2::2].min()
                    ymax = ymax_array[row, col%2::2].max()
                    axes.set_ylim(-5, ymax+5)
            # Adding axes labels
            for row, axes in enumerate(axs[:, 0]):
                quantity_label_list = ['force', 'displacement', 'stress', 
                    'strain', 'sener']
                axes.set_ylabel('Predicted from ' + quantity_label_list[row]\
                    +'\nMean firing (Hz)')
            for col, axes in enumerate(axs[-1, :]):
                if col < 2:
                    axes.set_xlabel(r'Displacement ($\mu$m)')
                else:
                    axes.set_xlabel(r'Force (mN)')
            # Addting titles
            for col, axes in enumerate(axs[0, :]):
                title = ['Dynamic', 'Static'][col%2]
                axes.set_title(title)
            # Adding figure texts
            fig.text(0.25, .99, 'Displacement controlled', fontsize=10, 
                 fontweight='bold', ha='center', va='top')
            fig.text(0.75, .99, 'Force controlled', fontsize=10,
                fontweight='bold', ha='center', va='top')
            # Adding panel labels
            for axes_id, axes in enumerate(axs.ravel()):
                if axes_id % 4 == 1 or axes_id % 4 == 3:
                    pos_x = -0.26
                else:
                    pos_x = -0.31
                axes.text(pos_x, 1.1, chr(65+axes_id), 
                    transform=axes.transAxes, fontsize=12, fontweight='bold',
                    va='top')
            # Adding legend
            h, l = axs[0, 0].get_legend_handles_labels()
            legend = fig.legend(h, l, bbox_to_anchor=(.05, 0.02, .9, .1),
                loc=3, ncol=level_num, mode='expand', borderaxespad=0.,
                frameon=True)
            frame = legend.get_frame()
            frame.set_linewidth(.5)
            # Save figure
            fig.tight_layout()
            fig.subplots_adjust(top=.95, bottom=.1)
            fig.savefig('./plots/'+factor+'Fiber%d.png'%fiber_id, dpi=300)
    #%% Fit all the variances and compare force vs. displ
    resvar_list = [[{} for k in range(len(control_list))]
        for i in range(len(factor_list))]
    prediction_list = [[{} for k in range(len(control_list))]
        for i in range(len(factor_list))]
    for fiber_id in FIBER_FIT_ID_LIST:
        for i, factor in enumerate(factor_list):
            for k, control in enumerate(control_list):
                simFiberLevelList = [simFiberList[i][j][k] for j in
                    range(level_num)]
                for l, quantity in enumerate(quantity_list[-3:]):
                    resvar_list[i][k][quantity], prediction_list[i][k][
                        quantity] = fit_sim_cluster(simFiberLevelList,
                        fiber_id, quantity)            
    #%% Factors explaining the force-alignment - static
    for k, quantity in enumerate(quantity_list[-3:]):
        fig, axs = plt.subplots(3, 2, figsize=(3.27, 4.5))
        fiber_id = FIBER_FIT_ID_LIST[0]
        for i, factor in enumerate(factor_list[:3]):
            factor_display = factor_display_list[i]
            for level in range(level_num):
                color = str(.6 - .15 * level)
                axs[i, 0].plot(simFiberList[i][level][0].static_displ_exp, 
                    simFiberList[i][level][0].predicted_fr[fiber_id][quantity]
                    [:, 1], c=color, mec=color, ms=MS, marker='o',
                    label=quantile_label_list[level])
                axs[i, 0].set_ylabel('Mean response (Hz)\nVary %s'
                    % factor_display[5:])
                axs[i, 1].plot(simFiberList[i][level][0].static_force_exp, 
                    simFiberList[i][level][0].predicted_fr[fiber_id][quantity]
                    [:, 1], c=color, mec=color, ms=MS, marker='o',
                    label=quantile_label_list[level])
        # Plot linear regression
        for i, axes in enumerate(axs[:, 0]):
            axes.plot(np.sort(prediction_list[i][0][quantity]['displ']), 
                np.sort(prediction_list[i][0][quantity]['displ_static']),
                '-.k', label='Linear regression')
        for i, axes in enumerate(axs[:, 1]):
            axes.plot(np.sort(prediction_list[i][0][quantity]['force']), 
                np.sort(prediction_list[i][0][quantity]['force_static']),
                '-.k', label='Linear regression')
        # Formatting
        axs[-1, 0].set_xlabel(r'Static displ. ($\mu$m)')
        axs[-1, 1].set_xlabel(r'Static force (mN)')    
        ymargin = 5
        xmargin = 1
        for i, axes in enumerate(axs.ravel()):
            axes.set_ylim(bottom=axes.get_ylim()[0]-ymargin)
            axes.set_ylim(top=axes.get_ylim()[1]+ymargin)
            if i % 2 == 0:
                axes.set_xlim(300, 600)
                axes.set_xticks(np.arange(300, 700, 100))
            elif i % 2 == 1:
                xmargin = 2
                axes.set_xlim(left=axes.get_xlim()[0]-xmargin)
                axes.set_xlim(right=axes.get_xlim()[1]+xmargin) 
        for axes_id, axes in enumerate(axs.ravel()):
            xloc = -.27
            axes.text(xloc, 1.15, chr(65+axes_id), transform=axes.transAxes,
                fontsize=12, fontweight='bold', va='top')      
        # Legend
        h, l = axs[0, 0].get_legend_handles_labels()
        legend = fig.legend(h, l, bbox_to_anchor=(.05, 0.02, .9, .1),
            loc=3, ncol=level_num, mode='expand', borderaxespad=0.,
            frameon=True)
        frame = legend.get_frame()
        frame.set_linewidth(.5)
        # Save
        fig.tight_layout()
        fig.suptitle('Predicted by ' + quantity)
        fig.subplots_adjust(top=.92, bottom=.15)
        fig.savefig('./plots/paper_%s_align_force_static.png' % quantity,
                    dpi=300)
    #%% Print all force ratios
    for quantity in quantity_list[-3:]:
        fig, axs = plt.subplots(3, 2, figsize=(3.27, 4.5))
        fiber_id = FIBER_FIT_ID_LIST[0]
        for i, factor in enumerate(factor_list[:3]):
            for level in range(level_num):
                color = str(.6 - .15 * level)
                simFiber = simFiberList[i][level][0]
                quantity_array = simFiber.traces_mean[quantity]*1e-3\
                    if quantity is 'stress' or quantity is 'sener' else\
                    simFiber.traces_mean[quantity]
                axs[i, 0].plot(simFiber.static_displ_exp,
                    quantity_array, label=quantile_label_list[level],
                    c=color, mec=color, ms=MS, marker='o')
                axs[i, 1].plot(simFiber.static_force_exp,
                    quantity_array,  label=quantile_label_list[level],
                    c=color, mec=color, ms=MS, marker='o')
        # Legend
        h, l = axs[0, 0].get_legend_handles_labels()
        legend = fig.legend(h, l, bbox_to_anchor=(.05, 0.02, .9, .1),
            loc=3, ncol=level_num, mode='expand', borderaxespad=0.,
            frameon=True)
        frame = legend.get_frame()
        frame.set_linewidth(.5)
        # Formatting
        for i, axes in enumerate(axs.ravel()):
            if quantity is 'strain':
                ymargin = .1
            elif quantity is 'stress':
                ymargin = 1
            elif quantity is 'sener':
                ymargi = .2
            axes.set_ylim(bottom=axes.get_ylim()[0]-ymargin)
            axes.set_ylim(top=axes.get_ylim()[1]+ymargin)
            if i % 2 == 0:
                axes.set_xlim(300, 600)
                axes.set_xticks(np.arange(300, 700, 100))
            elif i % 2 == 1:
                xmargin = 2
                axes.set_xlim(left=axes.get_xlim()[0]-xmargin)
                axes.set_xlim(right=axes.get_xlim()[1]+xmargin)       
        axs[-1, 1].set_xlabel(r'Static force (mN)')
        axs[-1, 0].set_xlabel(r'Static displ. ($\mu$m)')
        for row, axes in enumerate(axs[:, 0]):
            factor_display = factor_display_list[row][5:]
            if quantity is 'stress':
                axes.set_ylabel(r'Static stress (kPa)'+'\nVary %s'
                    % factor_display)
            elif quantity is 'sener':
                axes.set_ylabel(r'Static sener (kJ/$m^3$)'+'\nVary %s'
                    % factor_display)
            else:
                axes.set_ylabel(r'Static strain'+'\nVary %s' % factor_display)
        for axes_id, axes in enumerate(axs.ravel()):
            if quantity is 'stress':
                xloc = -.27
            elif quantity is 'strain':
                xloc = -.35
            elif quantity is 'sener':
                xloc = -.3
            axes.text(xloc, 1.15, chr(65+axes_id), transform=axes.transAxes,
                fontsize=12, fontweight='bold', va='top')      
        fig.tight_layout()
        fig.subplots_adjust(top=.95, bottom=.15)
        fig.savefig('./plots/paper_%s-force.png'%quantity, dpi=300)
    #%% Force traces under force control
    fiber_id = FIBER_FIT_ID_LIST[0]
    for quantity in quantity_list[-3:]:
        fig, axs = plt.subplots(1, 2, figsize=(3.27, 2))        
        for level in range(level_num):
            color = str(.6 - .15 * level)
            traces = simFiberList[0][level][0].traces[2]
            axs[0].plot(traces['time'], traces[quantity], c=color,
                mec=color)
            axs[0].set_label('Displ control')
            traces = simFiberList[0][level][1].traces[2]                
            axs[1].plot(traces['time'], traces[quantity], c=color,
                mec=color)
            axs[1].set_label('Force control')                
            for axes in axs:            
                axes.set_xlim(0, 0.3)
                axes.set_xlabel('Time (s)')
                axes.set_ylim(top=max([axs[i].get_ylim()[1] for i in range(2)]))
            axs[0].set_ylabel(quantity.capitalize())
        for axes_id, axes in enumerate(axs.ravel()):
            axes.text(-.15, 1.2, chr(65+axes_id), transform=axes.transAxes,
                fontsize=12, fontweight='bold', va='top')      
        fig.tight_layout()
        fig.suptitle('Predicted by ' + quantity)
        fig.subplots_adjust(top=.8)
        fig.savefig('./plots/paper_%s_fc_variability.png' % quantity,
                    dpi=300)
    #%% Exapmle quantity traces - skinthickness, displ vs force control
    fiber_id = FIBER_FIT_ID_LIST[0]
    fig, axs = plt.subplots(3, 2, figsize=(3.27, 5))
    i, factor = 0, factor_list[0]
    for k, control in enumerate(control_list):
        for row, quantity in enumerate(quantity_list[2:]):
            scale = 1 if quantity is 'strain' else 1e-3
            for level in range(level_num):
                color = str(.6 - .15 * level)
                simFiber = simFiberList[i][level][k]
                axes = axs[row, k]
                axes.plot(
                    simFiber.traces[stim_num//2]['time'],
                    simFiber.traces[stim_num//2][quantity]*scale, ls='-',
                    c=color, label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Stress (kPa)')
    axs[1, 0].set_ylabel('Strain')
    axs[2, 0].set_ylabel(r'SED (kPa/$m^3$)')
    # Set x and y lim
    ymin_array = np.empty_like(axs, dtype=np.float)
    ymax_array = np.empty_like(axs, dtype=np.float)
    for row, axs_row in enumerate(axs):
        for col, axes in enumerate(axs_row):
            # Y-lim record
            ymin_array[row, col] = axes.get_ylim()[0]                        
            ymax_array[row, col] = axes.get_ylim()[1]
    for row, axs_row in enumerate(axs):
        for col, axes in enumerate(axs_row):
            ymin = ymin_array[row, :].min()
            ymax = ymax_array[row, :].max()
            axes.set_ylim(ymin, ymax)        
    # Formatting
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.27, 1.12, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
        axes.set_xlim(-.5, 5.5)
    # Legend
    h, l = axs[0, 0].get_legend_handles_labels()
    legend = fig.legend(h, l, bbox_to_anchor=(.05, 0.02, .9, .1),
        loc=3, ncol=level_num, mode='expand', borderaxespad=0.,
        frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(.5)    
    fig.tight_layout()
    fig.subplots_adjust(bottom=.15)
    fig.savefig('./plots/example_sim_force_traces.png', dpi=300)
    """
    #%% Generate table for integration and plot distribution
    int_table = np.empty([6, 3])
    for i, factor in enumerate(factor_list[:3]):
        for j, control in enumerate(control_list):
            for k, quantity in enumerate(quantity_list[2:]):
                int_table[3*j+k, i] = np.abs(simFiberList[i][3][j].dist[
                    'm%sint'%quantity] - simFiberList[i][1][j].dist[
                    'm%sint'%quantity]) / simFiberList[i][2][j].dist[
                    'm%sint'%quantity]
    int_table_mean = int_table.mean(axis=1)
    # Plot distribution
    fig, axs = plt.subplots(4, 2, figsize=(6.83, 8), sharex=True)
    mquantity_list = ['mstress', 'mstrain', 'msener']
    cquantity_list = ['cy', 'cpress']
    for i, factor in enumerate(factor_list[:3]):
        for j, control in enumerate(control_list):
            for level in range(level_num):
                color = str(.6-.15 * level)
                ls = LS_LIST[i]
                dist = simFiberList[i][level][j].dist
                xscale = 1e3
                cquantity = cquantity_list[j]
                if 'y' in cquantity:
                    cscale = 1e6
                elif 'ress' in cquantity:
                    cscale = 1e-3
                axs[0, j].plot(dist['cxnew'][-1, :] * xscale, 
                    dist[cquantity][-1, :] * cscale,
                    ls=ls, c=color, label=quantile_label_list[level])
                for row, mquantity in enumerate(mquantity_list):
                    # Scaling the axes
                    if 'ress' in mquantity or 'sener' in mquantity:
                        mscale = 1e-3
                    else:
                        mscale = 1
                    # Plotting
                    axs[row+1, j].plot(dist['mxnew'][-1, :] * xscale, 
                        dist[mquantity][-1, :] * mscale, 
                        ls=ls, c=color, label=quantile_label_list[level])
#    # Annotate
#    for i, axes in enumerate(axs[1:].T.ravel()):
#        axes.text(.95, .8,
#                  r'$\overline{(\frac{IQR}{median})}$=%.3f'%int_table_mean[i],
#                  ha='right', va='top', transform=axes.transAxes, fontsize=8)
    # Set x and y lim
    ymin_array = np.empty_like(axs[1:], dtype=np.float)
    ymax_array = np.empty_like(axs[1:], dtype=np.float)
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            # Y-lim record
            ymin_array[row, col] = axes.get_ylim()[0]                        
            ymax_array[row, col] = axes.get_ylim()[1]
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            ymin = ymin_array[row, :].min()
            ymax = ymax_array[row, :].max()
            axes.set_ylim(ymin, ymax)
            axes.set_xlim(0, 1)
    axs[0, 0].set_ylim(-150, 100)
    axs[0, 1].set_ylim(axs[1, 1].get_ylim())
    # Formatting labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Location (mm)')
    axs[0, 0].set_ylabel(r'Deformation ($\mu$m)')
    axs[0, 1].set_ylabel(r'Pressure (kPa)')
    axs[1, 0].set_ylabel('Stress (kPa)')
    axs[1, 1].set_ylabel('Stress (kPa)')
    axs[2, 0].set_ylabel('Strain')
    axs[2, 1].set_ylabel('Strain')
    axs[3, 0].set_ylabel(r'SED (kJ/m$^3$)')
    axs[3, 1].set_ylabel(r'SED (kJ/m$^3$)')
    # Added panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.125, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles[4::5], [factor_display[5:].capitalize() for
        factor_display in factor_display_list[:3]], loc=2)
    # The 5 quantile labels
    axs[0, 1].legend(handles[:5], labels[:5], loc=1)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/spatial_distribution.png', dpi=300)
    #%% Calculating iqr for all and plot temporal traces
    # Calculate iqrs
    def calculate_iqr(simFiberLevelList):
        iqr_dict = {}
        for quantity in quantity_list:
            iqr_dict[quantity] = \
                np.abs(simFiberLevelList[-2].traces[stim_num//2][quantity][-1]
                - simFiberLevelList[1].traces[stim_num//2][quantity][-1]
                ) / simFiberLevelList[len(simFiberLevelList)//2].traces[
                stim_num//2][quantity][-1]
        return iqr_dict    
    iqr_table = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            iqr_dict = calculate_iqr(
                [simFiberList[i][j][k] for j in range(level_num)])
            for row, quantity in enumerate(quantity_list[2:]):
                iqr_table[3*k+row, i] = iqr_dict[quantity]
    iqr_table_mean = iqr_table.mean(axis=1)
    # Plot temporal traces
    fiber_id = FIBER_FIT_ID_LIST[0]
    fig, axs = plt.subplots(4, 2, figsize=(6.83, 8), sharex=True)
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            control = control.lower()
            for level in range(level_num):
                color = str(.6 - .15 * level)
                ls = LS_LIST[i]
                simFiber = simFiberList[i][level][k]
                cscale = 1e6 if control == 'displ' else 1e3
                axs[0, k].plot(
                    simFiber.traces[stim_num//2]['time'],
                    simFiber.traces[stim_num//2][control] * cscale,
                    ls=ls, c=color, label=quantile_label_list[level])
                for row, quantity in enumerate(quantity_list[2:]):
                    scale = 1 if quantity is 'strain' else 1e-3
                    axes = axs[row+1, k]
                    axes.plot(
                        simFiber.traces[stim_num//2]['time'],
                        simFiber.traces[stim_num//2][quantity]*scale, ls=ls,
                        c=color, label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(r'Displacement ($\mu$m)')
    axs[0, 1].set_ylabel(r'Force (mN)')
    axs[1, 0].set_ylabel('Stress (kPa)')
    axs[1, 1].set_ylabel('Stress (kPa)')
    axs[2, 0].set_ylabel('Strain')
    axs[2, 1].set_ylabel('Strain')
    axs[3, 0].set_ylabel(r'SED (kPa/$m^3$)')
    axs[3, 1].set_ylabel(r'SED (kPa/$m^3$)')
    # Set x and y lim
    ymin_array = np.empty_like(axs[1:], dtype=np.float)
    ymax_array = np.empty_like(axs[1:], dtype=np.float)
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            # Y-lim record
            ymin_array[row, col] = axes.get_ylim()[0]                        
            ymax_array[row, col] = axes.get_ylim()[1]
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            ymin = ymin_array[row, :].min()
            ymax = ymax_array[row, :].max()
            axes.set_ylim(ymin, ymax)        
    axs[0, 0].set_ylim(0, 150)
    axs[0, 1].set_ylim(0, 6)
    # Formatting
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.125, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
        axes.set_xlim(-.25, 5.5)
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles[4::5], [factor_display[5:].capitalize() for
        factor_display in factor_display_list[:3]], loc=4)
    # The 5 quantile labels
    axs[0, 1].legend(handles[:5], labels[:5], loc=4)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/example_sim_traces.png', dpi=300)
    #%% Calculating rate iqr for all and plot rate temporal traces
    # Calculate iqrs
    def calculate_iqr(simFiberLevelList):
        iqr_dict = {}
        for quantity in quantity_list:
            iqr_dict[quantity] = \
                np.abs(simFiberLevelList[-2].traces[stim_num//2][quantity][-1]
                - simFiberLevelList[1].traces[stim_num//2][quantity][-1]
                ) / simFiberLevelList[len(simFiberLevelList)//2].traces[
                stim_num//2][quantity][-1]
        return iqr_dict    
    iqr_table = np.empty((6, 3))
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            iqr_dict = calculate_iqr(
                [simFiberList[i][j][k] for j in range(level_num)])
            for row, quantity in enumerate(quantity_list[2:]):
                iqr_table[3*k+row, i] = iqr_dict[quantity]
    iqr_table_mean = iqr_table.mean(axis=1)
    # Plot temporal traces
    fiber_id = FIBER_FIT_ID_LIST[0]
    fig, axs = plt.subplots(4, 2, figsize=(6.83, 8), sharex=True)
    for i, factor in enumerate(factor_list[:3]):
        for k, control in enumerate(control_list):
            control = control.lower()
            for level in range(level_num):
                color = str(.6 - .15 * level)
                ls = LS_LIST[i]
                simFiber = simFiberList[i][level][k]
                cscale = 1e6 if control == 'displ' else 1e3
                cscale *= 1e5 # compensate for frequency
                axs[0, k].plot(
                    simFiber.traces[stim_num//2]['time'][1:],
                    np.diff(simFiber.traces[stim_num//2][control]) * cscale,
                    ls=ls, c=color, label=quantile_label_list[level])
                for row, quantity in enumerate(quantity_list[2:]):
                    scale = 1 if quantity is 'strain' else 1e-3
                    scale *= 1e5 # compensate for frequency
                    axes = axs[row+1, k]
                    axes.plot(
                        simFiber.traces[stim_num//2]['time'][1:],
                        np.diff(simFiber.traces[stim_num//2][quantity])*scale,
                        ls=ls, c=color, label=quantile_label_list[level])
    # Add axes labels
    for axes in axs[-1, :]:
        axes.set_xlabel('Time (s)')
    axs[0, 0].set_ylabel(r'Displacement rate ($\mu$m/s)')
    axs[0, 1].set_ylabel(r'Force rate (mN/s)')
    axs[1, 0].set_ylabel('Stress rate (kPa/s)')
    axs[1, 1].set_ylabel('Stress rate (kPa/s)')
    axs[2, 0].set_ylabel(r'Strain rate (s$^{-1}$)')
    axs[2, 1].set_ylabel(r'Strain rate (s$^{-1}$)')
    axs[3, 0].set_ylabel(r'SED (kPa$\cdot m^{-3}s^{-1}$)')
    axs[3, 1].set_ylabel(r'SED (kPa$\cdot m^{-3}s^{-1}$)')
    # Set x and y lim
    ymin_array = np.empty_like(axs[1:], dtype=np.float)
    ymax_array = np.empty_like(axs[1:], dtype=np.float)
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            # Y-lim record
            ymin_array[row, col] = axes.get_ylim()[0]                        
            ymax_array[row, col] = axes.get_ylim()[1]
    for row, axs_row in enumerate(axs[1:]):
        for col, axes in enumerate(axs_row):
            ymin = ymin_array[row, :].min()
            ymax = ymax_array[row, :].max()
            axes.set_ylim(ymin, ymax)
#    axs[0, 0].set_ylim(0, 150)
#    axs[0, 1].set_ylim(0, 6)
    # Formatting
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.125, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
        axes.set_xlim(-.05, 0.35)
    # Add legends
    # The line type labels
    handles, labels = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(handles[4::5], [factor_display[5:].capitalize() for
        factor_display in factor_display_list[:3]], loc=1)
    # The 5 quantile labels
    axs[0, 1].legend(handles[:5], labels[:5], loc=1)
    # Save figure
    fig.tight_layout()
    fig.savefig('./plots/example_sim_rate_traces.png', dpi=300)

    