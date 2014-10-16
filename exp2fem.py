# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 21:10:47 2014

@author: Yuxiang Wang
"""

#%%

import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import os
import pickle
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from cleandata.convert import CleanFiber
from constants import (DT, FIBER_TOT_NUM, MARKER_LIST, COLOR_LIST, MS,
    STATIC_START ,STATIC_END, FIBER_MECH_ID, FIBER_FIT_ID_LIST,
    EVAL_DISPL, EVAL_FORCE, LS_LIST)
quantity_name_list = ['force', 'displ', 'stress', 'strain', 'sener']



def main():
    return


class Fiber:
    def __init__(self, fiber_id, make_plot=False):
        # Set the class attributes
        self.fiber_id = fiber_id
        self.make_plot = make_plot
        # Load the bulk summary-csv file from convert.py output
        self.load_fiber_data()
        self.get_stim_group_num_list()
        self.get_stim_group()
        self.get_displ_coeff()
        self.get_ramp_time_coeff()
        self.generate_stim_block_array()
        self.generate_binned_exp()
        self.get_lumped_dict()
        return
    
    def get_lumped_dict(self):
        """        
        Construct the target FR array
        """
        dynamic_fr_list, static_fr_list, stim_num_list, displ_list,\
            force_list = [], [], [], [], []
        for i in range(len(self.stim_group_dict)):
            dynamic_fr_list.extend(self.stim_group_dict[i][
                'dynamic_avg_fr'])
            static_fr_list.extend(self.stim_group_dict[i]
                ['static_avg_fr'])
            displ_list.extend(self.stim_group_dict[i]
                ['static_displ'])
            force_list.extend(self.stim_group_dict[i]
                ['static_force'])
            stim_num_list.extend(i*np.ones(self.stim_group_dict[i][
                'static_avg_fr'].shape[0]))
        # The entire lumped array
        self.lumped_dict = {
            'stim_num': np.asarray(stim_num_list),
            'dynamic_fr': np.asarray(dynamic_fr_list),
            'static_fr': np.asarray(static_fr_list),
            'displ': np.asarray(displ_list),
            'force': np.asarray(force_list),
            }
        self.lumped_dict_fit = {
            'displ_dynamic': np.polyfit(self.lumped_dict['displ'], 
                self.lumped_dict['dynamic_fr'], deg=1),
            'displ_static': np.polyfit(self.lumped_dict['displ'], 
                self.lumped_dict['static_fr'], deg=1),
            'force_dynamic': np.polyfit(self.lumped_dict['force'], 
                self.lumped_dict['dynamic_fr'], deg=1),
            'force_static': np.polyfit(self.lumped_dict['force'], 
                self.lumped_dict['static_fr'], deg=1),
            }
        self.lumped_median_predict = {
            key: np.polyval(self.lumped_dict_fit[key], 
            globals()['EVAL_'+key[:5].upper()])
            for key in iter(self.lumped_dict_fit)
            }
        return
    
    def generate_binned_exp(self):
        self.binned_exp = {
            'displ_mean': [],
            'displ_std': [],
            'displ_all': [],
            'force_mean': [],
            'force_std': [],
            'force_all': [],
            'static_fr_mean': [],
            'static_fr_std': [],
            'static_fr_all': [],
            'dynamic_fr_mean': [],
            'dynamic_fr_std': [], 
            'dynamic_fr_all': [], 
            }
        for i, stim_group in enumerate(self.stim_group_dict):
            self.binned_exp['displ_mean'].append(stim_group['static_displ'
                ].mean())
            self.binned_exp['displ_std'].append(stim_group['static_displ'
                ].std())
            self.binned_exp['displ_all'].extend(stim_group['static_displ'])
            self.binned_exp['force_mean'].append(stim_group['static_force'
                ].mean())
            self.binned_exp['force_std'].append(stim_group['static_force'
                ].std())
            self.binned_exp['force_all'].extend(stim_group['static_force'])
            self.binned_exp['static_fr_mean'].append(stim_group[
                'static_avg_fr'].mean())
            self.binned_exp['static_fr_std'].append(stim_group[
                'static_avg_fr'].std())
            self.binned_exp['static_fr_all'].extend(stim_group[
                'static_avg_fr'])
            self.binned_exp['dynamic_fr_mean'].append(stim_group[
                'dynamic_avg_fr'].mean())
            self.binned_exp['dynamic_fr_std'].append(stim_group[
                'dynamic_avg_fr'].std())
            self.binned_exp['dynamic_fr_all'].extend(stim_group[
                'dynamic_avg_fr'])
        for key in self.binned_exp.keys():
            if not key.endswith('all') and key is not 'displ_mean':
                self.binned_exp[key] = np.array(self.binned_exp[key])[
                    np.array(self.binned_exp['displ_mean']).argsort()]
            if key.endswith('all'):
                self.binned_exp[key] = np.array(self.binned_exp[key])
        self.binned_exp['displ_mean'] = np.array(sorted(
            self.binned_exp['displ_mean']))
        with open('./pickles/binned_exp_%d.pkl' % self.fiber_id, 'wb') as f:
            pickle.dump(self.binned_exp, f)
        if self.make_plot:
            self.fig_binned_exp, self.axs_binned_exp = plt.subplots(2, 2, 
                figsize=(6.83, 6.83))
            self.axs_binned_exp[0, 0].errorbar(self.binned_exp['displ_mean'], 
                self.binned_exp['static_fr_mean'], 
                self.binned_exp['static_fr_std'], fmt='-k')
            self.axs_binned_exp[0, 1].errorbar(self.binned_exp['displ_mean'], 
                self.binned_exp['dynamic_fr_mean'], 
                self.binned_exp['dynamic_fr_std'], fmt='-k')
            self.axs_binned_exp[1, 0].plot(self.binned_exp['displ_all'],
                self.binned_exp['static_fr_all'], '.k')
            self.axs_binned_exp[1, 1].plot(self.binned_exp['displ_all'],
                self.binned_exp['dynamic_fr_all'], '.k')
            for axes in self.axs_binned_exp[:, 0]:
                axes.set_xlabel(r'Displ ($mu$m)')
                axes.set_ylabel('Static FR (Hz)')
            for axes in self.axs_binned_exp[:, 1]:
                axes.set_xlabel(r'Displ ($mu$m)')
                axes.set_ylabel('Dynamic FR (Hz)')
            self.fig_binned_exp.savefig('./plots/binned_exp_%d.png' % 
                self.fiber_id, dpi=300)
        return
        
    
    def load_fiber_data(self):
        all_data = np.genfromtxt('./cleandata/csvs/static_dynamic.csv', 
                         delimiter=',')
        self.fiber_data = all_data[all_data[:, 0] == self.fiber_id][:, 1:]
        self.stim_num, self.static_displ, self.static_force, \
            self.static_avg_fr, self.dynamic_avg_fr, self.ramp_time = \
            self.fiber_data.T
        self.stim_num = self.stim_num.astype(np.int)
        return
    
    def get_stim_group_num_list(self):
        # Choose features to do the grouping
#        feature_unscaled = np.c_[self.static_displ, self.static_force
#                                 self.ramp_time]
#        feature_unscaled = np.c_[self.static_displ, self.static_force]
        feature_unscaled = np.c_[self.static_displ]
        feature = StandardScaler().fit_transform(feature_unscaled)
        db = DBSCAN(eps=.3, min_samples=2).fit(feature)
        self.stim_group_num_list = db.labels_.astype(np.int)
        self.unique_labels = set(self.stim_group_num_list)                
        if self.make_plot: # Plot out the grouping
            self.fig_grouping, self.axs_grouping = plt.subplots(2, 1, 
                figsize=(3.27, 6))
            colors = plt.cm.get_cmap('Spectral')(np.linspace(0, 1, len(
                self.unique_labels)))
            for k, col in zip(self.unique_labels, colors):
                if k == -1:
                    col = 'k'
                class_members = [index[0] for index in np.argwhere(
                    self.stim_group_num_list == k)]
                for index in class_members:
                    feature_row = feature_unscaled[index]
                    self.axs_grouping[0].plot(feature_row[0], feature_row[0],
                        'o', markerfacecolor=col)
                    self.axs_grouping[1].plot(feature_row[0], feature_row[0],
                        'o', markerfacecolor=col)
            self.axs_grouping[0].set_xlabel(r'Displ ($\mu$m)')
            self.axs_grouping[0].set_ylabel(r'Force (mN)')
            self.axs_grouping[1].set_xlabel(r'Force (mN)')
            self.axs_grouping[1].set_ylabel(r'Ramp time (ms)')
            self.fig_grouping.tight_layout()
            self.fig_grouping.savefig('./plots/grouping_%d.png' % 
                self.fiber_id, dpi=300)
        return
    
    def get_stim_group(self):
        # Total amount of groups
        if -1 in self.unique_labels:
            stim_group_list = [[] for i in range(
                len(self.unique_labels)-1)]
        else:
            stim_group_list = [[] for i in range(
                len(self.unique_labels))]
        for i, stim_group_num in enumerate(self.stim_group_num_list):
            if stim_group_num != -1:
                stim_group_list[stim_group_num].append(self.fiber_data[i])
        self.stim_group_dict = [[] for i in range(
            self.stim_group_num_list.max()+1)]
        for i, stim_group in enumerate(stim_group_list):
            stim_group_list[i] = np.array(stim_group)
            self.stim_group_dict[i] = {
                'stim_num': stim_group_list[i][:, 0].astype(np.int),
                'static_displ': stim_group_list[i][:, 1],
                'static_force': stim_group_list[i][:, 2],
                'static_avg_fr': stim_group_list[i][:, 3],
                'dynamic_avg_fr': stim_group_list[i][:, 4],
                'ramp_time': stim_group_list[i][:, 5],
                }
        # To sort the stim groups
        displ_array = np.array([self.stim_group_dict[i]['static_displ'].mean()
            for i in range(len(self.stim_group_dict))])
        ordered_stim_group = [[] for i in range(
            self.stim_group_num_list.max()+1)]
        for i in range(len(ordered_stim_group)):
            ordered_stim_group[i] = self.stim_group_dict[
                displ_array.argsort()[i]]
        self.stim_group_dict = ordered_stim_group
        return
    
    def get_fem_displ_by_force(self, force):
        return np.interp(force, self.abq_force, self.abq_displ) * 1e-3
    
    def get_fem_displ_by_displ(self, displ):
        return (displ - self.displ_coeff[0]) / self.displ_coeff[1] * 1e-3
    
    def get_fem_ramp_time(self, ramp_time):
        return ramp_time / self.displ_coeff[1]
    
    def get_fem_displ_ramp_time(self, match='displ', stim_group_dict=None,
        ramp_time_match_experiment=False):
        # By default, refer to self
        if stim_group_dict is None:
            stim_group_dict = self.stim_group_dict
        fem_displ_ramp_time = [{} for i in range(len(stim_group_dict))]
        for i, stim_group in enumerate(stim_group_dict):
            if match is 'displ':
                fem_displ_ramp_time[i]['fem_displ'] = \
                    self.get_fem_displ_by_displ(stim_group['static_displ'])
            if match is 'force':
                fem_displ_ramp_time[i]['fem_displ'] = \
                    self.get_fem_displ_by_force(stim_group['static_force'])
            if ramp_time_match_experiment:
                fem_displ_ramp_time[i]['fem_ramp_time'] = \
                    self.get_fem_ramp_time(stim_group['ramp_time'])
            else:
                fem_displ_ramp_time[i]['fem_ramp_time'] = \
                    self.get_fem_ramp_time(np.polyval(self.ramp_time_coeff, 
                        stim_group['static_displ']))
        return fem_displ_ramp_time

    def get_displ_coeff(self):
        
        def get_r2(a, abq_force, abq_displ, static_force, exp_displ, sign=1.):
            abq_displ_scaled = a[0] + a[1] * abq_displ 
            p = np.polyfit(abq_displ_scaled, abq_force, 3)
            abq_force_interp = np.polyval(p, exp_displ)
            sst = static_force.var() * static_force.shape[0]
            sse = np.linalg.norm(static_force - abq_force_interp) ** 2
            r2 = 1 - sse / sst
            return sign * r2
        
        self.abq_displ, self.abq_force = np.genfromtxt(
            'x:/WorkFolder/AbaqusFolder/YoshiModel/csvs/FitFemDisplForce.csv', 
            delimiter=',').T
        self.abq_displ *= 1e6
        self.abq_force *= 1e3
        bounds = ((0., 1000.), (0, 5))
        res = minimize(get_r2, [250., 2.], args=(self.abq_force, 
                       self.abq_displ, self.static_force, self.static_displ, 
                       -1.), method='L-BFGS-B', bounds=bounds)
        self.displ_coeff = res.x
        self.displ_coeff_r2 = -res.fun
        # Make the plot
        self.abq_displ_scaled = res.x[0] + res.x[1] * self.abq_displ 
        if self.make_plot:
            self.fig_displ, self.axs_displ = plt.subplots()
            self.axs_displ.plot(self.static_displ, self.static_force, '.k')
            self.axs_displ.plot(self.abq_displ_scaled, self.abq_force, '-r')
            self.axs_displ.set_xlim(right=self.static_displ.max()*1.2)
            self.axs_displ.set_ylim(top=self.static_force.max()*1.2)
            self.axs_displ.set_xlabel(r'Displ. ($\mu$m)')
            self.axs_displ.set_ylabel(r'Force (mN)')
            self.fig_displ.savefig('./plots/displ_%d.png' % self.fiber_id, 
                                   dpi=300)
        return
    
    def get_ramp_time_coeff(self):
        self.ramp_time_coeff = np.polyfit(self.static_displ, self.ramp_time, 
                                          1)
        if self.make_plot:
            self.fig_ramp_time, self.axs_ramp_time = plt.subplots()
            self.axs_ramp_time.plot(self.static_displ, self.ramp_time, '.k')
            self.axs_ramp_time.plot(self.static_displ, np.polyval(
                self.ramp_time_coeff, self.static_displ), '-r')
#            self.axs_ramp_time.set_xlim(left=0)
            self.axs_ramp_time.set_xlabel(r'Displ. ($\mu$m)')
            self.axs_ramp_time.set_ylabel(r'Ramp time (s)')
            self.fig_ramp_time.savefig('./plots/ramp_time_%d.png' % 
                self.fiber_id, dpi=300)
        return
    
    def generate_script(self):
        with open('x:/WorkFolder/AbaqusFolder/YoshiModel/fittemplate.py', 'r'
            ) as f:
            template_script = f.read()
        self.abq_script = template_script.replace('CSVFILEPATH', 
            '\'x:/WorkFolder/DataAnalysis/YoshiRecordingData/csvs/stim_block_'
            +str(self.fiber_id)+'.csv\'').replace('BASEMODELNAME', '\'Fiber'
            +str(self.fiber_id)+'\'')
        with open('./scripts/'+str(self.fiber_id)+'.py', 'w') as f:
            f.write(self.abq_script)
        return
    
    def run_script(self):
        os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae script=x:/WorkFolder/DataAnalysis/YoshiRecordingData/scripts/%d.py' % self.fiber_id)
        return

    def generate_stim_block_array(self, stim_group_dict=None, fiber_id=None):
        fem_displ_ramp_time_list = self.get_fem_displ_ramp_time(
            stim_group_dict=stim_group_dict)
        self.stim_block_array = [[fem_displ_ramp_time['fem_ramp_time'].mean(), 
            fem_displ_ramp_time['fem_displ'].mean()] 
            for fem_displ_ramp_time in fem_displ_ramp_time_list]
        self.stim_block_array = np.array(self.stim_block_array)
        if fiber_id is None:
            fiber_id = self.fiber_id
        np.savetxt('./csvs/stim_block_%d.csv' % fiber_id, 
                   self.stim_block_array, delimiter=',')
        return
    
    def plot_force_trace_fitting(self, axes):
        self.get_stim_block_trace_exp()
        self.get_stim_block_trace_fem()
        for i, stim_group in enumerate(self.stim_group_dict):
            for j, stim_num in enumerate(stim_group['stim_num']):
                axes.plot(self.stim_group_dict[i]['traces_exp']
                    [j]['time'], self.stim_group_dict[i]['traces_exp'][j][
                    'force'], '.', color='.5')
        for i, stim_group in enumerate(self.stim_group_dict):
            axes.plot(self.stim_group_dict[i]['traces_fem'][
                'time'], self.stim_group_dict[i]['traces_fem']['force']*1e3, 
                '-k')
        axes.set_xlabel('Time (s)')
        axes.set_ylabel('Force (mN)')
        return
    
    def get_stim_block_trace_exp(self):
        with open('./cleandata/finaldata/cleanFiberList.pkl', 'rb') as f:
            traces_exp = pickle.load(f)[self.fiber_id].traces
        for i, stim_group in enumerate(self.stim_group_dict):
            self.stim_group_dict[i]['traces_exp'] = []
            for j, stim_num in enumerate(stim_group['stim_num']):
                self.stim_group_dict[i]['traces_exp'].append(
                    traces_exp[stim_num])
        return
        
    def get_stim_block_trace_fem(self):
        for i, stim_group in enumerate(self.stim_group_dict):
            file_path = 'x:/WorkFolder/AbaqusFolder/YoshiModel/csvs/Fiber' +\
                str(self.fiber_id) + 'Output' + str(i) + '.csv'
            time, force, displ, stress, strain, sener = np.genfromtxt(
                file_path, delimiter=',').T
            max_force_time = time[force.argmax()]
            time_shift = stim_group['ramp_time'].mean() - max_force_time
            time += time_shift
            # Perform linear interpolation on all traces
            time_fine = np.arange(0, time.max(), DT)
            force_fine = np.interp(time_fine, time, force)
            displ_fine = np.interp(time_fine, time, displ)
            stress_fine = np.interp(time_fine, time, stress)
            strain_fine = np.interp(time_fine, time, strain)
            sener_fine = np.interp(time_fine, time, sener)
            self.stim_group_dict[i]['traces_fem'] = {
                'time': time_fine,
                'force': force_fine,
                'displ': displ_fine,
                'stress': stress_fine,
                'strain': strain_fine,
                'sener': sener_fine,
                }
            self.stim_group_dict[i]['traces_fem']['max_index'] = \
                self.stim_group_dict[i]['traces_fem']['force'].argmax()
        return



if __name__ == '__main__':
    # Decide whether we want to run all the FEA this time!
    run_calibration = False
    make_plot = False
    run_fiber_mech = False
    run_each_fiber = False
    run_fitting = False
    # Run calibration
    if run_calibration:
        os.system('call \"C:/SIMULIA/Abaqus/Commands/abaqus.bat\" cae script=x:/WorkFolder/AbaqusFolder/YoshiModel/calibration.py')
    # Real coding starts here!
    fiber_list = []
    for i in range(FIBER_TOT_NUM):
        fiber = Fiber(i, make_plot=make_plot)
        fiber_list.append(fiber)
    plt.close('all')
    fiber_mech = fiber_list[FIBER_MECH_ID]
    # Save fiber_mech's fem displ - ramp_time coeff, and displcoeff.
    displtimecoeff = np.polyfit(fiber_mech.stim_block_array[:, 1], 
        fiber_mech.stim_block_array[:, 0], 1)
    np.savetxt('X:/WorkFolder/AbaqusFolder/YoshiModel/csvs/displtimecoeff.csv'
        , displtimecoeff, delimiter=',')
    np.savetxt('./csvs/displcoeff.csv', fiber_mech.displ_coeff, delimiter=',')
    # To plot the exact fit to force trace
    if run_fiber_mech:
        fiber_mech.generate_script()
        fiber_mech.run_script()
        fig, axs = plt.subplots()
        fiber_mech.plot_force_trace_fitting(axs)
        fig.savefig('./plots/fitting_%d.png' % fiber_mech.fiber_id, dpi=300)
    # Initialize fiber.lif_r2 and fiber.lif_fr
    for fiber in fiber_list:
        fiber.lif_r2 = {
            quantity: (.0, .0) for quantity in quantity_name_list
            }
        fiber.lif_fr = {
            quantity: np.zeros([len(fiber.stim_group_dict), 2])
            for quantity in quantity_name_list
            }
    # Run the Abaqus model
    for fiber in fiber_list:
        if fiber.fiber_id in FIBER_FIT_ID_LIST:
            fiber_mech.generate_stim_block_array(
                stim_group_dict=fiber.stim_group_dict, fiber_id=fiber.fiber_id)
            if run_each_fiber:
                fiber.generate_script()
                fiber.run_script()
            # Read the FEM outputs
            fiber.get_stim_block_trace_fem()
            # Construct the FEM output quantity data
            fiber.trans_param, fiber.lif_fr, fiber.lif_r2 = {}, {}, {}
            for quantity_name in quantity_name_list:
                quantity_array_list = [fiber.stim_group_dict[i]['traces_fem'][
                    quantity_name] for i in range(len(fiber.stim_group_dict))]
                max_index_list = [fiber.stim_group_dict[i]['traces_fem'][
                    'max_index'] for i in range(len(fiber.stim_group_dict))]
                quantity_dict_list = [{} for i in range(len(max_index_list))]
                for i in range(len(max_index_list)):
                    quantity_dict_list[i]['quantity_array'] = quantity_array_list[
                        i]
                    quantity_dict_list[i]['max_index'] = max_index_list[i]
                    quantity_dict_list[i]['max_index'] = max_index_list[i]
                # Perform the fitting for diff-form
                from fitlif import (fit_trans_param, 
                    trans_param_to_predicted_fr, get_lstsq_fit)
                target_fr_array = np.c_[fiber.lumped_dict['stim_num'],
                    fiber.lumped_dict['static_fr'], 
                    fiber.lumped_dict['dynamic_fr'],
                    ]
                if run_fitting:
                    fiber.trans_param[quantity_name] = fit_trans_param(
                        quantity_dict_list, target_fr_array)
#                    fiber.trans_param[quantity_name] = get_lstsq_fit(
#                        quantity_dict_list, target_fr_array)
                    with open('./pickles/trans_params_%d.pkl'%fiber.fiber_id,
                        'wb') as f:
                        pickle.dump(fiber.trans_param, f)
                else:
                    with open('./pickles/trans_params_%d.pkl'%fiber.fiber_id,
                        'rb') as f:
                        fiber.trans_param = pickle.load(f)
                fiber.lif_fr[quantity_name] = trans_param_to_predicted_fr(
                    quantity_dict_list, fiber.trans_param[quantity_name])[:,
                    1:]
                def get_lif_r2(target_fr_array, predicted_fr):
                    predicted_fr_array = np.empty_like(target_fr_array)
                    for i in range(target_fr_array.shape[0]):
                        predicted_fr_array[i, 0] = i
                        predicted_fr_array[i, 1] = predicted_fr[int(
                            target_fr_array[i, 0]), 0]
                        predicted_fr_array[i, 2] = predicted_fr[int(
                            target_fr_array[i, 0]), 1]
                    sstot_static = target_fr_array[:, 1].var(
                        ) * target_fr_array.shape[0]
                    sstot_dynamic = target_fr_array[:, 2].var(
                        ) * target_fr_array.shape[0]
                    ssres_static = (((target_fr_array-predicted_fr_array)[:, 1])
                        **2).sum()
                    ssres_dynamic = (((target_fr_array-predicted_fr_array)[:, 2])
                        **2).sum()
                    static_r2 = 1. - ssres_static / sstot_static
                    dynamic_r2 = 1. - ssres_dynamic / sstot_dynamic
                    return static_r2, dynamic_r2
                fiber.lif_r2[quantity_name] = get_lif_r2(target_fr_array,
                    fiber.lif_fr[quantity_name])
                fiber.df_lif_r2 = pd.DataFrame(fiber.lif_r2)
    #%% Plot fitting figure for paper
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 6.83))
    for fiber_id in FIBER_FIT_ID_LIST:
        fiber = fiber_list[fiber_id]
        fmt = MARKER_LIST[fiber_id]
#        color = COLOR_LIST[fiber_id]
        color = 'k'
        # Plot experiment
        axs[0].errorbar(fiber.binned_exp['displ_mean'], fiber.binned_exp[
            'dynamic_fr_mean'], fiber.binned_exp['dynamic_fr_std'],
            fmt=fmt, c=color, mec=color, ms=MS, label='Experiment')
        axs[1].errorbar(fiber.binned_exp['displ_mean'], fiber.binned_exp[
            'static_fr_mean'], fiber.binned_exp['static_fr_std'],
            fmt=fmt, c=color, mec=color, ms=MS, label='Experiment')
    # Plot fitting
    for quantity_id, quantity in enumerate(['stress', 'strain', 'sener']):            
        ls = LS_LIST[quantity_id]
        if fiber_id in FIBER_FIT_ID_LIST:
            axs[0].plot(fiber.binned_exp['displ_mean'], fiber.lif_fr[
                quantity][:, 1], c=color, ls=ls, label='Predicted by ' + \
                quantity)
            axs[1].plot(fiber.binned_exp['displ_mean'], fiber.lif_fr[
                quantity][:, 0], c=color, ls=ls, label='Predicted by' + \
                quantity)
    # Adjust formatting
    for axes in axs:
        axes.set_xlim(390, 550)
    axs[1].set_xlabel(r'Displacement ($\mu$m)')
    axs[0].set_ylabel('Dynamic mean firing (Hz)')
    axs[1].set_ylabel('Static mean firing (Hz)')
    h, l = axs[0].get_legend_handles_labels()
    legend = fig.legend(h, l, bbox_to_anchor=(0.05, 0.85, 0.9, .14), ncol=2,
        mode='expand', frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(.5)
    # Adding panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.savefig('./plots/paper_plot_fitting.png', dpi=300)
    #%% Plot force-displ fitting
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 5))
    for fiber_id in FIBER_FIT_ID_LIST:
        # Plot static force/displ
        fiber = fiber_list[fiber_id]
        fmt = MARKER_LIST[fiber_id]
        color = 'k'
        axs[0].errorbar(fiber.binned_exp['displ_mean'], fiber.binned_exp[
            'force_mean'], fiber.binned_exp['force_std'],
            fmt=fmt, c=color, mec=color, ms=MS, label='Experiment')
        axs[0].plot(fiber.abq_displ_scaled, fiber.abq_force, ls='-', c=color,
            label='Model')
        # Plot force trace
        group_id = 1
        fiber_mech.get_stim_block_trace_exp()
        stim_group = fiber_mech.stim_group_dict[group_id]
        for i, stim_num in enumerate(stim_group['stim_num']):
            axs[1].plot(stim_group['traces_exp'][i]['time'], 
                stim_group['traces_exp'][i]['force'], '.', color='.5')
        axs[1].get_lines()[0].set_label('Experiment')
        axs[1].plot(stim_group['traces_fem']['time'], stim_group['traces_fem']
                ['force']*1e3, '-k', label='Model')
    axs[0].legend(loc=2)
    axs[0].set_xlabel(r'Static displacement ($\mu$m)')
    axs[0].set_ylabel(r'Static force (mN)')
    axs[1].legend()
    axs[1].set_xlabel(r'Time (s)')
    axs[1].set_ylabel(r'Force (mN)')
    # Setting the range
    axs[0].set_xlim(375, 575)    
    axs[0].set_ylim(0, 10)
    axs[1].set_ylim(-1, 5)
    # Adding panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')    
    fig.tight_layout()
    fig.savefig('./plots/paper_plot_fitting_mechanical.png', dpi=300)
    #%% Plot model figure for paper
    # Plot force traces
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 5))
    i = 2
    fiber_mech.get_stim_block_trace_exp()
    stim_group = fiber_mech.stim_group_dict[i]
    for j, stim_num in enumerate(stim_group['stim_num']):
        axs[0].plot(stim_group['traces_exp'][j]['time'], 
            stim_group['traces_exp'][j]['force'], '.', color='.5')
    axs[0].plot(stim_group['traces_fem']['time'], stim_group['traces_fem'][
        'force']*1e3, '-k')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Force (mN)')
    axs[0].set_xlim(-.5, 6.5)
    # Get raw firing data, from group 1, picking the 0th trace - arbitrarily
    # picked
    group_num = 2
    run_num = 0
    # Get raw spike data
    raw_spike = fiber_mech.stim_group_dict[group_num]['traces_exp'][run_num
        ]['raw_spike']
#    raw_spike = (raw_spike - raw_spike.mean()) / raw_spike.max() / 2
    scale = 250.
    raw_spike /= scale
    raw_spike_time = np.arange(raw_spike.size) / 16e3
    # Start plotting
    axs[1].plot(raw_spike_time, raw_spike, '-k')
    axs[1].set_frame_on(False)
    axs[1].get_yaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks([])
    axs[1].set_ylim(-1., 1.)
    axs[1].set_xlim(-.5, 6.5)
    # Draw the static / dynamic phase bars
    ypos = .75
    start_time = raw_spike_time[(raw_spike>.25).nonzero()[0][0]]
    max_time = fiber_mech.stim_group_dict[group_num]['traces_exp'][run_num][
        'force'].argmax()/16e3
    axs[1].plot([start_time, max_time], [ypos, ypos], '-k', lw=4.)
    axs[1].plot([max_time+STATIC_START, max_time+STATIC_END], [ypos, ypos],
        '-k', lw=4.)
    axs[1].text(start_time+(max_time-start_time)/2, ypos+.1, 'Dynamic', 
        transform=axs[1].transData, fontsize=10, ha='center')
    axs[1].text(max_time+STATIC_START+(STATIC_END-STATIC_START)/2, ypos+.1, 
        'Static', transform=axs[1].transData, fontsize=10, ha='center')        
    # Draw a scale bar
    xpos = 5
    ypos = -0.8
    ylen = 0.3
    axs[1].plot([xpos, xpos+1], [ypos, ypos], '-k', lw=4.)
    axs[1].plot([xpos, xpos], [ypos, ypos+ylen], '-k', lw=4.)
    axs[1].text(xpos+0.5, ypos-.15, '1 s', transform=axs[1].transData, 
        fontsize=10, ha='center')
    axs[1].text(xpos-.5, ypos, '%d mV'%int(ylen*scale), transform=axs[1].
        transData, fontsize=10, rotation='vertical', va='bottom')
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.15, 1.05, chr(65+axes_id), transform=axes.transAxes,
            fontsize=12, fontweight='bold', va='top')    
    fig.tight_layout()
    fig.savefig('./plots/paper_plot_time_series.png', dpi=300)
    #%% Plot experiment data with displ / force aligned - static, separate
    # Gather data for fitting
    displ_list, force_list, static_fr_list, dynamic_fr_list = [], [], [], []
    for fiber_id, fiber in enumerate(fiber_list):
        displ_list.extend(fiber.binned_exp['displ_mean'])
        force_list.extend(fiber.binned_exp['force_mean'])
        static_fr_list.extend(fiber.binned_exp['static_fr_mean'])
        dynamic_fr_list.extend(fiber.binned_exp['dynamic_fr_mean'])
#        displ_list.extend(fiber.lumped_dict['displ'])
#        force_list.extend(fiber.lumped_dict['force'])
#        static_fr_list.extend(fiber.lumped_dict['static_fr'])
#        dynamic_fr_list.extend(fiber.lumped_dict['dynamic_fr'])
    # Perform fitting
    displ_dynamic_fit_param = np.polyfit(displ_list, dynamic_fr_list, 1)
    force_dynamic_fit_param = np.polyfit(force_list, dynamic_fr_list, 1)
    displ_static_fit_param = np.polyfit(displ_list, static_fr_list, 1)
    force_static_fit_param = np.polyfit(force_list, static_fr_list, 1)
    displ_dynamic_predict = np.polyval(displ_dynamic_fit_param, displ_list)
    force_dynamic_predict = np.polyval(force_dynamic_fit_param, force_list)
    displ_static_predict = np.polyval(displ_static_fit_param, displ_list)
    force_static_predict = np.polyval(force_static_fit_param, force_list)
    # Calculate residual variance
    displ_dynamic_fit_res = displ_dynamic_predict - np.asarray(dynamic_fr_list)
    force_dynamic_fit_res = force_dynamic_predict - np.asarray(dynamic_fr_list)
    displ_static_fit_res = displ_static_predict - np.asarray(static_fr_list)
    force_static_fit_res = force_static_predict - np.asarray(static_fr_list)
    displ_dynamic_fit_resvar = displ_dynamic_fit_res.var()
    force_dynamic_fit_resvar = force_dynamic_fit_res.var()
    displ_static_fit_resvar = displ_static_fit_res.var()
    force_static_fit_resvar = force_static_fit_res.var()
    displ_dynamic_fit_resstd = displ_dynamic_fit_res.std()
    force_dynamic_fit_resstd = force_dynamic_fit_res.std()
    displ_static_fit_resstd = displ_static_fit_res.std()
    force_static_fit_resstd = force_static_fit_res.std()
    def get_r2(exp, mod):
        exp = np.asarray(exp)
        ssres = ((exp - mod)**2).sum()
        sstot = exp.var() * exp.size
        r2 = 1. - ssres / sstot
        return r2
    displ_dynamic_fit_r2 = get_r2(dynamic_fr_list, displ_dynamic_predict)
    force_dynamic_fit_r2 = get_r2(dynamic_fr_list, force_dynamic_predict)
    displ_static_fit_r2 = get_r2(static_fr_list, displ_static_predict)
    force_static_fit_r2 = get_r2(static_fr_list, force_static_predict)
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 9.19))
    for i, fiber in enumerate(fiber_list):
        fmt = MARKER_LIST[i] + ':'
        color = 'k'
        axs[0].errorbar(fiber.binned_exp['displ_mean'], fiber.binned_exp[
            'force_mean'], fiber.binned_exp['force_std'], fmt=fmt, c=color,
            mec=color, ms=MS, label='Fiber #%d' % i)        
        axs[1].errorbar(fiber.binned_exp['displ_mean'], fiber.binned_exp[
            'static_fr_mean'], fiber.binned_exp['static_fr_std'], fmt=fmt,
            c=color, mec=color, ms=MS, label='Fiber #%d' % i)
        axs[2].errorbar(fiber.binned_exp['force_mean'], fiber.binned_exp[
            'static_fr_mean'], fiber.binned_exp['static_fr_std'], fmt=fmt,
            c=color, mec=color, ms=MS, label='Fiber #%d' % i)
    axs[1].plot(sorted(displ_list), np.sort(displ_static_predict), '-k',
        label='Linear regression')
    axs[2].plot(sorted(force_list), np.sort(force_static_predict), '-k',
        label='Linear regression')
    axs[0].set_xlabel(r'Static displ. ($\mu$m)')
    axs[1].set_xlabel(r'Static displ. ($\mu$m)')
    axs[2].set_xlabel(r'Static force (mN)')    
    axs[0].set_ylabel(r'Static force (mN)')
    axs[1].set_ylabel('Mean static FR (Hz)')
    axs[2].set_ylabel('Mean static FR (Hz)')
    axs[0].legend(loc=2)
    fig.tight_layout()
    fig.savefig('./plots/compare_variance.png', dpi=300)
#    print(force_static_fit_resvar, displ_static_fit_resvar)