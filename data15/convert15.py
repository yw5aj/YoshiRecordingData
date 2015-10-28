# Import installed packages
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from lmfit import Model
import pandas as pd
from collections import defaultdict

# Import my own packages
MARKER_LIST = ['v', 'D', 'o', 's', '*', 'h', '.', 'x', 'h', '+']
COLOR_LIST = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b']
LS_LIST = ['-', '--', '-.', ':']
MS = 6
START_STATIC = 0
END_STATIC = 5
rep_stim_id_dict = {
    '2014-07-11-01': 20,
    '2014-07-11-02': 1,
    '2013-12-07-01': 11,
    '2013-12-21-01': 7,
    '2014-01-16-01': 7,
    }
rep_stim_id_dict = defaultdict(lambda: 0, rep_stim_id_dict)
rep_spike_id_dict = {
    '2014-07-11-01': 0,
    '2014-07-11-02': 0,
    '2013-12-07-01': 0,
    '2013-12-21-01': 0,
    '2014-01-16-01': 0,
    }
rep_spike_id_dict = defaultdict(lambda: 0, rep_spike_id_dict)


# New indenter list
NEW_INDENTER = ['2014-07-11-01', '2014-07-11-02']


class CleanFiber:

    def __init__(self, mat_filename, mat_pathname, make_plot=False):
        self.mat_filename = mat_filename
        self.mat_pathname = mat_pathname
        if self.mat_filename[:13] in NEW_INDENTER:
            self.fs = int(20e3)
        else:
            self.fs = int(16e3)
        self.get_animal_info()
        self.get_mat_data()
        self.sort_traces()
        self.find_contact_by_spike()
        if not np.isnan(self.contact_pos):
            self.cut_traces(make_plot=make_plot)
        return

    def get_animal_info(self):
        self.animal_info = {'date_str': self.mat_filename[:10],
                            'unitid': int(self.mat_filename[11:13])}
        return self.animal_info

    def get_mat_data(self):
        self.mat_data = loadmat(
            os.path.join(self.mat_pathname, self.mat_filename))
        self.stim_block = []
        i = 0
        while True:
            try:
                self.stim_block.append(
                    self.mat_data[
                        'COUT_PUT_F' + str(self.animal_info['unitid']) + '0' +
                        str(i + 1)].shape[1])
                i += 1
            except:
                break
        return self.mat_data, self.stim_block

    def sort_traces(self):
        b, a = butter(8, 250 / self.fs)  # Fs = 250 Hz
        self.traces_full = []
        for i, j_num in enumerate(self.stim_block):
            for j in range(j_num):
                traces_dict = {
                    'raw_force': self.mat_data[
                        'COUT_PUT_F' + str(self.animal_info['unitid']) + '0' +
                        str(i + 1)][:, j],
                    'raw_displ': self.mat_data[
                        'COUT_PUT_D' + str(self.animal_info['unitid']) + '0' +
                        str(i + 1)][:, j],
                    'spike_trace': self.mat_data[
                        'OUT_PUT_CS' + str(self.animal_info['unitid']) + '0' +
                        str(i + 1)][:, j],
                    'time': np.arange(
                        self.mat_data['COUT_PUT_F' + str(
                            self.animal_info['unitid']) + '0' +
                            str(i + 1)].shape[0]) / self.fs}
                try:
                    traces_dict['raw_spike'] = self.mat_data[
                        'OUT_PUT_R' + str(self.animal_info['unitid']) + '0' +
                        str(i + 1)][:, j]
                except KeyError:
                    pass
                traces_dict['force'] = filtfilt(b, a, traces_dict['raw_force'])
                traces_dict['displ'] = filtfilt(b, a, traces_dict['raw_displ'])
                traces_dict['displ'] = traces_dict[
                    'displ'] - traces_dict['displ'][0]
                # Organize the spike firing rate and timings
                if np.any(traces_dict['spike_trace']):
                    traces_dict['spike_time'], traces_dict['spike_isi'],\
                        traces_dict['spike_fr'] = self._get_spike_data(
                            traces_dict['spike_trace'])
                    traces_dict['spike_fr_trace'] = np.interp(
                        traces_dict['time'], traces_dict['spike_time'],
                        traces_dict['spike_fr'])
                else:
                    traces_dict['spike_fr_trace'] = np.zeros(
                        traces_dict['time'].shape)
                self.traces_full.append(traces_dict)
        return self.traces_full

    def _get_spike_data(self, spike_trace):
        spike_index = np.nonzero(spike_trace)[0]
        spike_time = spike_index / self.fs
        spike_isi = np.r_[np.inf, np.diff(spike_time)]
        spike_fr = 1. / spike_isi
        return spike_time, spike_isi, spike_fr

    def find_contact_by_spike(self):
        rep_stim_id = rep_stim_id_dict[self.mat_filename[:13]]
        rep_spike_id = rep_spike_id_dict[self.mat_filename[:13]]
        stim_traces_full = self.traces_full[rep_stim_id]
        contact_index = stim_traces_full['spike_trace'].nonzero(
            )[0][rep_spike_id]
        self.contact_pos = stim_traces_full['displ'][contact_index]
        return self.contact_pos

    def cut_traces(self, make_plot=False):
        self.traces = []
        for stim_id, stim_traces_full in enumerate(self.traces_full):
            contact_index = np.nonzero(
                stim_traces_full['displ'] >= self.contact_pos)[0][0]
            max_force_index = stim_traces_full['force'].argmax()
            trace = {
                'force': stim_traces_full['force'][
                    contact_index:contact_index + 6 * self.fs],
                'displ': stim_traces_full['displ'][
                    contact_index:contact_index + 6 * self.fs] -
                stim_traces_full['displ'][contact_index],
                'spike_trace': stim_traces_full['spike_trace'][
                    contact_index:contact_index + 6 * self.fs],
                'time': np.arange(6 * self.fs) / self.fs,
                'ramp_time': (max_force_index - contact_index) / self.fs,
                'static_force': stim_traces_full['force'][
                    max_force_index + START_STATIC * self.fs:max_force_index +
                    int(END_STATIC * self.fs)].mean(),
                'static_displ': stim_traces_full['displ'][
                    max_force_index + START_STATIC * self.fs:max_force_index +
                    int(END_STATIC * self.fs)].mean() - stim_traces_full[
                        'displ'][contact_index],
                'dynamic_force_rate': np.diff(
                    stim_traces_full['force'][
                        contact_index:max_force_index]).mean() * self.fs,
                'dynamic_displ_rate': np.diff(
                    stim_traces_full['displ'][
                        contact_index:max_force_index]).mean() * self.fs}
            try:
                trace['raw_spike'] = stim_traces_full[
                    'raw_spike'][contact_index:contact_index + 6 * self.fs]
            except KeyError:
                pass
            trace['static_avg_fr'] = self._get_avg_fr(
                stim_traces_full['spike_trace'][
                    max_force_index + int(START_STATIC * self.fs):
                    max_force_index + int(
                        END_STATIC * self.fs)])
            trace['dynamic_avg_fr'] = self._get_avg_fr(
                stim_traces_full['spike_trace'][contact_index:max_force_index])
            if np.any(trace['spike_trace']):
                trace['spike_time'], trace['spike_isi'],\
                    trace['spike_fr'] = self._get_spike_data(
                        trace['spike_trace'])
            if make_plot:
                fig, axs = plt.subplots(3, 1, figsize=(6.83, 9.19))
                for i, item in enumerate(['displ', 'force', 'spike_trace']):
                    axs[i].plot(
                        stim_traces_full['time'],
                        stim_traces_full[item], '-k', color='.0')
                    axs[i].axvline(contact_index / self.fs, color='.0')
                    axs[i].axvline(contact_index / self.fs + END_STATIC,
                                   color='.0')
                    axs[i].set_xlabel('Time (s)')
                axs[0].set_title(
                    'displ = %f, force = %f, missed spikes = %d' % (
                        trace['static_displ'],
                        trace['static_force'],
                        stim_traces_full['spike_trace'][:contact_index].sum()))
                axs[0].set_ylabel('Displ. ($\mu$m)')
                axs[1].set_ylabel('Force (mN)')
                axs[2].set_ylabel('Spikes')
                axs[2].set_ylim(-2, 2)
                fig.savefig(
                    './plots/repsample/traces/' + self.mat_filename[:-4] +
                    'stim' + str(stim_id) + '.png', dpi=300)
            # Clean the traces with negative displacements
            if trace['static_displ'] > 0:
                self.traces.append(trace)
        # Clear all unclosed plots to save memory
        plt.close('all')
        return self.traces

    def _get_avg_fr(self, spike_trace):
        spike_index = np.nonzero(spike_trace)[0]
        spike_count = spike_index.shape[0]
        if spike_count > 1:
            spike_duration = (spike_index[-1] - spike_index[0]) / self.fs
            avg_fr = (spike_count - 1) / spike_duration
        else:
            avg_fr = 0.
        return avg_fr


def sigmoid(x, a, b, c):
    return a / (1 + np.exp(b * (c - x)))


def linear(x, a, b):
    return a * x + b


def get_resvar(x, y, mod='sigmoid'):
    assert len(x) == len(y)
    if len(x) < 3 or mod == 'linear':
        sigmoidmod = Model(linear)
        modfit = sigmoidmod.fit(y, x=x,
                                a=(y.max() - y.min()) / (x.max() - x.min()),
                                b=y.min())
    else:
        sigmoidmod = Model(sigmoid)
        modfit = sigmoidmod.fit(y, x=x, a=y.max(), b=1 / (x.max() - x.min()),
                                c=x.max())
    finex = np.linspace(x.min(), x.max(), 50)
    result = {
        'mod': mod,
        'params': modfit.best_values,
        'resvar': modfit.residual.var(),
        'y': modfit.best_fit,
        'finex': finex,
        'finey': modfit.eval(x=finex)}
    return result


def plot_specific_fibers(fiber_list, fname='temp'):
    if not isinstance(fiber_list, list):
        fiber_list = [fiber_list]
    fig, axs, static_dynamic_array = plot_static_dynamic(
        fiber_list, save_data=False, fname=fname)
    return fig, axs, static_dynamic_array


def plot_static_dynamic(cleanFiber_list, save_data=False,
                        fname='static_dynamic'):
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 6.83))
    static_dynamic_list = [[] for i in range(cleanFiber_list[-1].fiber_id + 1)]
    fmt_list = ['*', 'D', 'v', 's', '.', 'o', '.', 'x', 'h', '+']
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
    for cleanFiber in cleanFiber_list:
        fiber_id = cleanFiber.fiber_id
        static_displ_list, static_force_list, static_avg_fr_list,\
            dynamic_avg_fr_list, dynamic_ramp_time_list = [], [], [], [], []
        dynamic_displ_rate_list, dynamic_force_rate_list = [], []
        for stim_id, stim_traces in enumerate(cleanFiber.traces):
            static_displ_list.append(stim_traces['static_displ'])
            static_force_list.append(stim_traces['static_force'])
            static_avg_fr_list.append(stim_traces['static_avg_fr'])
            dynamic_avg_fr_list.append(stim_traces['dynamic_avg_fr'])
            dynamic_ramp_time_list.append(stim_traces['ramp_time'])
            dynamic_displ_rate_list.append(stim_traces['dynamic_displ_rate'])
            dynamic_force_rate_list.append(stim_traces['dynamic_force_rate'])
        fmt, color = fmt_list[fiber_id], color_list[fiber_id]
        label_text = cleanFiber.mat_filename[:13]
        axs[0].plot(static_displ_list, static_force_list,
                    fmt, color=color, ms=6, label=label_text)
        axs[1].plot(static_displ_list, static_avg_fr_list,
                    fmt, color=color, ms=6, label=label_text)
        axs[2].plot(static_force_list, static_avg_fr_list,
                    fmt, color=color, ms=6, label=label_text)
        static_dynamic_list[fiber_id] = np.c_[fiber_id * np.ones(len(
            static_displ_list)), range(len(static_displ_list)),
            static_displ_list, static_force_list, static_avg_fr_list,
            dynamic_avg_fr_list, dynamic_ramp_time_list,
            dynamic_displ_rate_list, dynamic_force_rate_list]
    for axes in axs:
        axes.set_xlabel(r'Displ. ($\mu$m)')
#        axes.set_xlim(left=0)
    # Treatement for force subplot
    axs[2].set_xlabel('Force (mN)')
    # Other labels
    axs[0].set_ylabel('Force (mN)')
    axs[1].set_ylabel(r'FR$_s$ (Hz)')
    axs[2].set_ylabel(r'FR$_s$ (Hz)')
    # Do legend for the entire figure
    handles, labels = axs[0].get_legend_handles_labels()
    legend = axs[0].legend(handles, labels, bbox_to_anchor=(0, 1.4, 1., 0.3),
                           mode='expand', ncol=2, frameon=True)
    frame = legend.get_frame()
    frame.set_linewidth(.5)
    # Tighten the figure and save
    fig.tight_layout()
    fig.subplots_adjust(top=.85)
    fig.savefig('./plots/repsample/%s.png' % fname, dpi=300)
    plt.close(fig)
    static_dynamic_array = None
    static_dynamic_array = np.vstack((static_dynamic_list))
    if save_data:
        np.savetxt('./csvs/repsample%s.csv' % fname, static_dynamic_array,
                   delimiter=',')
    return fig, axs, static_dynamic_array


def extract_ramp_time(cleanFiber_list):
    ramp_time_list = []
    for cleanFiber in cleanFiber_list:
        for stim_id, stim_traces in enumerate(cleanFiber.traces):
            ramp_time_list.append(stim_traces['ramp_time'])
    return ramp_time_list


def extract_regulated_ramp_curve(cleanFiber_list):
    regulated_ramp_curve_list = []
    for cleanFiber in cleanFiber_list:
        for stim_id, stim_traces in enumerate(cleanFiber.traces):
            max_force_index = stim_traces['force'].argmax()
            ramp_curve = stim_traces['displ'][:max_force_index]
            regulated_ramp_curve_list.append(ramp_curve / ramp_curve.max())

    def ramp_curve_func(xdata, a):
        """
        Parabola form of y = a * x**2 + (1 - a) * x
        """
        ydata = a * xdata**2 + (1. - a) * xdata
        return ydata
    xdata = np.r_[0:1:1000j]
    median_regulated_ramp_curve = get_median_curve(regulated_ramp_curve_list)
    popt = curve_fit(ramp_curve_func, xdata, median_regulated_ramp_curve)[0]
    return regulated_ramp_curve_list, median_regulated_ramp_curve, popt


def get_median_curve(curve_list, xnew=np.r_[0:1:1000j]):
    ynew = []
    for ydata in curve_list:
        xdata = np.linspace(0, 1, ydata.shape[0])
        ynew.append(np.interp(xnew, xdata, ydata))
    ynew = np.array(ynew)
    median_curve = np.median(ynew, axis=0)
    return median_curve


def group_fr(static_dynamic_array, figname='compare_variance.png'):
    tot_fiber_no = int(static_dynamic_array.T[0].max()) + 1
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    class Fiber:

        def __init__(self, fiber_id):
            self.fiber_id = fiber_id
            self.load_fiber_data()
            self.get_stim_group_num_list()
            self.get_stim_group()
            self.generate_binned_exp()
            self.get_var()

        def load_fiber_data(self):
            all_data = static_dynamic_array
            self.fiber_data = all_data[all_data[:, 0] == self.fiber_id][:, 1:]
            self.stim_num, self.static_displ, self.static_force, \
                self.static_avg_fr, self.dynamic_avg_fr, self.ramp_time, \
                self.dynamic_displ_rate, self.dynamic_force_rate = \
                self.fiber_data.T
            self.stim_num = self.stim_num.astype(np.int)

        def get_stim_group_num_list(self):
            feature_unscaled = np.c_[self.static_displ, self.static_force]
            feature = StandardScaler().fit_transform(feature_unscaled)
            db = DBSCAN(eps=.3, min_samples=1).fit(feature)
            self.stim_group_num_list = db.labels_.astype(np.int)
            self.unique_labels = set(self.stim_group_num_list)
            if True:  # Plot out the grouping
                self.fig_grouping, self.axs_grouping = plt.subplots(
                    2, 1, figsize=(3.27, 6))
                colors = plt.cm.get_cmap('Spectral')(np.linspace(0, 1, len(
                    self.unique_labels)))
                for k, col in zip(self.unique_labels, colors):
                    if k == -1:
                        col = 'k'
                    class_members = [index[0] for index in np.argwhere(
                        self.stim_group_num_list == k)]
                    for index in class_members:
                        feature_row = feature_unscaled[index]
                        self.axs_grouping[0].plot(
                            feature_row[0], feature_row[1],
                            'o', markerfacecolor=col)
                        self.axs_grouping[1].plot(
                            feature_row[0], feature_row[1],
                            'o', markerfacecolor=col)
                self.axs_grouping[0].set_xlabel(r'Displ ($\mu$m)')
                self.axs_grouping[0].set_ylabel(r'Force (mN)')
                self.axs_grouping[1].set_xlabel(r'Force (mN)')
                self.axs_grouping[1].set_ylabel(r'Ramp time (ms)')
                self.fig_grouping.tight_layout()
                self.fig_grouping.savefig('./plots/repsample/grouping_%d.png' %
                                          self.fiber_id, dpi=300)
                plt.close(self.fig_grouping)

        def get_stim_group(self):
            # Total amount of groups
            if -1 in self.unique_labels:
                stim_group_list = [[] for i in range(
                    len(self.unique_labels) - 1)]
            else:
                stim_group_list = [[] for i in range(
                    len(self.unique_labels))]
            for i, stim_group_num in enumerate(self.stim_group_num_list):
                if stim_group_num != -1:
                    stim_group_list[stim_group_num].append(self.fiber_data[i])
            self.stim_group_dict = [[] for i in range(
                self.stim_group_num_list.max() + 1)]
            for i, stim_group in enumerate(stim_group_list):
                stim_group_list[i] = np.array(stim_group)
                self.stim_group_dict[i] = {
                    'stim_num': stim_group_list[i][:, 0].astype(np.int),
                    'static_displ': stim_group_list[i][:, 1],
                    'static_force': stim_group_list[i][:, 2],
                    'static_avg_fr': stim_group_list[i][:, 3],
                    'dynamic_avg_fr': stim_group_list[i][:, 4],
                    'ramp_time': stim_group_list[i][:, 5],
                    'dynamic_displ_rate': stim_group_list[i][:, 6],
                    'dynamic_force_rate': stim_group_list[i][:, 7],
                }
            # To sort the stim groups
            displ_array = np.array(
                [self.stim_group_dict[i]['static_displ'].mean()
                 for i in range(len(self.stim_group_dict))])
            ordered_stim_group = [[] for i in range(
                self.stim_group_num_list.max() + 1)]
            for i in range(len(ordered_stim_group)):
                ordered_stim_group[i] = self.stim_group_dict[
                    displ_array.argsort()[i]]
            self.stim_group_dict = ordered_stim_group

        def generate_binned_exp(self):
            self.binned_exp = {
                'bin_size': [],
                'displ_mean': [],
                'displ_std': [],
                'displ_all': [],
                'displ_sem': [],
                'force_mean': [],
                'force_std': [],
                'force_all': [],
                'force_sem': [],
                'static_fr_mean': [],
                'static_fr_std': [],
                'static_fr_all': [],
                'static_fr_sem': [],
                'dynamic_fr_mean': [],
                'dynamic_fr_std': [],
                'dynamic_fr_all': [],
                'dynamic_fr_sem': [],
                'dynamic_displ_rate_mean': [],
                'dynamic_displ_rate_std': [],
                'dynamic_displ_rate_all': [],
                'dynamic_displ_rate_sem': [],
                'dynamic_force_rate_mean': [],
                'dynamic_force_rate_std': [],
                'dynamic_force_rate_all': [],
                'dynamic_force_rate_sem': [],
            }
            for i, stim_group in enumerate(self.stim_group_dict):
                self.binned_exp['bin_size'].append(
                    stim_group['static_displ'].size)
                self.binned_exp['displ_mean'].append(stim_group['static_displ'
                                                                ].mean())
                self.binned_exp['displ_std'].append(stim_group['static_displ'
                                                               ].std(ddof=1))
                self.binned_exp['displ_all'].extend(stim_group['static_displ'])
                self.binned_exp['force_mean'].append(stim_group['static_force'
                                                                ].mean())
                self.binned_exp['force_std'].append(stim_group['static_force'
                                                               ].std(ddof=1))
                self.binned_exp['force_all'].extend(stim_group['static_force'])
                self.binned_exp['static_fr_mean'].append(stim_group[
                    'static_avg_fr'].mean())
                self.binned_exp['static_fr_std'].append(stim_group[
                    'static_avg_fr'].std(ddof=1))
                self.binned_exp['static_fr_all'].extend(stim_group[
                    'static_avg_fr'])
                self.binned_exp['dynamic_fr_mean'].append(stim_group[
                    'dynamic_avg_fr'].mean())
                self.binned_exp['dynamic_fr_std'].append(stim_group[
                    'dynamic_avg_fr'].std(ddof=1))
                self.binned_exp['dynamic_fr_all'].extend(stim_group[
                    'dynamic_avg_fr'])
                self.binned_exp['dynamic_displ_rate_mean'].append(stim_group[
                    'dynamic_displ_rate'].mean())
                self.binned_exp['dynamic_displ_rate_std'].append(stim_group[
                    'dynamic_displ_rate'].std(ddof=1))
                self.binned_exp['dynamic_displ_rate_all'].extend(stim_group[
                    'dynamic_displ_rate'])
                self.binned_exp['dynamic_force_rate_mean'].append(stim_group[
                    'dynamic_force_rate'].mean())
                self.binned_exp['dynamic_force_rate_std'].append(stim_group[
                    'dynamic_force_rate'].std(ddof=1))
                self.binned_exp['dynamic_force_rate_all'].extend(stim_group[
                    'dynamic_force_rate'])
            binned_exp_key_list = ['displ', 'force', 'static_fr', 'dynamic_fr',
                                   'dynamic_displ_rate', 'dynamic_force_rate']
            for key in binned_exp_key_list:
                self.binned_exp[key + '_sem'] = np.array(
                    self.binned_exp[key + '_std']
                    ) / np.sqrt((np.array(self.binned_exp['bin_size'])))
            for key in self.binned_exp.keys():
                if not key.endswith('all') and key is not 'displ_mean':
                    self.binned_exp[key] = np.array(self.binned_exp[key])[
                        np.array(self.binned_exp['displ_mean']).argsort()]
                if key.endswith('all'):
                    self.binned_exp[key] = np.array(self.binned_exp[key])
            self.binned_exp['displ_mean'] = np.array(sorted(
                self.binned_exp['displ_mean']))

        def get_var(self):
            displ_result = get_resvar(
                self.static_displ, self.static_avg_fr)
            displvar = displ_result['resvar']
            displfit = displ_result['finey']
            displfine = displ_result['finex']
            force_result = get_resvar(
                self.static_force, self.static_avg_fr)
            forcevar = force_result['resvar']
            forcefit = force_result['finey']
            forcefine = force_result['finex']
            for key, item in locals().items():
                if 'displ' in key or 'force' in key:
                    setattr(self, key, item)
            return

    # Collect fiber data
    fiber_list = []
    for fiber_id in set(static_dynamic_array.T[0].astype('i')):
        fiber_list.append(Fiber(fiber_id))
    # Calculate variance
    displ_list, force_list, static_fr_list = [], [], []
    for fiber_id, fiber in enumerate(fiber_list):
        displ_list.extend(fiber.binned_exp['displ_mean'])
        force_list.extend(fiber.binned_exp['force_mean'])
        static_fr_list.extend(fiber.binned_exp['static_fr_mean'])
    # Perform fitting
    displ_result = get_resvar(np.array(displ_list), np.array(static_fr_list))
    displ_static_fit_resvar = displ_result['resvar']
    displ_static_predict = displ_result['y']
    force_result = get_resvar(np.array(force_list), np.array(static_fr_list))
    force_static_fit_resvar = force_result['resvar']
    force_static_predict = force_result['y']
    # Get resvar for each fiber
    displvar_list, forcevar_list = [], []
    for fiber in fiber_list:
        displvar_list.append(fiber.displvar)
        forcevar_list.append(fiber.forcevar)
    displvar_array = np.array(displvar_list)
    forcevar_array = np.array(forcevar_list)
    # Plot each fiber
    fig, axs = plt.subplots(3, tot_fiber_no, figsize=(3 * tot_fiber_no, 6))
    for i, fiber in enumerate(fiber_list):
        fmt = MARKER_LIST[i]  # + ':'
        color = 'k'
        axs[0, i].plot(
            np.sort(fiber.static_displ) * 1e-3,
            fiber.static_force[fiber.static_displ.argsort()] * 1e-3,
            fmt, color=color,
            mec=color, ms=MS, label='Fiber #%d' % (i + 1))
        axs[1, i].plot(
            np.sort(fiber.static_displ) * 1e-3,
            fiber.static_avg_fr[fiber.static_displ.argsort()],
            fmt, color=color,
            mec=color, ms=MS, label='#%d' % (i + 1))
        axs[2, i].plot(
            np.sort(fiber.static_force) * 1e-3,
            fiber.static_avg_fr[fiber.static_force.argsort()],
            fmt, color=color,
            mec=color, ms=MS, label='#%d' % (i + 1))
        axs[0, i].plot([], [], '-', color='.5', label='Sigmoidal regression')
        axs[1, i].plot(fiber.displfine * 1e-3,
                       fiber.displfit,
                       '-', color='.5', label='Sigmoidal regression')
        axs[2, i].plot(fiber.forcefine * 1e-3,
                       fiber.forcefit,
                       '-', color='.5', label='Sigmoidal regression')
    # Formatting
    for i, axes in enumerate(axs[0].ravel()):
        axes.set_xlabel('Displacement (mm)')
        axes.set_ylabel('Force (N)')
        axes.legend(loc=2)
    for i, axes in enumerate(axs[1].ravel()):
        axes.set_xlabel('Displacement (mm)')
        axes.set_ylabel('Mean firing (Hz)')
        axes.set_title(r'Within-fiber variance = %.2f $Hz^2$' %
                       displvar_array[i])
    for i, axes in enumerate(axs[2].ravel()):
        axes.set_xlabel('Force (N)')
        axes.set_ylabel('Mean firing (Hz)')
        axes.set_title(r'Within-fiber variance = %.2f $Hz^2$' %
                       forcevar_array[i])
    fig.tight_layout()
    fig.savefig('./plots/repsample/each_%s' % figname, dpi=300)
    plt.close(fig)
    # Plotting all
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 9))
    for i, fiber in enumerate(fiber_list):
        fmt = MARKER_LIST[i] + ':'
        color = 'k'
        axs[0].errorbar(
            fiber.binned_exp['displ_mean'] * 1e-3,
            fiber.binned_exp['force_mean'] * 1e-3,
            fiber.binned_exp['force_sem'] * 1e-3, fmt=fmt,
            color=color, mec=color, ms=MS, label='#%d' % (i + 1))
        axs[1].errorbar(
            fiber.binned_exp['displ_mean'] * 1e-3, fiber.binned_exp
            ['static_fr_mean'], fiber.binned_exp['static_fr_sem'], fmt=fmt,
            color=color, mec=color, ms=MS, label='Fiber #%d' % (i + 1))
        axs[2].errorbar(np.sort(fiber.binned_exp['force_mean']) * 1e-3,
                        fiber.binned_exp['static_fr_mean'][
                            fiber.binned_exp['force_mean'].argsort()],
                        fiber.binned_exp['static_fr_sem'][
                            fiber.binned_exp['force_mean'].argsort()],
                        fmt=fmt, color=color, mec=color, ms=MS,
                        label='Fiber #%d' % (i + 1))
    axs[0].plot([], [], '-', color='.5', label='Sigmoidal regression')
    axs[1].plot(np.sort(displ_list) * 1e-3,
                np.sort(displ_static_predict), '-', color='.5',
                label='Sigmoidal regression')
    axs[2].plot(np.sort(force_list) * 1e-3, np.sort(force_static_predict),
                '-', color='.5',
                label='Sigmoidal regression')
    # Formatting
    axs[1].set_title(r'Between-fiber variance = %.2f $Hz^2$'
                     % displ_static_fit_resvar)
    axs[2].set_title(r'Between-fiber variance = %.2f $Hz^2$'
                     % force_static_fit_resvar)
    axs[0].set_xlabel('Static displ. (mm)')
    axs[1].set_xlabel('Static displ. (mm)')
    axs[2].set_xlabel('Static force (N)')
    axs[0].set_ylabel('Static fore (N)')
    axs[1].set_ylabel('Mean firing (Hz)')
    axs[2].set_ylabel('Mean firing (Hz)')
    axs[0].legend(loc=2)
    axs[0].set_xlim(left=0)
    axs[1].set_xlim(left=0)
    axs[1].set_ylim(bottom=-5)
    axs[2].set_ylim(bottom=-5)
    axs[2].set_xlim(-.05)
    # Adding panel labels
#    for axes_id, axes in enumerate(axs.ravel()):
#        axes.text(-.125, 1.05, chr(65 + axes_id), transform=axes.transAxes,
#                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.savefig('./plots/repsample/all_%s' % figname, dpi=300)
    plt.close(fig)
    return (displ_static_fit_resvar, force_static_fit_resvar,
            displvar_array, forcevar_array, fiber_list)


if __name__ == '__main__':
    # Set the flags
    make_plot = False
    exclude_no_force = True
    exclude_inhibition = True
    run_fiber = True
    pickle_fname = './data/cleanFiber_list.pkl'
    if make_plot:
        # Clear all old plots
        for file_name in os.listdir('./plots/repsample/traces'):
            if file_name.endswith('.png'):
                os.remove('./plots/repsample/traces/'+file_name)
    if run_fiber:
        cleanFiber_list = []
        if exclude_inhibition:
            inhibit_list = ['2013-12-21-02', '2014-01-10-01']
        else:
            inhibit_list = []
        if exclude_no_force:
            no_force_list = ['2013-03-19-01']
        else:
            no_force_list = []
        odd_list = []  # ['2014-07-11-02']
        fiber_table = {}
        for root, subdirs, files in os.walk('data'):
            for fname in files:
                if fname.endswith('.mat') and 'calibrated.mat' in fname\
                        and 'CONT' in fname and 'inhibition' not in fname:
                    cleanFiber = CleanFiber(
                        fname, root, make_plot=make_plot)
                    if not np.isnan(cleanFiber.contact_pos) and\
                            cleanFiber.mat_filename[:13] not in inhibit_list\
                            and cleanFiber.mat_filename[
                                :13] not in no_force_list\
                            and cleanFiber.mat_filename[
                                :13] not in odd_list:
                        cleanFiber_list.append(cleanFiber)
        for i, cleanFiber in enumerate(cleanFiber_list):
            cleanFiber.fiber_id = i
            fiber_table['#%d' % (i + 1)] = cleanFiber.mat_filename
        fiber_series = pd.Series(fiber_table)
        fiber_series.to_csv('./csvs/repsample/fiber_series.csv')
        with open(pickle_fname, 'wb') as f:
            pickle.dump(cleanFiber_list, f)
    else:
        with open(pickle_fname, 'rb') as f:
            cleanFiber_list = pickle.load(f)
    fig, axs, static_dynamic_array = plot_static_dynamic(
        cleanFiber_list, save_data=True)
    # Get the variance
    _, displ_res, _, _, _ = np.polyfit(
        static_dynamic_array.T[2], static_dynamic_array.T[4], 1, full=True)
    _, force_res, _, _, _ = np.polyfit(
        static_dynamic_array.T[3], static_dynamic_array.T[4], 1, full=True)
    # Get extra data
    ramp_time_list = extract_ramp_time(cleanFiber_list)
    displ_list = static_dynamic_array[:, 1]
    ramp_time_coeff = np.polyfit(displ_list, ramp_time_list, 1)
#    regulated_ramp_curve_list, median_regulated_ramp_curve, popt =\
#        extract_regulated_ramp_curve(cleanFiber_list)
    # %% Get grouped view
    displvar, forcevar, displvar_array, forcevar_array, fiber_list = group_fr(
        static_dynamic_array, 'compare_variance.png')
    displvar_gross = get_resvar(static_dynamic_array.T[2],
                                static_dynamic_array.T[4])['resvar']
    forcevar_gross = get_resvar(static_dynamic_array.T[3],
                                static_dynamic_array.T[4])['resvar']
    # %% Compare each fiber by linear fit
    slope_displ_list, slope_force_list = [], []
    for fiber in fiber_list:
        slope_displ = np.polyfit(fiber.binned_exp['displ_mean'],
                                 fiber.binned_exp['static_fr_mean'], 1)[0]
        slope_force = np.polyfit(fiber.binned_exp['force_mean'],
                                 fiber.binned_exp['static_fr_mean'], 1)[0]
        slope_displ_list.append(slope_displ)
        slope_force_list.append(slope_force)
    slope_displ_arr = np.array(slope_displ_list)
    slope_force_arr = np.array(slope_force_list)
    print(sorted(slope_displ_arr) / np.median(slope_displ_arr))
    print(sorted(slope_force_arr) / np.median(slope_force_arr))
    # %% Compare each fiber by sigmoidal parameters
    force_params_list, displ_params_list = [], []
    for fiber in fiber_list:
        force_params_list.append(fiber.force_result['params'])
        displ_params_list.append(fiber.displ_result['params'])
    force_params_df = pd.DataFrame(force_params_list)
    displ_params_df = pd.DataFrame(displ_params_list)
