import re
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit


class CleanFiber:

    def __init__(self, mat_filename, mat_pathname, threshold=.25, pad=300.,
                 make_plot=False):
        self.contact_standard = {'threshold': threshold, 'pad': pad}
        self.mat_filename = mat_filename
        self.mat_pathname = mat_pathname
        self.fs = int(16e3)  # Yoshi's sampling frequency
        self.get_animal_info()
        self.get_mat_data()
        self.sort_traces()
        self.find_contact_by_force(threshold, pad)
        self.cut_traces(make_plot=make_plot)
        return

    def get_animal_info(self):
        with open('./notes/ModelGfpInfo0729.csv', 'r') as f:
            animal_info_splitted = re.findall(
                self.mat_filename + r'.+', f.read())[0].split(',')
        self.animal_info = {'mat_filename': animal_info_splitted[0],
                            'log_filename': animal_info_splitted[1],
                            'date_recorded': animal_info_splitted[2],
                            'unitid': int(animal_info_splitted[3]),
                            'age': float(animal_info_splitted[4]),
                            'weight': float(animal_info_splitted[5]),
                            'tip_dia': float(animal_info_splitted[6]),
                            'mouseid': int(animal_info_splitted[7])}
        return self.animal_info

    def get_mat_data(self):
        self.mat_data = loadmat(self.mat_pathname + self.mat_filename)
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
        b, a = butter(8, 1 / 64)  # Fs = fs/64 = 250 Hz
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

    def find_contact_by_force(self, threshold, pad):
        self.contact_pos = []
        for stim_id, stim_traces_full in enumerate(self.traces_full):
            # try-except is used in case some force traces are too trivial &
            # below threshold force
            try:
                contact_index = np.nonzero(
                    stim_traces_full['force'] > threshold)[0][0]
            except:
                continue
            self.contact_pos.append(stim_traces_full['displ'][contact_index])
        self.contact_pos = np.array(self.contact_pos)
        self.contact_pos = np.median(self.contact_pos) - pad
        return self.contact_pos

    def cut_traces(self, make_plot=False):
        self.traces = []
        for stim_id, stim_traces_full in enumerate(self.traces_full):
            contact_index = np.nonzero(
                stim_traces_full['displ'] >= self.contact_pos)[0][0]
            max_force_index = stim_traces_full['force'].argmax()
            self.traces.append({
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
                    max_force_index + 2 * self.fs:max_force_index + int(
                        4.5 * self.fs)].mean(),
                'static_displ': stim_traces_full['displ'][
                    max_force_index + 2 * self.fs:max_force_index + int(
                        4.5 * self.fs)].mean() - stim_traces_full['displ'][
                        contact_index],
                'dynamic_force_rate': np.diff(
                    stim_traces_full['force'][
                        contact_index:max_force_index]).mean() * self.fs,
                'dynamic_displ_rate': np.diff(
                    stim_traces_full['displ'][
                        contact_index:max_force_index]).mean() * self.fs,
            })
            try:
                self.traces[-1]['raw_spike'] = stim_traces_full[
                    'raw_spike'][contact_index:contact_index + 6 * self.fs]
            except KeyError:
                pass
            self.traces[-1]['static_avg_fr'] = self._get_avg_fr(
                stim_traces_full['spike_trace'][
                    max_force_index + int(2. * self.fs):max_force_index + int(
                        4.5 * self.fs)])
            self.traces[-1]['dynamic_avg_fr'] = self._get_avg_fr(
                stim_traces_full['spike_trace'][contact_index:max_force_index])
            if np.any(self.traces[-1]['spike_trace']):
                self.traces[-1]['spike_time'], self.traces[-1]['spike_isi'],\
                    self.traces[-1]['spike_fr'] = self._get_spike_data(
                        self.traces[-1]['spike_trace'])
            if make_plot:
                fig, axs = plt.subplots(3, 1, figsize=(6.83, 9.19))
                for i, item in enumerate(['displ', 'force', 'spike_trace']):
                    axs[i].plot(
                        stim_traces_full['time'][
                            contact_index - .5 * self.fs:contact_index +
                            6.5 * self.fs],
                        stim_traces_full[item][
                            contact_index - .5 * self.fs:contact_index +
                            6.5 * self.fs], '-k', color='.0')
                    axs[i].axvline(contact_index / self.fs, color='.0')
                    axs[i].axvline(contact_index / self.fs + 6., color='.0')
                    axs[i].set_xlabel('Time (s)')
                axs[0].set_title(
                    'displ = %f, force = %f, missed spikes = %d' % (
                        self.traces[-1]['static_displ'],
                        self.traces[-1]['static_force'],
                        stim_traces_full['spike_trace'][:contact_index].sum()))
                axs[0].set_ylabel('Displ. ($\mu$m)')
                axs[1].set_ylabel('Force (mN)')
                axs[2].set_ylabel('Spikes')
                axs[2].set_ylim(-2, 2)
                fig.savefig(
                    './plots/traces/' + self.mat_filename[:-4] + 'stim' +
                    str(stim_id) + '.png', dpi=300)
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


def plot_specific_fibers(fiberList, fname='temp'):
    if not isinstance(fiberList, list):
        fiberList = [fiberList]
    fig, axs, static_dynamic_array = plot_static_dynamic(
        fiberList, save_data=False, fname=fname)
    return fig, axs, static_dynamic_array


def plot_static_dynamic(cleanFiberList, fs=16e3, save_data=False,
                        fname='static_dynamic'):
    fig, axs = plt.subplots(3, 1, figsize=(3.27, 6.83))
    static_dynamic_list = [[] for i in range(cleanFiberList[-1].fiber_id + 1)]
    fmt_list = ['*', 'D', 'v', 's', '.', 'o', '.', 'x', 'h', '+']
    color_list = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
    for cleanFiber in cleanFiberList:
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
        label_text = cleanFiber.animal_info['mat_filename'][4:-4]
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
    fig.savefig('./plots/%s.png' % fname, dpi=300)
    static_dynamic_array = None
    static_dynamic_array = np.vstack((static_dynamic_list))
    if save_data:
        np.savetxt('./csvs/%s.csv' % fname, static_dynamic_array,
                   delimiter=',')
    return fig, axs, static_dynamic_array


def extract_ramp_time(cleanFiberList):
    ramp_time_list = []
    for cleanFiber in cleanFiberList:
        for stim_id, stim_traces in enumerate(cleanFiber.traces):
            ramp_time_list.append(stim_traces['ramp_time'])
    return ramp_time_list


def extract_regulated_ramp_curve(cleanFiberList):
    regulated_ramp_curve_list = []
    for cleanFiber in cleanFiberList:
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


if __name__ == '__main__':
    # Clear all old plots
    """
    for file_name in os.listdir('./plots/traces'):
        if file_name.endswith('.png'):
            os.remove('./plots/traces/'+file_name)
    """
    cleanFiberList = []
    pathname = './rawData/finalSAI/'
    for filename in os.listdir(pathname):
        if filename.endswith('.mat'):
            cleanFiberList.append(CleanFiber(
                filename, pathname, threshold=.25, pad=300., make_plot=False))
            print(filename + ' completed...')
    for i, cleanFiber in enumerate(cleanFiberList):
        cleanFiber.fiber_id = i
    fig, axs, static_dynamic_array = plot_static_dynamic(
        cleanFiberList, save_data=True)
    ramp_time_list = extract_ramp_time(cleanFiberList)
    displ_list = static_dynamic_array[:, 1]
    ramp_time_coeff = np.polyfit(displ_list, ramp_time_list, 1)
    regulated_ramp_curve_list, median_regulated_ramp_curve, popt =\
        extract_regulated_ramp_curve(cleanFiberList)
    with open('./finaldata/cleanFiberList.pkl', 'wb') as f:
        pickle.dump(cleanFiberList, f)
    # Plot after exclusion
    exclude_list = ['2012042701V_01.mat', '2012042703V_01.mat',
                    '2012031501V_01.mat', '2012043002V_01.mat']
    finalCleanFiberList = [cleanFiber for cleanFiber in cleanFiberList if
                           cleanFiber.animal_info['mat_filename'] not in
                           exclude_list]
