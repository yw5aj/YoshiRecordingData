import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from convert import CleanFiber, plot_static_dynamic


# Plotting constants
MARKER_LIST = ['v', 'D', 'o', 's', '.', '*', '.', 'x', 'h', '+']
COLOR_LIST = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b']
LS_LIST = ['-', '--', '-.', ':']
MS = 6


def plot_exclude(cleanFiberList, exclude_fname_list=[]):
    new_list = [cleanFiber for cleanFiber in cleanFiberList if
                cleanFiber.mat_filename not in exclude_fname_list]
    fig, axs, static_dynamic_array = plot_static_dynamic(new_list)
    return fig, axs


def get_residual_variance(x, y):
    popt = np.polyfit(x, y, 1)
    yfit = np.polyval(popt, x)
    res = yfit - y
    return res.var()


def group_fr(static_dynamic_array):
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    class Fiber:

        def __init__(self, fiber_id):
            self.fiber_id = fiber_id
            self.load_fiber_data()
            self.get_stim_group_num_list()
            self.get_stim_group()
            self.generate_binned_exp()

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
            db = DBSCAN(eps=.3, min_samples=2).fit(feature)
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
                self.fig_grouping.savefig('./plots/grouping_%d.png' %
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
    displ_static_fit_param = np.polyfit(displ_list, static_fr_list, 1)
    force_static_fit_param = np.polyfit(force_list, static_fr_list, 1)
    displ_static_predict = np.polyval(displ_static_fit_param, displ_list)
    force_static_predict = np.polyval(force_static_fit_param, force_list)
    # Calculate residual variance
    displ_static_fit_res = displ_static_predict - np.asarray(static_fr_list)
    force_static_fit_res = force_static_predict - np.asarray(static_fr_list)
    displ_static_fit_resvar = displ_static_fit_res.var(ddof=1)
    force_static_fit_resvar = force_static_fit_res.var(ddof=1)
    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(3.5, 9))
    for i, fiber in enumerate(fiber_list):
        fmt = MARKER_LIST[i] + ':'
        color = 'k'
        axs[0].errorbar(
            fiber.binned_exp['displ_mean'] * 1e-3, fiber.binned_exp
            ['force_mean'], fiber.binned_exp['force_sem'], fmt=fmt,
            color=color, mec=color, ms=MS, label='#%d' % (i + 1))
        axs[1].errorbar(
            fiber.binned_exp['displ_mean'] * 1e-3, fiber.binned_exp
            ['static_fr_mean'], fiber.binned_exp['static_fr_sem'], fmt=fmt,
            color=color, mec=color, ms=MS, label='Fiber #%d' % (i + 1))
        axs[2].errorbar(np.sort(fiber.binned_exp['force_mean']),
                        fiber.binned_exp['static_fr_mean'][
                            fiber.binned_exp['force_mean'].argsort()],
                        fiber.binned_exp['static_fr_sem'][
                            fiber.binned_exp['force_mean'].argsort()],
                        fmt=fmt, color=color, mec=color, ms=MS,
                        label='Fiber #%d' % (i + 1))
    axs[0].plot([], [], '-', c='.5', label='Linear regression')
    axs[1].plot(np.sort(displ_list) * 1e-3,
                np.sort(displ_static_predict), '-', c='.5',
                label='Linear regression')
    axs[2].plot(sorted(force_list), np.sort(force_static_predict),
                '-k', c='.5',
                label='Linear regression')
    axs[0].set_xlabel(r'Static displ. (mm)')
    axs[1].set_xlabel(r'Static displ. (mm)')
    axs[2].set_xlabel(r'Static force (mN)')
    axs[0].set_ylabel(r'Static force (mN)')
    axs[1].set_ylabel('Mean firing (Hz)')
    axs[2].set_ylabel('Mean firing (Hz)')
    axs[0].legend(loc=2)
    axs[1].set_ylim(bottom=-5)
    axs[2].set_ylim(bottom=-5)
    # Adding panel labels
    for axes_id, axes in enumerate(axs.ravel()):
        axes.text(-.125, 1.05, chr(65 + axes_id), transform=axes.transAxes,
                  fontsize=12, fontweight='bold', va='top')
    fig.tight_layout()
    fig.savefig('./plots/compare_variance.png', dpi=300)
    plt.close(fig)
    return displ_static_fit_resvar, force_static_fit_resvar


if __name__ == '__main__':
    cleanFiberDict = {}
    key_list = ['in', 'out']
    fpath_dict = {
        'in': './rawData/finalSAI/',
        'out': './rawData/finalSAI/excluded/'}
    cleanFiberDict['all'] = []
    for key in key_list:
        cleanFiberDict[key] = []
        fpath = fpath_dict[key]
        for fname in os.listdir(fpath):
            if fname.endswith('.mat'):
                cleanFiber = CleanFiber(
                    fname, fpath, threshold=.25, pad=300, make_plot=False)
                cleanFiberDict[key].append(cleanFiber)
                cleanFiberDict['all'].append(cleanFiber)
                print(fname + ' completed...')
    # Plotting
    for i, cleanFiber in enumerate(cleanFiberDict['all']):
        cleanFiber.fiber_id = i
    figall, axsall, static_dynamic_array = plot_static_dynamic(
        cleanFiberDict['all'])
    # Exclude bad fibers
    bad_fiber_list = []
    for i, cleanFiber in enumerate(cleanFiberDict['all']):
        if '042701V_01' in cleanFiber.mat_filename\
                or '042703V_01' in cleanFiber.mat_filename:
            bad_fiber_list.append(cleanFiber)
    for cleanFiber in bad_fiber_list:
        cleanFiberDict['all'].remove(cleanFiber)
    for i, cleanFiber in enumerate(cleanFiberDict['all']):
        cleanFiber.fiber_id = i
    figall, axsall, static_dynamic_array = plot_static_dynamic(
        cleanFiberDict['all'])
    # Get the exclusions
#    exclude_fname_list = ['2012042701V_01.mat', '2012042703V_01.mat']
#    plot_exclude(cleanFiberDict['all'], exclude_fname_list)
