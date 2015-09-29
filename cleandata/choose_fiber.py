import os
import copy
import numpy as np

from convert import CleanFiber, plot_static_dynamic


def plot_exclude(cleanFiberList, exclude_fname_list=[]):
    new_list = [cleanFiber for cleanFiber in cleanFiberList if
                cleanFiber.mat_filename not in exclude_fname_list]
    fig, axs, static_dynamic_array = plot_static_dynamic(new_list)
    displ_var = get_residual_variance(static_dynamic_array[:, 2],
                                      static_dynamic_array[:, 4])
    force_var = get_residual_variance(static_dynamic_array[:, 3],
                                      static_dynamic_array[:, 4])
    return fig, axs, displ_var, force_var


def get_residual_variance(x, y):
    popt = np.polyfit(x, y, 1)
    yfit = np.polyval(popt, x)
    res = yfit - y
    return res.var()


if __name__ == '__main__':
    cleanFiberDict = {}
    key_list = ['in', 'out']
    fpath_dict = {
        'in': './rawData/finalSAI/',
        'out': './rawData/finalSAI/excluded/'}
    for key in key_list:
        cleanFiberDict[key] = []
        fpath = fpath_dict[key]
        for fname in os.listdir(fpath):
            if fname.endswith('.mat'):
                cleanFiberDict[key].append(CleanFiber(
                    fname, fpath, threshold=.25, pad=300, make_plot=False))
                print(fname + ' completed...')
    # Plotting
    for key in key_list:
        i = 0
        for cleanFiber in cleanFiberDict[key]:
            cleanFiber.fiber_id = i
            i += 1
    figin, axsin, _ = plot_static_dynamic(cleanFiberDict['in'])
    cleanFiberDict['all'] = copy.deepcopy(cleanFiberDict['in'])
    cleanFiberDict['all'].extend(cleanFiberDict['out'])
    figall, axsall, static_dynamic_array = plot_static_dynamic(
        cleanFiberDict['all'])
    # Get the exclusions
    exclude_fname_list = ['2012042701V_01.mat', '2012042703V_01.mat']
    plot_exclude(cleanFiberDict['all'], exclude_fname_list)
