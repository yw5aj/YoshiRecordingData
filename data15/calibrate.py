import os
import time
import copy
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import numpy as np


fs = 16e3  # Sampling frequency


def if_earlier(time_str, std_time_str='2014-07-01', fmt_str='%Y-%m-%d'):
    std_struct_time = time.strptime(std_time_str, fmt_str)
    struct_time = time.strptime(time_str, fmt_str)
    return struct_time < std_struct_time


def convert_force(force_volt, time_str):
    """
    Calibration note taken from Mouse+data+Atoh1+Piezo2+UVA+2015-9-25.xlsx
    """
    force_volt -= force_volt[:100].mean(axis=0)
    if if_earlier(time_str):
        force = 0.69 * force_volt
    else:
        force = 175.53 * force_volt
    return force


def convert_displ(displ_volt, time_str):
    """
    Calibration note taken from Mouse+data+Atoh1+Piezo2+UVA+2015-9-25.xlsx
    """
    displ_volt -= displ_volt[:100].mean(axis=0)
    if if_earlier(time_str):
        displ = 2.5 * displ_volt
    else:
        displ = 2.5 * displ_volt * 1000
    return displ


def calibrate_file(root, fname, plot=False):
    data_old = loadmat(os.path.join(root, fname))
    data_new = copy.deepcopy(data_old)
    for key, item in data_old.items():
        if key.startswith('OUT_PUT_D'):
            data_new['C' + key] = convert_displ(item, fname[:10])
        elif key.startswith('OUT_PUT_F'):
            data_new['C' + key] = convert_force(item, fname[:10])
    savemat(os.path.join(root, fname[:-4]) + '_calibrated.mat', data_new,
            do_compression=True)
    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(3.5, 7))
        axs[0].set_title(fname)
        for key, item in data_new.items():
            if key.startswith('COUT_PUT_D'):
                axs[0].plot(item)
            if key.startswith('COUT_PUT_F'):
                axs[1].plot(item)
        axs[0].set_xlabel(r'Sampling point')
        axs[0].set_ylabel(r'Displacement ($\mu$m)')
        axs[1].set_xlabel(r'Sampling point')
        axs[1].set_ylabel(r'Force (mN)')
        fig.tight_layout()
        fig.savefig('./plots/recordings/fiber%s.png' % fname[:13], dpi=300)
        plt.close(fig)
    return data_new


if __name__ == '__main__':
    for root, subdirs, files in os.walk('data'):
        for fname in files:
            if fname.endswith('.mat') and 'calibrated' not in fname:
                calibrate_file(root, fname, plot=True)
