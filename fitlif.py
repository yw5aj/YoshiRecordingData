# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:55:53 2014

@author: Yuxiang Wang
"""

import numpy as np
import matplotlib.pyplot as plt
import ctypes
from scipy.optimize import minimize

from constants import (DT, RESISTANCE_LIF, CAPACITANCE_LIF,
    VOLTAGE_THRESHOLD_LIF, STATIC_START, STATIC_END)


# Load dll for LIF model
_get_spike_trace_array_lif = ctypes.cdll.LoadLibrary('./lifmodule.dll'
    ).get_spike_trace_array_lif
_get_spike_trace_array_lif.argtypes = [ctypes.c_double, ctypes.c_double, 
    ctypes.c_double, np.ctypeslib.ndpointer(ctypes.c_double), 
    ctypes.c_int, ctypes.c_double, np.ctypeslib.ndpointer(ctypes.c_int)]
_get_spike_trace_array_lif.restype = None

# Load dll for Lesniak model
_get_spike_trace_array_lesniak = ctypes.cdll.LoadLibrary('./lesniakmodule.dll'
    ).get_spike_trace_array_lesniak
_get_spike_trace_array_lesniak.argtypes = [ctypes.c_double, ctypes.c_double,
    ctypes.c_double, np.ctypeslib.ndpointer(np.uintp, ndim=1), 
    ctypes.c_int, ctypes.c_double, np.ctypeslib.ndpointer(ctypes.c_int),
    ctypes.c_int, np.ctypeslib.ndpointer(ctypes.c_int)]
_get_spike_trace_array_lesniak.restype = None


def current_array_to_spike_array(current_array, model='LIF',
                                 mcnc_grouping=None):
    # Make sure it is contiguous in C
    if not current_array.flags['C_CONTIGUOUS']:
        current_array = current_array.copy(order='C')
    # Initialize output array
    spike_array = np.zeros(current_array.shape[0], dtype=np.int)
    # Call C function
    if model == 'LIF':
        _get_spike_trace_array_lif(RESISTANCE_LIF, CAPACITANCE_LIF,
            VOLTAGE_THRESHOLD_LIF, current_array, current_array.shape[0], DT, 
            spike_array)
    elif model == 'Lesniak':
        assert not mcnc_grouping is None, 'mcnc_grouping undefined'
        current_array_pp = (current_array.__array_interface__['data'][0]\
            + np.arange(current_array.shape[0])*current_array.strides[0]
            ).astype(np.uintp)
        _get_spike_trace_array_lesniak(RESISTANCE_LIF, CAPACITANCE_LIF,
            VOLTAGE_THRESHOLD_LIF, current_array_pp, current_array.shape[0], 
            DT, spike_array, mcnc_grouping.size, mcnc_grouping)
    return spike_array


def fr2current(fr):
    """
    Governing equation for LIF in steady state:
    u/R + C * du/dt = I
    Solution with u(0) = 0 is:
    u = IR(1-exp(-t/(R*C)))
    Therefore,
    f = -1/(R*C*ln(1-U/(I*R))) <- from current to frequency
    I = U/(R*(1-exp(-1/(R*C*f)))) <- from frequency to current
    """
    if fr > 0:
        current = VOLTAGE_THRESHOLD_LIF / (RESISTANCE_LIF * (1. - np.exp(-1. / 
            (RESISTANCE_LIF * CAPACITANCE_LIF * fr))))
    else:
        current = VOLTAGE_THRESHOLD_LIF / RESISTANCE_LIF
    return current


def current2fr(current):
    if current > VOLTAGE_THRESHOLD_LIF / RESISTANCE_LIF:
        fr = -1./ (RESISTANCE_LIF * CAPACITANCE_LIF * np.log(1. - 
            VOLTAGE_THRESHOLD_LIF / (current * RESISTANCE_LIF)))
    else:
        fr = 0.
    return fr


def current_array_to_fr(current_array, max_index, model='LIF',
                        mcnc_grouping=None):
    current_array[current_array<0] = 0.
    spike_array = current_array_to_spike_array(current_array, model=model,
                                               mcnc_grouping=mcnc_grouping)
    # Get time windows
    static_window = np.arange(max_index+STATIC_START/DT,
                              max_index+STATIC_END/DT, dtype=np.int)
    dynamic_window = np.arange(0., max_index, dtype=np.int)    
    static_fr = get_avg_fr(spike_array[static_window])
    dynamic_fr = get_avg_fr(spike_array[dynamic_window])
    return static_fr, dynamic_fr


def get_avg_fr(spike_array):
    spike_index = np.nonzero(spike_array)[0]
    spike_count = spike_index.shape[0]
    if spike_count > 1:
        spike_duration = (spike_index[-1] - spike_index[0]) * DT
        avg_fr = (spike_count - 1) / spike_duration
    else:
        avg_fr = 0.
    return avg_fr


def trans_param_to_fr(quantity_dict, trans_param, model='LIF',
                      mcnc_grouping=None):
    quantity_array = quantity_dict['quantity_array']
    max_index = quantity_dict['max_index']
    quantity_rate_array = np.abs(np.gradient(quantity_array)) / DT
    current_array = trans_param[0] * quantity_array +\
        trans_param[1] * quantity_rate_array + trans_param[2]
    static_fr, dynamic_fr = current_array_to_fr(current_array, max_index,
        model=model, mcnc_grouping=mcnc_grouping)
    return static_fr, dynamic_fr


def trans_param_to_predicted_fr(quantity_dict_list, trans_param, model='LIF',
                                mcnc_grouping=None):
    """
    Different between trans_param_to_predicted_fr and trans_param_to_fr:
    trans_param_to_fr is for one quantity trace
    trans_param_to_predicted_fr is for all quantity traces
    """
    predicted_static_fr, predicted_dynamic_fr = [], []
    for quantity_dict in quantity_dict_list:
        static_fr, dynamic_fr = trans_param_to_fr(quantity_dict, trans_param,
            model=model, mcnc_grouping=mcnc_grouping)
        predicted_static_fr.append(static_fr)
        predicted_dynamic_fr.append(dynamic_fr)
    predicted_fr = np.c_[range(len(quantity_dict_list)),
                         predicted_static_fr,
                         predicted_dynamic_fr]
    return predicted_fr


def trans_param_to_fr_r2_fitting(fitx, trans_param_init, quantity_dict_list,
    target_fr_array, sign=1.):
    trans_param = fitx * trans_param_init
    r2 = trans_param_to_fr_r2(trans_param, quantity_dict_list, target_fr_array)
    return sign * r2


def trans_param_to_fr_r2(trans_param, quantity_dict_list, target_fr_array):
    predicted_fr = trans_param_to_predicted_fr(quantity_dict_list, trans_param)
    predicted_fr_array = np.empty_like(target_fr_array)
    for i in range(predicted_fr_array.shape[0]):
        predicted_fr_array[i, 0] = target_fr_array[i, 0]
        predicted_fr_array[i, 1] = predicted_fr[int(target_fr_array[i, 0]), 1]
        predicted_fr_array[i, 2] = predicted_fr[int(target_fr_array[i, 0]), 2]
    static_r2 = get_r2(target_fr_array[:, 1], predicted_fr_array[:, 1])
    dynamic_r2 = get_r2(target_fr_array[:, 2], predicted_fr_array[:, 2])
    r2 = .5 * (static_r2 + dynamic_r2)
#    r2 = static_r2
    print(r2)
    return r2


def get_r2(target_array, predicted_array):
    ssres = ((target_array - predicted_array)**2).sum()
    sstot = target_array.var() * target_array.size
    r2 = 1. - ssres / sstot
    return r2


def fit_trans_param(quantity_dict_list, target_fr_array):
    trans_param_init = get_lstsq_fit(quantity_dict_list, target_fr_array)
    res = minimize(trans_param_to_fr_r2_fitting, np.ones(3), 
        args=(trans_param_init, quantity_dict_list, target_fr_array, 
        -1.), method='SLSQP', options={'eps': 1e-3})
    trans_param = res.x * trans_param_init
    return trans_param


def get_lstsq_fit(quantity_dict_list, target_fr_array):
    # Get the mean of all firing rates at different indentation depth
    target_fr_vector = np.empty(2*(target_fr_array[:, 0].max().astype(np.int)
        +1))
    for i in range(target_fr_vector.size//2):
        target_fr_vector[i] = target_fr_array[:, 1][target_fr_array[:, 0]==i
            ].mean()
        target_fr_vector[target_fr_vector.size//2+i] = target_fr_array[:, 2][
            target_fr_array[:, 0]==i].mean()
    # Get the corresponding current vector
    target_current_vector = np.empty_like(target_fr_vector)
    for i, fr in enumerate(target_fr_vector):
        target_current_vector[i] = fr2current(fr)
    # Get the quantity and rate matrix
    (static_mean_quantity_array, static_mean_quantity_rate_array,
        dynamic_mean_quantity_array, dynamic_mean_quantity_rate_array
        ) = get_mean_quantity_and_rate(quantity_dict_list)
    quantity_and_rate_matrix = np.c_[
        np.r_[static_mean_quantity_array, dynamic_mean_quantity_array],
        np.r_[static_mean_quantity_rate_array, dynamic_mean_quantity_rate_array],
        np.ones(target_fr_vector.size)]
    trans_param = np.linalg.lstsq(quantity_and_rate_matrix, 
        target_current_vector)[0]
    return trans_param


def get_mean_quantity_and_rate(quantity_dict_list):
    static_mean_quantity_list, static_mean_quantity_rate_list,\
        dynamic_mean_quantity_list, dynamic_mean_quantity_rate_list =\
        [], [], [], []
    for i, quantity_dict in enumerate(quantity_dict_list):
        # Unpack FEM outputs
        quantity_array = quantity_dict['quantity_array']
        max_index = quantity_dict['max_index']
        max_index -= (quantity_array<=0).sum()
        quantity_array = quantity_array[quantity_array>0]
        quantity_rate_array = np.abs(np.gradient(quantity_array)) / DT
        # Get time windows
        static_window = np.arange(max_index+STATIC_START/DT,
                                  max_index+STATIC_END/DT, dtype=np.int)
        dynamic_window = np.arange(0., max_index, dtype=np.int)
        # Get average quantities
        static_mean_quantity_list.append(quantity_array[static_window
            ].mean())
        static_mean_quantity_rate_list.append(quantity_rate_array[
            static_window].mean())
        dynamic_mean_quantity_list.append(quantity_array[dynamic_window
            ].mean())
        dynamic_mean_quantity_rate_list.append(quantity_rate_array[
            dynamic_window].mean())
    static_mean_quantity_array = np.array(static_mean_quantity_list)
    static_mean_quantity_rate_array = np.array(
        static_mean_quantity_rate_list)
    dynamic_mean_quantity_array = np.array(dynamic_mean_quantity_list)
    dynamic_mean_quantity_rate_array = np.array(
        dynamic_mean_quantity_rate_list)    
    return (static_mean_quantity_array,
            static_mean_quantity_rate_array,
            dynamic_mean_quantity_array,
            dynamic_mean_quantity_rate_array)


def plot_fitlif_res(binned_exp, trans_param, quantity_dict_list):
    fig, axs = plt.subplots(2, 1, figsize=(3.27, 5))
    ms = 6
    # Plot experiment data
    axs[0].errorbar(binned_exp['displ_mean'], binned_exp['static_fr_mean'], 
        binned_exp['static_fr_std'], fmt='o', ms=ms)
    axs[1].errorbar(binned_exp['displ_mean'], binned_exp['dynamic_fr_mean'], 
        binned_exp['dynamic_fr_std'], fmt='o', ms=ms)
    # Calculate fitted data
    predicted_fr = trans_param_to_predicted_fr(quantity_dict_list, trans_param)
    predicted_static_fr, predicted_dynamic_fr = predicted_fr.T[1:]
    axs[0].plot(binned_exp['displ_mean'], predicted_static_fr)
    axs[1].plot(binned_exp['displ_mean'], predicted_dynamic_fr)
    # Format the figure
    axs[0].set_xlabel(r'Displ. ($\mu$m)')
    axs[1].set_xlabel(r'Displ. ($\mu$m)')
    axs[0].set_ylabel('Static FR (Hz)')
    axs[1].set_ylabel('Dynamic FR (Hz)')
    fig.tight_layout()
    return fig, axs



if __name__ == '__main__':    
    pass
