# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:55:53 2014

@author: Yuxiang Wang
"""

import numpy as np
import ctypes
from scipy.optimize import minimize

from constants import (DT, RESISTANCE_LIF, CAPACITANCE_LIF,
                       VOLTAGE_THRESHOLD_LIF, STATIC_START, STATIC_END)


# Load dll for LIF model
_get_spike_trace_array_lif = ctypes.cdll.LoadLibrary(
    './lifmodule.dll').get_spike_trace_array_lif
_get_spike_trace_array_lif.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_double,
    np.ctypeslib.ndpointer(ctypes.c_double),
    ctypes.c_int, ctypes.c_double, np.ctypeslib.ndpointer(ctypes.c_int)]
_get_spike_trace_array_lif.restype = None

# Load dll for Lesniak model
_get_spike_trace_array_lesniak = ctypes.cdll.LoadLibrary(
    './lesniakmodule.dll').get_spike_trace_array_lesniak
_get_spike_trace_array_lesniak.argtypes = [
    ctypes.c_double, ctypes.c_double,
    ctypes.c_double, np.ctypeslib.ndpointer(np.uintp, ndim=1),
    ctypes.c_int, ctypes.c_double, np.ctypeslib.ndpointer(ctypes.c_int),
    ctypes.c_int, np.ctypeslib.ndpointer(ctypes.c_int)]
_get_spike_trace_array_lesniak.restype = None


class LifModel:

    def __init__(self, r=None, c=None, v=None):
        """
        Initialize the LIF model by specifying RC params.

        Parameters
        ----------
        r : float
            Resistance
        c : float
            Capacitance
        v : float
            Voltage threshold to generate a spike and reset
        """
        if r is None and c is None and v is None:
            self.r = RESISTANCE_LIF
            self.c = CAPACITANCE_LIF
            self.v = VOLTAGE_THRESHOLD_LIF
        else:
            self.r, self.c, self.v = r, c, v

    def current_array_to_spike_array(self, current_array, model='LIF',
                                     mcnc_grouping=None):
        # Make sure it is contiguous in C
        if not current_array.flags['C_CONTIGUOUS']:
            current_array = current_array.copy(order='C')
        # Initialize output array
        spike_array = np.zeros(current_array.shape[0], dtype=np.int)
        # Call C function
        if model == 'LIF':
            _get_spike_trace_array_lif(
                self.r, self.c, self.v, current_array, current_array.shape[0],
                DT, spike_array)
        elif model == 'Lesniak':
            assert mcnc_grouping is not None, 'mcnc_grouping undefined'
            current_array_pp = (current_array.__array_interface__['data'][0] +
                                np.arange(current_array.shape[0]) *
                                current_array.strides[0]).astype(np.uintp)
            _get_spike_trace_array_lesniak(
                self.r, self.c, self.v,
                current_array_pp, current_array.shape[0],
                DT, spike_array, mcnc_grouping.size, mcnc_grouping)
        return spike_array

    def fr2current(self, fr):
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
            current = self.v / (
                self.r * (1. - np.exp(-1. / (self.r * self.c * fr))))
        else:
            current = self.v / self.r
        return current

    def current2fr(self, current):
        if current > self.v / self.r:
            fr = -1. / (
                self.r * self.c * np.log(
                    1. - self.v / (current * self.r)))
        else:
            fr = 0.
        return fr

    def current_array_to_fr(self, current_array, max_index, model='LIF',
                            mcnc_grouping=None):
        # Is this constraint really valid?
        current_array[current_array < 0] = 0.
        spike_array = self.current_array_to_spike_array(
            current_array, model=model, mcnc_grouping=mcnc_grouping)
        # Get time windows
        static_window = np.arange(max_index + STATIC_START / DT,
                                  max_index + STATIC_END / DT, dtype=np.int)
        dynamic_window = np.arange(0., max_index, dtype=np.int)
        static_fr = self.get_avg_fr(spike_array[static_window])
        dynamic_fr = self.get_avg_fr(spike_array[dynamic_window])
        return static_fr, dynamic_fr

    def get_avg_fr(self, spike_array):
        spike_index = np.nonzero(spike_array)[0]
        spike_count = spike_index.shape[0]
        if spike_count > 1:
            spike_duration = (spike_index[-1] - spike_index[0]) * DT
            avg_fr = (spike_count - 1) / spike_duration
        else:
            avg_fr = 0.
        return avg_fr

    def trans_param_to_fr(self, quantity_dict, trans_param, model='LIF',
                          mcnc_grouping=None, std=None):
        max_index = quantity_dict['max_index']
        current_array = self.trans_param_to_current_array(
            quantity_dict, trans_param, model=model,
            mcnc_grouping=mcnc_grouping, std=std)
        static_fr, dynamic_fr = self.current_array_to_fr(
            current_array, max_index, model=model, mcnc_grouping=mcnc_grouping)
        return static_fr, dynamic_fr

    def trans_param_to_cov(self, quantity_dict, trans_param, model='LIF',
                           mcnc_grouping=None, std=None):
        current_array = self.trans_param_to_current_array(
            quantity_dict, trans_param, model=model,
            mcnc_grouping=mcnc_grouping, std=std)
        spike_array = self.current_array_to_spike_array(
            current_array, model=model, mcnc_grouping=mcnc_grouping)
        spike_timings = spike_array.nonzero()[0] * DT
        isi = np.diff(spike_timings)
        cov = isi.std() / isi.mean()
        return cov

    def trans_param_to_current_array(self, quantity_dict, trans_param,
                                     model='LIF', mcnc_grouping=None,
                                     std=None):
        quantity_array = quantity_dict['quantity_array']
        quantity_rate_array = np.abs(np.gradient(quantity_array)) / DT
        if model == 'LIF':
            current_array = trans_param[0] * quantity_array +\
                trans_param[1] * quantity_rate_array + trans_param[2]
            if std is not None:
                std = 0 if std < 0 else std
                current_array += np.random.normal(
                    loc=0., scale=std, size=quantity_array.shape)
        if model == 'Lesniak':
            trans_param = np.tile(trans_param, (4, 1))
            trans_param[:, :2] = np.multiply(
                trans_param[:, :2].T, mcnc_grouping).T
            quantity_array = np.tile(quantity_array, (mcnc_grouping.size, 1)).T
            quantity_rate_array = np.tile(
                quantity_rate_array, (mcnc_grouping.size, 1)).T
            current_array = np.multiply(quantity_array, trans_param[:, 0]) +\
                np.multiply(quantity_rate_array, trans_param[:, 1]) +\
                np.multiply(np.ones_like(quantity_array), trans_param[:, 2])
            if std is not None:
                std = 0 if std < 0 else std
                current_array += np.random.normal(loc=0., scale=std,
                                                  size=quantity_array.shape)
        return current_array

    def trans_param_to_fsl(self, quantity_dict, trans_param, model='LIF',
                           mcnc_grouping=None, std=None):
        current_array = self.trans_param_to_current_array(
            quantity_dict, trans_param, model=model,
            mcnc_grouping=mcnc_grouping, std=std)
        spike_array = self.current_array_to_spike_array(
            current_array, model=model, mcnc_grouping=mcnc_grouping)
        spike_timings = spike_array.nonzero()[0] * DT
        if spike_timings.size > 0:
            fsl = spike_timings[0]
        else:
            fsl = np.inf
        return fsl

    def get_fr_fsl(self, quantity_dict_list, trans_param, model='LIF',
                   mcnc_grouping=None):
        """
        Returns
        -------
        frs : ndarray
            An array of static firing rate, Hz.
        frd : ndarray
            Dynamic firing rate, Hz.
         fsl : ndarray
            First spike latency, seconds.
        """
        frs, frd = self.trans_param_to_predicted_fr(
            quantity_dict_list, trans_param, model=model,
            mcnc_grouping=mcnc_grouping).T[1:]
        fsl = self.trans_param_to_predicted_fsl(
            quantity_dict_list, trans_param, model=model,
            mcnc_grouping=mcnc_grouping).T[1]
        return frs, frd, fsl

    def trans_param_to_predicted_fsl(self, quantity_dict_list, trans_param,
                                     model='LIF', mcnc_grouping=None,
                                     std=None):
        predicted_fsl = []
        for quantity_dict in quantity_dict_list:
            fsl = self.trans_param_to_fsl(
                quantity_dict, trans_param, model=model,
                mcnc_grouping=mcnc_grouping, std=std)
            predicted_fsl.append(fsl)
        predicted_fsl = np.c_[range(len(quantity_dict_list)),
                              predicted_fsl]
        return predicted_fsl

    def trans_param_to_predicted_cov(self, quantity_dict_list, trans_param,
                                     model='LIF', mcnc_grouping=None,
                                     std=None):
        predicted_cov = []
        for quantity_dict in quantity_dict_list:
            cov = self.trans_param_to_cov(
                quantity_dict, trans_param, model=model,
                mcnc_grouping=mcnc_grouping, std=std)
            predicted_cov.append(cov)
        predicted_cov = np.c_[range(len(quantity_dict_list)),
                              predicted_cov]
        return predicted_cov

    def trans_param_to_predicted_fr(self, quantity_dict_list, trans_param,
                                    model='LIF', mcnc_grouping=None, std=None):
        """
        Different between trans_param_to_predicted_fr and trans_param_to_fr:
        trans_param_to_fr is for one quantity trace
        trans_param_to_predicted_fr is for all quantity traces
        """
        predicted_static_fr, predicted_dynamic_fr = [], []
        for quantity_dict in quantity_dict_list:
            static_fr, dynamic_fr = self.trans_param_to_fr(
                quantity_dict, trans_param, model=model,
                mcnc_grouping=mcnc_grouping, std=std)
            predicted_static_fr.append(static_fr)
            predicted_dynamic_fr.append(dynamic_fr)
        predicted_fr = np.c_[range(len(quantity_dict_list)),
                             predicted_static_fr,
                             predicted_dynamic_fr]
        return predicted_fr

    def trans_param_to_fr_r2(self, trans_param, quantity_dict_list,
                             target_fr_array):
        predicted_fr = self.trans_param_to_predicted_fr(
            quantity_dict_list, trans_param)
        predicted_fr_array = np.empty_like(target_fr_array)
        for i in range(predicted_fr_array.shape[0]):
            predicted_fr_array[i, 0] = target_fr_array[i, 0]
            predicted_fr_array[i, 1] = predicted_fr[
                int(target_fr_array[i, 0]), 1]
            predicted_fr_array[i, 2] = predicted_fr[
                int(target_fr_array[i, 0]), 2]
        static_r2 = get_r2(target_fr_array[:, 1], predicted_fr_array[:, 1])
        dynamic_r2 = get_r2(target_fr_array[:, 2], predicted_fr_array[:, 2])
        r2 = .5 * (static_r2 + dynamic_r2)
        print(r2)
        return r2

    def trans_param_to_fr_r2_fitting(self, fitx, trans_param_init,
                                     quantity_dict_list,
                                     target_fr_array, sign=1.):
        trans_param = fitx * trans_param_init
        r2 = self.trans_param_to_fr_r2(trans_param, quantity_dict_list,
                                       target_fr_array)
        return sign * r2

    def get_lstsq_fit(self, quantity_dict_list, target_fr_array):
        # Get the mean of all firing rates at different indentation depth
        target_fr_vector = np.empty(
            2 * (target_fr_array[:, 0].max().astype(np.int) + 1))
        for i in range(target_fr_vector.size // 2):
            target_fr_vector[i] = target_fr_array[:, 1][
                target_fr_array[:, 0] == i].mean()
            target_fr_vector[target_fr_vector.size // 2 + i] = \
                target_fr_array[:, 2][
                    target_fr_array[:, 0] == i].mean()
        # Get the corresponding current vector
        target_current_vector = np.empty_like(target_fr_vector)
        for i, fr in enumerate(target_fr_vector):
            target_current_vector[i] = self.fr2current(fr)
        # Get the quantity and rate matrix
        (static_mean_quantity_array, static_mean_quantity_rate_array,
            dynamic_mean_quantity_array, dynamic_mean_quantity_rate_array
         ) = get_mean_quantity_and_rate(quantity_dict_list)
        quantity_and_rate_matrix = np.c_[
            np.r_[static_mean_quantity_array, dynamic_mean_quantity_array],
            np.r_[static_mean_quantity_rate_array,
                  dynamic_mean_quantity_rate_array],
            np.ones(target_fr_vector.size)]
        trans_param = np.linalg.lstsq(quantity_and_rate_matrix,
                                      target_current_vector)[0]
        return trans_param

    def fit_trans_param(self, quantity_dict_list, target_fr_array):
        trans_param_init = self.get_lstsq_fit(
            quantity_dict_list, target_fr_array)
        res = minimize(self.trans_param_to_fr_r2_fitting, np.ones(3),
                       args=(trans_param_init, quantity_dict_list,
                             target_fr_array, -1.), method='SLSQP',
                       options={'eps': 1e-3})
        trans_param = res.x * trans_param_init
        return trans_param

    def fit_noise(self, trans_param, quantity_dict_list, target_cov,
                  model='LIF', mcnc_grouping=None):
        def get_abs_err(fitx, std0, target_cov):
            std = fitx * std0
            predicted_cov = self.trans_param_to_predicted_cov(
                quantity_dict_list, trans_param, model=model,
                mcnc_grouping=mcnc_grouping, std=std)
            mean_cov = predicted_cov.T[1].mean()
            err = abs(mean_cov - target_cov)
            return err

        def avg_abs_err(fitx, std0, target_cov, n):
            """
            Runs n repetitions of get_abs_err and get average.
            """
            err = np.empty(n)
            for i in range(n):
                err[i] = get_abs_err(fitx, std0, target_cov)
            print(fitx, err.mean())
            return err.mean()

        std0 = self.fr2current(50.) * .5
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots()
        n = 50
        std_array = np.empty(n)
        err_array = np.empty(n)
        for i in range(n):
            pass
        return std


def get_r2(target_array, predicted_array):
    ssres = ((target_array - predicted_array) ** 2).sum()
    sstot = target_array.var() * target_array.size
    r2 = 1. - ssres / sstot
    return r2


def get_mean_quantity_and_rate(quantity_dict_list):
    static_mean_quantity_list, static_mean_quantity_rate_list,\
        dynamic_mean_quantity_list, dynamic_mean_quantity_rate_list =\
        [], [], [], []
    for i, quantity_dict in enumerate(quantity_dict_list):
        # Unpack FEM outputs
        quantity_array = quantity_dict['quantity_array']
        max_index = quantity_dict['max_index']
        max_index -= (quantity_array <= 0).sum()
        quantity_array = quantity_array[quantity_array > 0]
        quantity_rate_array = np.abs(np.gradient(quantity_array)) / DT
        # Get time windows
        static_window = np.arange(max_index + STATIC_START / DT,
                                  max_index + STATIC_END / DT, dtype=np.int)
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


if __name__ == '__main__':
    pass
