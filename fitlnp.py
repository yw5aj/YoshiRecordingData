# -*- coding: utf-8 -*-
"""
Created on Tue May  5 10:58:47 2015

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
from constants import DT
from scipy.optimize import minimize


def apply_linear_filter(stimuli, linear_filter):
    stimuli_rate = np.r_[0, np.diff(stimuli) / DT]
    response = (np.convolve(stimuli_rate, linear_filter) *
                DT)[:stimuli_rate.size]
    return response


def get_stimuli(params_k, stress):
    stress_rate = np.r_[0, np.diff(stress) / DT]
    stimuli = params_k[0] * stress + params_k[1] * np.abs(stress_rate)
    return stimuli


def get_linear_filter(params_prony, time):
    g1, tau1 = params_prony
    ginf = 1 - g1
    linear_filter = g1 * np.exp(-time / tau1) + ginf
    return linear_filter


def stress2response(params_k, params_prony, stress, time):
    stimuli = get_stimuli(params_k, stress)
    linear_filter = get_linear_filter(params_prony, time)
    response = apply_linear_filter(stimuli, linear_filter)
    return response


def sse_stress_response(fitx, params_init,
                        stress, time, target_time, target_response):
    params = fitx * params_init
    params_k = params[:2]
    params_prony = params[2:]
    response = stress2response(params_k, params_prony, stress, time)
    interp_response = np.interp(target_time, time, response)
    sse = ((interp_response - target_response) ** 2).sum()
    return sse


def sse_whole_fiber(fitx, params_init, fit_input_list):
    sse = 0
    for fit_input in fit_input_list:
        sse += sse_stress_response(fitx, params_init, **fit_input)
    return sse


def fit_stress_response(stress, time, target_time, target_response):
    bounds = ((0, None), (0, None), (0, 1), (0, None))
    params_init = (100/stress.max(), 5/stress.max(), .5, .5)
    res = minimize(
        sse_stress_response, np.ones(4),
        args=(params_init, stress, time, target_time, target_response),
        method='SLSQP', bounds=bounds)
    params_hat = res.x * params_init
    return params_hat


def fit_whole_fiber(fit_input_list):
    bounds = ((0, None), (0, None), (0, 1), (0, None))
    params_init = np.array((1e-2, 1e-3, .5, .5))
    res = minimize(
        sse_whole_fiber, np.ones(4), args=(params_init, fit_input_list),
        method='SLSQP', bounds=bounds)
    params_hat = res.x * params_init
    return params_hat


if __name__ == '__main__':
    time, stress = np.loadtxt('./csvs/test_lnp_time_stress.csv',
                              delimiter=',').T
    spike_time_aggregate, spike_fr_aggregate = np.loadtxt(
        './csvs/test_lnp_target.csv', delimiter=',').T
    plt.plot(spike_time_aggregate, spike_fr_aggregate, '.', color='.5')
    # %%
    params_k = (100/stress.max(), 5/stress.max())
    params_prony = (.5, .5)
#    plt.plot(time, stress2response(params_k, params_prony, stress, time))
    res = fit_stress_response(stress, time,
                              spike_time_aggregate, spike_fr_aggregate)
    plt.plot(time, stress2response(res[:2], res[2:], stress, time))
    # %%
    import pickle
    with open('./pickles/test_lnp_fit_input.pkl', 'rb') as f:
        fit_input_list = pickle.load(f)
    fit_whole_fiber(fit_input_list)
# array([ 0.01257859,  0.00078881,  0.49779717,  0.59975368])
