import numpy as np
import matplotlib.pyplot as plt
from exp2fem import Fiber
from cleandata.convert import CleanFiber
from constants import (FIBER_TOT_NUM, FIBER_MECH_ID, MARKER_LIST, COLOR_LIST,
                       LS_LIST, MS)
from fitlnp import fit_whole_fiber, stress2response


class RelaxAdapt(Fiber):

    def __init__(self, fiber_id, make_plot=False):
        Fiber.__init__(self, fiber_id, make_plot)
        # Generate the data from experiment and model
        self.get_stim_block_trace_exp()
        self.get_stim_block_trace_fem()
        self.aggregate_fr_exp()

    def aggregate_fr_exp(self):
        for i, stim_group in enumerate(self.stim_group_dict):
            self.stim_group_dict[i]['spike_time_aggregate'] = []
            self.stim_group_dict[i]['spike_fr_aggregate'] = []
            self.stim_group_dict[i]['spike_isi_aggregate'] = []
            for traces_exp in stim_group['traces_exp']:
                self.stim_group_dict[i]['spike_time_aggregate'].extend(
                    traces_exp['spike_time'])
                self.stim_group_dict[i]['spike_fr_aggregate'].extend(
                    traces_exp['spike_fr'])
                self.stim_group_dict[i]['spike_isi_aggregate'].extend(
                    traces_exp['spike_isi'])
            # Remove the off-response
            indices = np.array(self.stim_group_dict[i]['spike_time_aggregate']
                               ) <= 5
            self.stim_group_dict[i]['spike_time_aggregate'] = np.array(
                self.stim_group_dict[i]['spike_time_aggregate'])[indices]
            self.stim_group_dict[i]['spike_fr_aggregate'] = np.array(
                self.stim_group_dict[i]['spike_fr_aggregate'])[indices]
            self.stim_group_dict[i]['spike_isi_aggregate'] = np.array(
                self.stim_group_dict[i]['spike_isi_aggregate'])[indices]

    def plot_inst_fr(self, lnp_params=None):
        n = int(np.sqrt(len(self.stim_group_dict)))
        fig, axs = plt.subplots(n, n, figsize=(3.27 * n, 3 * n))
        for i, stim_group in enumerate(self.stim_group_dict):
            axes = axs.ravel()[i]
            axes.plot(stim_group['spike_time_aggregate'],
                      stim_group['spike_fr_aggregate'],
                      '.', color='.5')
            if lnp_params is not None:
                response = stress2response(
                    lnp_params[:2], lnp_params[2:],
                    stim_group['traces_fem']['stress'],
                    stim_group['traces_fem']['time'])
                axes.plot(stim_group['traces_fem']['time'],
                          response, '-k')
            axes.set_xlim(0, 5)
        return fig, axs


if __name__ == '__main__':
    # Switches
    make_plot = False
    # Instantiate fibers
    relaxAdaptList = []
    for i in range(FIBER_TOT_NUM):
        relaxAdapt = RelaxAdapt(i, make_plot=make_plot)
        relaxAdaptList.append(relaxAdapt)
    relaxAdaptTypical = relaxAdaptList[FIBER_MECH_ID]
    params_hat = np.array([ 0.01214418,  0.00082817,  0.5       ,  0.77081547])
    relaxAdaptTypical.plot_inst_fr(lnp_params=params_hat)
    # Construct the fit_input_list to be fitted
    fit_input_list = []
    for stim_group in relaxAdapt.stim_group_dict:
        fit_input_list.append({
            'stress': stim_group['traces_fem']['stress'],
            'time': stim_group['traces_fem']['time'],
            'target_time': stim_group['spike_time_aggregate'],
            'target_response': stim_group['spike_fr_aggregate']})
