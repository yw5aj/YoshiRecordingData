import os

# SI units
DT = 1E-4  # sec
RESISTANCE_LIF = 15e8  # ohm
CAPACITANCE_LIF = 1e-11  # F
VOLTAGE_THRESHOLD_LIF = 30e-3  # V

# Time window for static phase
STATIC_START = 2.
STATIC_END = 4.5

# Target COV for SAI, from Wellnitz et al., 0.78 ± 0.09
COV = .78

# Total fiber number
fiber_mech_fname = '2012042702V_01.mat'
fiber_hmstss_fname = '2012042702V_01.mat'


def get_fiber_tot_num():
    fiber_dict = {}
    fiber_tot_num = 0
    for fname in os.listdir('./cleandata/rawData/finalSAI'):
        if fname.endswith('.mat'):
            fiber_dict[fiber_tot_num] = fname
            fiber_tot_num += 1
        if fname == fiber_mech_fname:
            fiber_mech_id = fiber_tot_num - 1
        if fname == fiber_hmstss_fname:
            fiber_hmstss_id = fiber_tot_num - 1
    return fiber_tot_num, fiber_mech_id, fiber_hmstss_id, fiber_dict
FIBER_TOT_NUM, FIBER_MECH_ID, FIBER_HMSTSS_ID, FIBER_DICT = get_fiber_tot_num()
# FIBER_FIT_ID_LIST = [FIBER_MECH_ID]
FIBER_FIT_ID_LIST = range(FIBER_TOT_NUM)

# Add the LIF parameters to each fiber by file name
FIBER_RCV = {}
for key in FIBER_DICT:
    if FIBER_DICT[key] == '2012030905V_01.mat':
        FIBER_RCV[key] = {
            'r': 40e8, 'c': CAPACITANCE_LIF, 'v': VOLTAGE_THRESHOLD_LIF}
    elif FIBER_DICT[key] == '2012042001V_01.mat':
        FIBER_RCV[key] = {
            'r': 60e8, 'c': CAPACITANCE_LIF, 'v': VOLTAGE_THRESHOLD_LIF}
    elif FIBER_DICT[key] == '2012042702V_01.mat':
        FIBER_RCV[key] = {
            'r': 15e8, 'c': CAPACITANCE_LIF, 'v': VOLTAGE_THRESHOLD_LIF}

# Plotting constants
MARKER_LIST = ['v', 'D', 'o', 's', '.', '*', '.', 'x', 'h', '+']
COLOR_LIST = ['k', 'r', 'g', 'b', 'c', 'm', 'y', 'r', 'g', 'b']
LS_LIST = ['-', '--', '-.', ':']
MS = 6

# Evalutation levels
EVAL_DISPL = 500  # in um
EVAL_FORCE = 6  # in mN
