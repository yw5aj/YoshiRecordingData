import os

# SI units
DT = 1E-5 # sec
RESISTANCE_LIF = 10 * 5e8 # ohm
CAPACITANCE_LIF = 1e-11 # F
VOLTAGE_THRESHOLD_LIF = 30e-3 # V

# Time window for static phase
STATIC_START = 2.
STATIC_END = 4.5

# Total fiber number
fiber_mech_fname = '2012042702V_01.mat'
def get_fiber_tot_num():
    fiber_tot_num = 0
    for fname in os.listdir('./cleandata/rawData/finalSAI'):
        if fname.endswith('.mat'):
            fiber_tot_num += 1
        if fname == fiber_mech_fname:
            fiber_mech_id = fiber_tot_num - 1
    return fiber_tot_num, fiber_mech_id 
FIBER_TOT_NUM, FIBER_MECH_ID = get_fiber_tot_num()
FIBER_FIT_ID_LIST = [FIBER_MECH_ID]

# Plotting constants
MARKER_LIST = ['.', 'o', 'v', '^', '<', '>', '8', 's', 'p', '*']
COLOR_LIST = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'r', 'g', 'b']
LS_LIST = ['-', '--', '-.', ':'] 
MS = 6

# Evalutation levels
EVAL_DISPL = 500 # in um
EVAL_FORCE = 6 # in mN