#this is a config file for langevin_sim series.

import time
from openmm import unit
import openmm

from openmm.unit import Quantity
from openmm import Vec3
import numpy as np

#simulation settings
num_sim = 15
sim_steps = int(5e6)
pbc = False 
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 6 
platform = openmm.Platform.getPlatformByName('CUDA')
start_state = Quantity(value = [Vec3(2.0,0.0,0.0)], unit = unit.nanometers)
end_state = Quantity(value = [Vec3(5.2,0.0,0.0)], unit = unit.nanometers)
fes_mode = 'multiwell'                      #['gaussian', 'multiwell', 'funnel']
load_global_gaussian_params_from_txt = False

#MD settings
T = 300 #unit in kelvin
propagation_step = 50000 #5000
stepsize = 0.002 * unit.picoseconds 
stepsize_unbias = 0.002 * unit.picoseconds 
dcdfreq = 100
dcdfreq_mfpt = 1
backbone_constraint_strength = 1000 #unit in kJ/mol/nm^2
fricCoef = 1/unit.picoseconds   # friction coefficient in 1/ps

#digitization & DHAM settings
qspace_num_bins = 100                       #used for plotting
qspace_low = 1
qspace_high = 8

use_dynamic_bins = False
DHAM_num_bins = 100                         #used in the DHAM to discretize the traj

qspace = np.linspace(qspace_low, qspace_high, qspace_num_bins)

#biasing settings
num_gaussian = 10 


#back n forth settings
max_cycle = 20                              #we will run max_iteration times of back n forth.
num_propagation = 400                       #max number of propagations 