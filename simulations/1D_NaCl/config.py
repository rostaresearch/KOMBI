#this is a config file for langevin_sim series.

import os
import time
from openmm import unit
import openmm

from openmm.unit import Quantity
from openmm import Vec3

NUMBER_OF_PROCESSES = 4 #must be lesser than num_sim.
num_sim = 1
sim_steps = int(5e7)
sim_steps_unbiased = int(1e7)
pbc = False #True is not implemented, we got problem fitting periodic function to 2D fes.
time_tag = time.strftime("%Y%m%d-%H%M%S")
amp = 6 #10 #for amp applied on fes. note the gaussian parameters for fes is normalized.
T = 300

propagation_step = 5000
stepsize = 0.002 * unit.picoseconds #equivalent to 2 * unit.femtoseconds 4fs.
stepsize_unbias = 0.2 * unit.picoseconds #100 times.
num_bins = 100 #used to discretize the traj, and used in the DHAM.
dcdfreq = 500
dcdfreq_mfpt = 1

def _platform(name=os.environ.get('OPENMM_PLATFORM', 'CUDA')):
    #CUDA by default; falls back to the fastest available platform (None) so that the
    #utilities can be imported and tested on machines without CUDA.
    try:
        return openmm.Platform.getPlatformByName(name)
    except Exception:
        print(f"OpenMM platform {name} not available, using the fastest available one.")
        return None

platform = _platform()
#platform = openmm.Platform.getPlatformByName('CPU')

#NaCl dissociation target, the same for KOMBI, MetaD and unbiased runs (A).
target_distance = 8.0
#all three methods run at T.
#nonbonded treatment: 'NoCutoff' has no periodic box (the CHARMM-GUI water cube in vacuum).
#'PME' needs box_nm.
nonbonded_method = 'PME'
box_nm = 2.558434
#first-passage resolution of the MetaD and unbiased runs (MD steps between distance checks).
#KOMBI checks every step (dcdfreq_mfpt).
fpt_stride = 10

#reproducible, fine-sampled runs (all off by default):
#  seed: one run seed; the integrator, velocity and optimiser seeds are derived from it and
#        written to <time_tag>_run_info.json (util.run_seeds);
#  deterministic_forces: CUDA DeterministicForces, so that a seeded run can be replayed exactly;
#  fine_every: save all atoms every fine_every MD steps (XTC, nm and ps);
#  ring_ps: the unbiased driver keeps only the last ring_ps of those frames in memory and writes
#           them at the end of the run (the first passage).
seed = int(os.environ['KOMBI_SEED']) if os.environ.get('KOMBI_SEED') else None
deterministic_forces = os.environ.get('KOMBI_DETERMINISTIC', '0') == '1'
fine_every = int(os.environ.get('KOMBI_FINE_EVERY', '0'))
ring_ps = float(os.environ.get('KOMBI_RING_PS', '0'))

num_gaussian = 10 #number of gaussians used to placing the bias.

#starting state (as in coordinate space, from 0 to 2pi.)
start_state = Quantity(value = [Vec3(2.0,0.0,0.0)], unit = unit.nanometers)
end_state = Quantity(value = [Vec3(5.2,0.0,0.0)], unit = unit.nanometers) #need to change.

end_state_unbiased = Quantity(value = [Vec3(3.9,0.0,0.0)], unit = unit.nanometers) #need to change.

#here we have 3 pre-defined 2D fes, stored as different functions.
fes_mode = 'multiwell' #chose from ['gaussian', 'multiwell', 'funnel']
#fes_param_path = ['./params/gaussian_fes_param.txt', './params/multi_well_fes_param.txt', './params/funnel_fes_param.txt']