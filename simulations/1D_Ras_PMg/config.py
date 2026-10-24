#config for the Ras (1D_PMg) KOMBI runs.
#values marked "as run" are those of the original 2024 production runs.
#explore_sim_PMg.py overrides the per-run settings from its command line and writes the
#values it used to <outdir>/config_used.json.

import time
from openmm import unit
import numpy as np

#simulation settings
num_sim = 1                                 #runs per process; one run per job (production runs: 15)
pbc = False                                 #as run: no periodic boundaries
time_tag = time.strftime("%Y%m%d-%H%M%S")
platform_name = 'CUDA'                      #as run (OpenMM 8.0 CUDA)
precision = 'mixed'                         #production runs used 'single' (the CUDA default)
load_global_gaussian_params_from_txt = False

#system (0-based atom indices; checked by name at start-up)
psf_file = 'step3_input.psf'
pdb_file = 'traj_0.restart.pdb'             #common start structure of every run (no box)
toppar_file = 'toppar.str'
mg_index = 7799                             #MG, residue 168, segment HETA
p_index = 7840                              #P of H2PO4-, residue 169, segment HETC
nonbonded_cutoff = 1.0                      #nm. no PBC: CutoffNonPeriodic (reaction field), as run
pme_cutoff = 1.2                            #nm, PBC runs only: as the NAMD preparation of this system (KOMBI SI.5:
pme_switch = 1.0                            #nm  PME, vdW switch 10-12 A); 1.0/0.9 is an alternative setup

#MD settings
T = 300 #unit in kelvin
propagation_step = 50000                    #100 ps, as run
stepsize = 0.002 * unit.picoseconds
dcdfreq_mfpt = 1                            #the CV is recorded every MD step, as run
fricCoef = 1/unit.picoseconds   # friction coefficient in 1/ps

#digitization & DHAM settings
qspace_num_bins = 100
qspace_low = 1
qspace_high = 8

use_dynamic_bins = False
DHAM_num_bins = 100                         #used in the DHAM to discretize the traj

qspace = np.linspace(qspace_low, qspace_high, qspace_num_bins)
dham_lag_steps = 1                          #lag of one MD step (2 fs), as run

#biasing settings
num_gaussian = 10
amp_bounds = (0.1, 1.0)                     #kcal/mol per Gaussian, as run: total <= 10
center_bounds = (qspace_low, qspace_high)   #A, as run
width_bounds = (0.7, 2.0)                   #A, as run
random_start_amp = 0.6                      #kcal/mol, amplitude of the 1000 random starts, as run
random_start_width = (0.7, 1.0)             #A, as run
n_random_starts = 1000                      #as run
initial_position = 3.7                      #A, centre of the tiny random bias of propagation 0, as run
initial_amp = 0.01                          #kcal/mol per Gaussian in propagation 0, as run
target_distance = 8.0                       #A, KOMBI's target, as run
mfpt_method = "state_reduction"             #optimiser's model MFPT: stable state reduction; "legacy_diag" = as run
                                            #(fundamental matrix / eig; fails when the target's peq < ~1e-16)

#run control
num_propagation = 400                       #max propagation index, as run (so at most 401 propagations)

#output
dcd_full_ps = 10.0                          #full-system frames (as run: every 0.2 ps)
dcd_solute_ps = 1.0                         #everything except the bulk water (segment SOLV)
shell_log_ps = 1.0                          #Mg2+ first-shell log
bias_check_tol = 1e-3                       #kcal/mol, plus bias_check_rtol * |V|
bias_check_rtol = 1e-4
