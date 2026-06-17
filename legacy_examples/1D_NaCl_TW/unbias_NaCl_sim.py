#here we do super long NaCl unbiased run.
# and log the total steps used to reach 7A.
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
import mdtraj
from util import *
sys.path.append("..")
from openmm.app import *
#from openmm.unit import *
#from openmm.unit import *
from dham import DHAM
import csv
import os
import sys
from PIL import Image
import time
from tqdm import tqdm

def minimize(context):
    st = time.time()
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in range(50):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    print("Minimization done in", time.time() - s, "seconds")
    s = time.time()
    return context, energy

platform = omm.Platform.getPlatformByName('CUDA') #CUDA
psf_file = 'toppar/step3_input.psf' # Path #tpr #prmtop
pdb_file = 'toppar/step3_input25A.pdb' # Path # gro # inpcrd
T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # MD integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
num_simulations = 20
sim_steps = 5000000 # simulate for 10 ns

if __name__ == "__main__":

    psf = omm_app.CharmmPsfFile(psf_file)
    pdb = omm_app.PDBFile(pdb_file)
    #forcefield = omm_app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    params = omm_app.CharmmParameterSet('toppar/toppar_water_ions.str') #we modified the combined LJ term between NaCl to have a -6.0kcal.mol at 2.5A

    for i_sim in tqdm(range(num_simulations)):
        
        system = psf.createSystem(params,
                                  nonbondedCutoff=1.0*unit.nanometers,
                                  constraints=omm_app.HBonds)
        
        platform = omm.Platform.getPlatformByName('CUDA')
        
        #### setup an OpenMM context
        integrator = omm.LangevinIntegrator(T*unit.kelvin, #Desired Integrator
                                            fricCoef/unit.picoseconds,
                                            stepsize*unit.femtoseconds)

        NaCl_dist = [] #initialise the NaCl distance list.
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        context = omm.Context(system, integrator)

        context, energy = minimize(context)

        file_handle = open(f'trajectories/unbiased/{time_tag}_unbiased_traj.dcd', 'wb')
        dcd_file = omm_app.DCDFile(file_handle, pdb.topology, dt = stepsize)

        #for _ in tqdm(range(int(sim_steps/dcdfreq))):
        while len(NaCl_dist) == 0 or NaCl_dist[-1] < 7e-1:
            integrator.step(dcdfreq)
            state = context.getState(getPositions=True)
            dcd_file.writeModel(state.getPositions(asNumpy=True))
            #determine the distance between Na and Cl
            pos0 = state.getPositions(asNumpy=True)[0]
            pos1 = state.getPositions(asNumpy=True)[1]
            NaCl_dist.append(np.linalg.norm(pos0-pos1))

        file_handle.close()

        #process and get the first NaCl distance.
        #traj = mdtraj.load(f'trajectories/unbiased/{time_tag}_unbiased_traj.dcd', top=pdb_file)
        #NaCl_dist = mdtraj.compute_distances(traj, [[0, 1]]) #this is in nm

        """        
        for index_d, d in enumerate(NaCl_dist):
            if d > 7e-1:
                steps_to_7A = index_d*dcdfreq
                break
            if index_d == len(NaCl_dist)-1:
                steps_to_7A = index_d*dcdfreq
        """
        steps_to_7A = len(NaCl_dist)*dcdfreq

        with open("total_steps_unbiased.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_7A])
        #save the NaCl_dist
        np.savetxt(f"visited_state/{time_tag}_NaCl_unbias_traj.txt", NaCl_dist)

    
    
    
    
    
    
    
