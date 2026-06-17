#here we do metaD simulation for NaCl in water.
# the CV is the distance between Na and Cl.
# and log the total steps used to reach 7A.
import numpy as np
import matplotlib.pyplot as plt

import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
from openmm.app import *

import mdtraj
import csv
import time
import os
from tqdm import tqdm

import config

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
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        system = psf.createSystem(params,
                                  nonbondedCutoff=1.0*unit.nanometers,
                                  constraints=omm_app.HBonds)
        
        #metaD
        bias_bond = omm.CustomBondForce("r")
        bias_bond.addBond(0, 1)
        dist_cv = BiasVariable(bias_bond, 2.0, 10, 0.2, config.pbc)
        
        aux_file_path = os.path.join("./MetaD_auxfiles/" + time_tag)
        if not os.path.exists(aux_file_path):
            os.makedirs(aux_file_path)
        metaD = Metadynamics(system=system, 
                            variables=[dist_cv], #define the cv
                            temperature=300*unit.kelvin,
                            biasFactor=2.0,
                            height=1.0*unit.kilojoules_per_mole,
                            frequency=1000, #every xxx step we place a bias.
                            saveFrequency=1000,
                            biasDir=aux_file_path)
        
        platform = omm.Platform.getPlatformByName('CUDA')
        integrator = omm.LangevinIntegrator(T*unit.kelvin, #Desired Integrator
                                            fricCoef/unit.picoseconds,
                                            stepsize*unit.femtoseconds)

        NaCl_dist = [[]] #initialise the NaCl distance list.
        #simulation object, note the context object is automatically created.
        sim = Simulation(pdb.topology, system, integrator, platform=platform)
        sim.context.setPositions(pdb.positions)

        #minimize the energy
        context, energy = minimize(sim.context)
        sim.context = context

        #run the simulation
        file_handle = open(f'trajectories/metaD/{time_tag}_metaD_traj.dcd', 'wb')
        dcd_file = omm_app.DCDFile(file_handle, pdb.topology, dt = stepsize)

        for _ in tqdm(range(int(sim_steps/dcdfreq))):
            metaD.step(sim, dcdfreq)
            state = context.getState(getPositions=True)
            dcd_file.writeModel(state.getPositions(asNumpy=True))
        file_handle.close()

        #here we get the energy as function of CV using metad object.
        energy_CV = metaD.getFreeEnergy()
        np.savetxt(f"visited_state/{time_tag}_energy_CV.txt", energy_CV)

        #process and get the first NaCl distance.
        traj = mdtraj.load(f'trajectories/metaD/{time_tag}_metaD_traj.dcd', top=pdb_file)
        NaCl_dist = mdtraj.compute_distances(traj, [[0, 1]]) #this is in nm

        for index_d, d in enumerate(NaCl_dist):
            if d > 7e-1:
                steps_to_7A = index_d*dcdfreq
                break
            if index_d == len(NaCl_dist)-1:
                steps_to_7A = index_d*dcdfreq

        with open("total_steps_metaD.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([steps_to_7A])
        #save the NaCl_dist
        np.savetxt(f"visited_state/{time_tag}_NaCl_metaD_traj.txt", NaCl_dist)

    
    
    
    
    
    
    
