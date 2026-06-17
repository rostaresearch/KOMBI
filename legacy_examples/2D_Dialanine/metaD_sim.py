#this is a unbiased simulation for di-alanine in TIP3P water box 30 A
#by TW 13rd Oct 2023

from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
from scipy.linalg import expm

import sys
import time
from tqdm import tqdm
from dham import DHAM
from sys import stdout
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit



plt.rcParams.update({'font.size': 16})


platform = omm.Platform.getPlatformByName('CUDA')
T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # MD integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
logfreq = 100
propagation_step = 500000 #we do 1000 ns total, so 500,000 steps

NVT_eq = True #if we do NVT equilibration
NPT_prod = True #if we do NPT production
total_copy = 5

if __name__ == "__main__":
    pdb = omm_app.PDBFile('./dialaB.pdb')

    #we shift the pdb coor to origin so we can do pbc.
    xyz = np.array(pdb.positions/unit.nanometer)
    xyz[:,0] -= np.min(xyz[:,0])
    xyz[:,1] -= np.min(xyz[:,1])
    xyz[:,2] -= np.min(xyz[:,2])
    pdb.positions = xyz*unit.nanometer #update the pdb file

    forcefield = omm_app.ForceField('amber14-all.xml', 'tip3p.xml')
    
    modeller = omm_app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, model='tip3p', boxSize=(30, 30, 30)*unit.angstroms)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=omm_app.PME, nonbondedCutoff=1.2*unit.nanometer, constraints=omm_app.HBonds, rigidWater=True, ewaldErrorTolerance=0.0005)

    integrator = omm.LangevinIntegrator(T*unit.kelvin, fricCoef/unit.picoseconds, stepsize*unit.femtoseconds)
    simulation = omm_app.Simulation(modeller.topology, system, integrator, platform)

    print("setting initial condition")
    simulation.context.setPositions(modeller.positions)
    if not NVT_eq:
        simulation.context.setVelocitiesToTemperature(T*unit.kelvin)

    #minimize the system
    print("minimizing")
    simulation.minimizeEnergy()

    #do 50ns NVT equilibration
    # 50,000 /2 steps = 25,000 steps
    if NVT_eq:
        print("NVT equilibration")
        #WarmUp with a NVT run.  Slowly warm up temperature - every 1000 steps raise the temperature by 5 K, ending at 300 K
        T = 5
        NVT_eq_steps = 25000
        simulation.context.setVelocitiesToTemperature(T*unit.kelvin) #set the initial temperature as 5
        for i in range(60):
            simulation.step(int(NVT_eq_steps/60))
            temperature = (T+(i*T))*unit.kelvin     #gradually increase the temperature to 300 K over 50ns.
            integrator.setTemperature(temperature)

    #NPT production.
    # add barostat
    barostat = system.addForce(omm.MonteCarloBarostat(1*unit.atmospheres, T*unit.kelvin, 25))
    simulation.context.reinitialize(True) #this is to reinitialize the context with the new barostat.

    #we do 1000 ns total, so 500,000 steps
    # we do 5 copies of the MD. also name the traj with time tag.
    print("start simulation")

    time_tag = time.strftime("%Y%m%d-%H%M%S")

    for i in range(total_copy):
        traj_name = './trajectories/unbiased/trajectory_'+time_tag+'_'+str(i)+'.dcd'
        log_name = './trajectories/unbiased/log_'+time_tag+'_'+str(i)+'.txt'
        pdb_name = './trajectories/unbiased/pdb_'+time_tag+'_'+str(i)+'.pdb'


        #we first clear reporter.
        simulation.reporters.clear()
        simulation.reporters.append(omm_app.DCDReporter(traj_name, dcdfreq))
        simulation.reporters.append(omm_app.StateDataReporter(log_name, logfreq, step=True, potentialEnergy=True, temperature=True, density=True, progress=True, remainingTime=True, speed=True, totalSteps=propagation_step, separator='\t'))
        simulation.reporters.append(omm_app.StateDataReporter(stdout, logfreq, step=True, potentialEnergy=True, temperature=True, density=True, progress=True, remainingTime=True, speed=True, totalSteps=propagation_step, separator='\t'))
        simulation.reporters.append(omm_app.PDBReporter(pdb_name, dcdfreq))

        for j in tqdm(range(int(propagation_step/dcdfreq))):
            simulation.step(dcdfreq)
        
        #we save the system setup in xml file.
        with open("./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_system.xml", 'w') as f:
            f.write(omm.XmlSerializer.serialize(system))
        with open("./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_integrator.xml", 'w') as f:
            f.write(omm.XmlSerializer.serialize(integrator))
        with open("./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_state.xml", 'w') as f:
            f.write(omm.XmlSerializer.serialize(simulation.context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True, getParameters=True, enforcePeriodicBox=True)))
        
        print("trajectory saved to "+traj_name)
        print("log saved to "+log_name)
        print("pdb saved to "+pdb_name)
        print("system setup saved to "+"./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_system.xml")
        print("integrator setup saved to "+"./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_integrator.xml")
        print("state setup saved to "+"./saved_system/unbiased/system_"+time_tag+"_"+str(i)+"_state.xml")

        
