#Here we explore the NaCl FES energy landscape on Na-Cl distance. place random bias.
# the workflow:
# propagate the system for a number of steps.
#    we have to start from loading the molecule, FF
#    then we put bias, log the bias form in txt file. (random guess around starting position if first propagation)
#    then we propagate the system for a number of steps.
#    use DHAM, feed in the NaCl distance, bias params for current propagation, get free energy landscape
#    use the partial free energy landscape to generate next bias params.
#    repeat.
## import required packages

import time
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
import mdtraj

from util import *
from openmm.app import *
from dham import DHAM
from tqdm import tqdm
from matplotlib import animation

sys.path.append("..")
psf_file = 'step3_input.psf' # Path
pdb_file = 'traj_0.restart.pdb' #'step3_input.pdb' # Path

minimizing = False #we don't minimize, we use the traj_restart.pdb.

fig, ax = plt.subplots()
env = os.environ.copy()
matrices = []
matricies_analytic =[]
frames = []

# Set up the plot
fig, ax = plt.subplots()


def animate(i):
    plt.imshow(matrices[i], cmap='hot')

def last_dcd_frames(fin_index): 
    print("In lasts dcd frames")
    top = mdtraj.load_psf(psf_file)
    print("Loaded psf file")
    traj = mdtraj.load_dcd(f"trajectories/explore_traj/NaCl_exploring_traj_{fin_index-1}.dcd", top=top)
    print("Loaded dcd file")
    dists = mdtraj.compute_distances(traj, [[0, 1]]) *10 #unit in A #get distance over the traj (in this propagation)
    print("this is fin  index: ", fin_index, "theses are the dists: ", dists)

    for frame, dist in enumerate(dists):
        if dist >= 7:
            print("this is frames:", frame)
            return frame 

def update_metrics(run_type, run_number, frames, time):
    filename = 'mm_metrics.csv'
    file_exists = os.path.isfile(filename)

    # Open the CSV file in append mode ('a'), so new rows can be added at the end
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # If the file didn't exist, write the header
        if not file_exists:
            writer.writerow(['Run Type', 'Run Number', 'NumFrames', 'Time'])

        # Write the new row with the provided data
        writer.writerow([run_type, run_number, frames,time])

def random_initial_bias(initial_position, num_gaussians):
    #returns random a,b,c for 10 gaussian functions. around the initial position
    # initial position is in Anstrom
    rng = np.random.default_rng()
    #a = np.ones(10)
    a = np.ones(num_gaussians) * 0.01 * 4.184 #convert to kJ/mol
    b = rng.uniform(initial_position-0.2, initial_position+0.2, num_gaussians) / 10
    c = rng.uniform(1, 5.0, 10) /10
    return np.concatenate((a,b,c), axis=None)
    
def DHAM_it(CV, global_gaussian_params, T=300, lagtime=2, num_bins=150, prop_index=0, time_tag=None, use_symmetry=True):
    """
    intput:
    CV: the collective variable we are interested in. 
    T: temperature 300
    output:
    DHAM qspace
    F_M: the free energy landscape. Free energy surface probed by DHAM.
    the Markov Matrix
    """
    d = DHAM(global_gaussian_params, num_bins=num_bins)
    d.setup(CV, T, prop_index=prop_index, time_tag=time_tag)

    d.lagtime = lagtime
    results = d.run(biased = True, plot=True, use_symmetry=use_symmetry, use_dynamic_bins=config.use_dynamic_bins)
    return results

def propagate(simulation,
              prop_index,
              global_gaussian_params, 
              PMg_dist, 
              time_tag=None, 
              steps=config.propagation_step, 
              dcdfreq=config.dcdfreq_mfpt, 
              stepsize=config.stepsize, 
              pbc=config.pbc, 
              reach = None,
              top=None,
              ):
    print("propagating")
    print("prop index: ", prop_index)


    #initialization
    dist = []
    
    #Actual MD simulation
    file_handle = open(f"trajectory/explore_traj/{time_tag}_PMg_exploring_traj_{prop_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, psf.topology, dt = stepsize)
    progress_bar = tqdm(total = int(steps/dcdfreq), desc="MD progress")
    for i_chunk in range(int(steps/dcdfreq)):
        if i_chunk % int(steps/dcdfreq/10) == 0:
            progress_bar.update(i_chunk - progress_bar.n)

        simulation.integrator.step(dcdfreq) 
        state = simulation.context.getState(getPositions=True)
        
        pos1 = state.getPositions(asNumpy=True)[7799] #Mg
        pos2 = state.getPositions(asNumpy=True)[7840] #P
        dist.append(np.linalg.norm(pos1-pos2) * 10) #convert from nm to angstrom
        
        if i_chunk % config.dcdfreq == 0: #we only save dcd every dcdfreq steps.
            dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #combine distance list
    combined_dist = np.concatenate((PMg_dist[-1], dist), axis=None)
    PMg_dist.append(combined_dist)
    combined_dist = combined_dist.reshape(prop_index+1, -1) #in shape: [prop_index, dcdfreq]
    print("combined dist to be fed into DHAM in shape: ", combined_dist.shape)

    #load global_gaussian_params
    if config.load_global_gaussian_params_from_txt:
        global_gaussian_params = np.zeros([prop_index+1, config.num_gaussian, 3])
        for i in range(prop_index+1):
            global_gaussian_params[i,:,:] = np.loadtxt(f"./params/{time_tag}_gaussian_param_prop_{i}.txt").reshape(-1,3)
            print(f"gaussian_params for propagation {i} loaded.")

    assert global_gaussian_params is not None, "global_gaussian_params is None, please check."
    assert global_gaussian_params.shape[0] == prop_index+1, "global_gaussian_params shape does not match prop_index+1, please check."
    assert global_gaussian_params.shape[1] == config.num_gaussian, "global_gaussian_params shape does not match num_gaussian, please check."
    assert global_gaussian_params.shape[2] == 3, "global_gaussian_params shape does not match 3 (1D), please check."

    #F_M, MM = DHAM_it(combined_dist.reshape(-1, 1), gaussian_params, T=300, lagtime=1, numbins=num_bins, prop_index = prop_index)
    dham_qspace, F_M, MM = DHAM_it(combined_dist,
                                   global_gaussian_params,
                                   T=config.T,
                                   lagtime=1,
                                   num_bins=config.DHAM_num_bins,
                                   time_tag=time_tag,
                                   prop_index=prop_index,
                                   use_symmetry=True,
                                   )
    #post processing
    np.savetxt(f"distance/{time_tag}_PMg_exploring_traj_{prop_index}.txt", dist) #save the P-Mg distance to a txt file. 
    for index_d, d in enumerate(dist):
        if d >= 8:
            print("at step: ", index_d * dcdfreq, "we reach 8A")
            reach = index_d * dcdfreq

    cur_pos = combined_dist[-1][-1]

    return cur_pos, PMg_dist, MM, reach, F_M, dham_qspace

def minimize(context):
    """
    energy minimization of the MD system object in OpenMM
    """
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in tqdm(range(1000), desc="Minimization"):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 10, 1000)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()
    
    print("Minimization done in", time.time() - s, "seconds")
    return context, energy

def add_bias(system, gaussian_params, num_gaussian=10):
    """
    deprecated, move to util as update_bias().
    """
    a = gaussian_params[:num_gaussian]
    b = gaussian_params[num_gaussian:2*num_gaussian]
    c = gaussian_params[2*num_gaussian:]
    potential = ' + '.join(f'a{i}*exp(-(r-b{i})^2/(2*c{i}^2))' for i in range(num_gaussian))
    custom_bias = omm.CustomBondForce(potential)
    
    for i in range(num_gaussian):
        custom_bias.addGlobalParameter(f'a{i}', a[i])
        custom_bias.addGlobalParameter(f'b{i}', b[i])
        custom_bias.addGlobalParameter(f'c{i}', c[i])
        custom_bias.addBond(7799, 7840)  #Mg 7799, P 7840
    
    system.addForce(custom_bias)
    return system

def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(1.0, 8.0, 100+1)
    target_state = 6 #find the closest state to 6A.
    """

    working_states = qspace[working_indices] #the PMg distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state

def read_params(filename):
    parFiles = ()
    for line in open(filename, 'r'):
        if '!' in line: line = line.split('!')[0]
        parfile = line.strip()
        if len(parfile) != 0: parFiles += ( parfile, )

    params = CharmmParameterSet( *parFiles )
    return params

def add_backbone_posres(system, psf, pdb, params, strength):
    """
    add backbone position restraint to the system.
    """
    force = omm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", strength)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for i, atom_crd in enumerate(pdb.positions):
        if psf.atom_list[i].name in ['N', 'CA', 'C']:
            force.addParticle(i, atom_crd.value_in_unit(unit.nanometer))
    
    system.addForce(force)
    return force
###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":

    #MD setup
    psf = omm_app.CharmmPsfFile(psf_file)
    #psf.setBox(100*unit.angstrom, 100*unit.angstrom, 100*unit.angstrom) #set the box size
    pdb = omm_app.PDBFile(pdb_file)
    params = read_params('toppar.str')

    for i_sim in range(config.num_sim):
        system = psf.createSystem(params, nonbondedMethod=CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometer, constraints=HBonds)
        #posres = add_backbone_posres(system, psf, pdb, params, config.backbone_constraint_strength) #no constraint needed as we are using restart structure.
        integrator = omm.LangevinIntegrator(config.T*unit.kelvin, #Desired Integrator
                                            config.fricCoef,
                                            config.stepsize)
        
        PMg_dist = [[]]
        time_tag = time.strftime("%Y%m%d-%H%M%S")
        reach = None
        i_prop = 0
        total_steps = None

        while reach is None:
            if i_prop > config.num_propagation:
                print("Max propagation reached. Exiting.")
                break
            if i_prop == 0:
                print("propagation number 0 STARTING.")
                gaussian_params = random_initial_bias(initial_position = 3.7, num_gaussians=config.num_gaussian) #initial position is 3.7A #note the random bias will be converted into nm then feed into OpenMM.
                biased_system = add_bias(system, gaussian_params, num_gaussian=config.num_gaussian)

                global_gaussian_params = gaussian_params.reshape(1, config.num_gaussian, 3)
                np.savetxt(f"./params/{time_tag}_gaussian_param_prop_{i_prop}.txt", gaussian_params)

                simulation = omm_app.Simulation(psf.topology, biased_system, integrator, config.platform)
                if minimizing:
                    simulation.context, energy = minimize(simulation.context)         #minimize the system
                else: #we need initialize the position from the pdb file.
                    simulation.context.setPositions(pdb.positions)
                simulation.context.setVelocitiesToTemperature(config.T*unit.kelvin)
                #print(simulation.context.getState().getPeriodicBoxVectors()) #box should be set when we defined the system from psf.
                ## MD run "propagation"
                cur_pos, PMg_dist, M, reach, F_M, dham_qspace = propagate(simulation, 
                                                                global_gaussian_params=global_gaussian_params, 
                                                                PMg_dist = PMg_dist,
                                                                time_tag = time_tag,
                                                                prop_index=i_prop,
                                                                steps=config.propagation_step, 
                                                                dcdfreq=config.dcdfreq_mfpt, 
                                                                stepsize=config.stepsize,
                                                                )
                working_MM, working_indices = get_working_MM(M)
                
                #Balint added this.
                matrices.append(M)
                analytic_matrix = np.linalg.matrix_power(M, 10000)
                matricies_analytic.append(analytic_matrix)
                
                #qspace clean up.
                current_qspace = dham_qspace if config.use_dynamic_bins else config.qspace
                last_visited_state = np.digitize(cur_pos, current_qspace).astype(np.int64) - 1
                final_index = np.digitize(8, current_qspace) - 1
                closest_index = working_indices[np.argmin(np.abs(working_indices - final_index))]
                print("closest_index (local target) updated. converted to x space that's: ", current_qspace[closest_index])
                print("last visited state (local start) updated. converted to x space that's: ", current_qspace[last_visited_state])
                    
                i_prop += 1

            else:
                print(f"propagation number {i_prop} STARTING.")
                
                gaussian_params = try_and_optim_M(working_MM,
                                                    working_indices = working_indices,
                                                    num_gaussian = config.num_gaussian,
                                                    start_index = last_visited_state,
                                                    end_index = closest_index,
                                                    qspace = current_qspace,
                                                    plot = False,
                                                    )

                if True:
                    #plot.
                    total_bias = get_total_bias(config.qspace, gaussian_params, num_gaussians=config.num_gaussian)
                    plt.figure()
                    plt.plot(config.qspace, total_bias, label="total bias applied")
                    plt.plot(current_qspace, F_M, label="DHAM fes")
                    plt.plot(current_qspace[last_visited_state], F_M[last_visited_state], marker='o', markersize=3, color="blue", label = "last visited state (local start)")
                    plt.plot(current_qspace[closest_index], F_M[closest_index], marker='o', markersize=3, color="red", label = "closest state (local target)")

                    #we plot here to check the original fes, total_bias and trajectory.
                    #plt.plot(config.qspace, (fes - fes.min()), label="original fes") #no fes. we can use US result here.
                    plt.xlabel("x-coor position (nm)")
                    plt.ylabel("fes (kcal/mol)")
                    plt.ylim(0, 20)
                    plt.title("total bias and DHAM fes for P-Mg unbinding")

                    history_traj = PMg_dist[-1][:-config.propagation_step]
                    recent_traj = PMg_dist[-1][-config.propagation_step:] #the last propagation

                    history_traj = np.digitize(history_traj, config.qspace) - 1
                    recent_traj = np.digitize(recent_traj, config.qspace) - 1

                    #plt.scatter(config.qspace[history_traj], (fes - fes.min())[history_traj], s=3.5, alpha=0.3, c='grey')
                    #plt.scatter(config.qspace[recent_traj], (fes - fes.min())[recent_traj], s=3.5, alpha=0.8, c='black')
                    plt.legend(loc='upper left')
                    plt.savefig(f"./figs/explore/{time_tag}_PMg_unbinding_prop_{i_prop}.png")
                    plt.close()

                global_gaussian_params = np.concatenate((global_gaussian_params, gaussian_params.reshape(1, config.num_gaussian, 3)), axis=0)
                np.savetxt(f"./params/{time_tag}_gaussian_param_prop_{i_prop}.txt", gaussian_params)

                #update the bias
                for j in range(10):
                    simulation.context.setParameter(f'a{j}', gaussian_params[j] * 4.184) #unit in openmm is kJ/mol, the a is fitted in kcal/mol, so we multiply by 4.184
                    simulation.context.setParameter(f'b{j}', gaussian_params[j+10] / 10) #unit openmm is nm, the b is fitted in A, so we divide by 10.
                    simulation.context.setParameter(f'c{j}', gaussian_params[j+20] / 10) #same to b.
            
                cur_pos, PMg_dist, M, reach, F_M, dham_qspace = propagate(simulation, 
                                                                global_gaussian_params=global_gaussian_params, 
                                                                PMg_dist = PMg_dist,
                                                                time_tag = time_tag,
                                                                prop_index=i_prop,
                                                                steps=config.propagation_step, 
                                                                dcdfreq=config.dcdfreq_mfpt, 
                                                                stepsize=config.stepsize,
                                                                )
                
                working_MM, working_indices = get_working_MM(M)

                #Balint added this.
                matrices.append(M)
                analytic_matrix = np.linalg.matrix_power(M, 10000)
                matricies_analytic.append(analytic_matrix)

                #qspace clean up.
                current_qspace = dham_qspace if config.use_dynamic_bins else config.qspace
                last_visited_state = np.digitize(cur_pos, current_qspace).astype(np.int64) - 1
                final_index = np.digitize(8, current_qspace) - 1
                closest_index = working_indices[np.argmin(np.abs(working_indices - final_index))]
                print("closest_index (local target) updated. converted to x space that's: ", current_qspace[closest_index])
                print("last visited state (local start) updated. converted to x space that's: ", current_qspace[last_visited_state])
                    
                i_prop += 1

        ######### post processing #########
        #we have reached 8A distance.
                
        total_steps = i_prop*config.propagation_step + reach*config.dcdfreq #reach is the index of the last step in the last propagation.
        print("total steps used: ", total_steps)

        with open("total_steps_mfpt.csv", 'a') as f:
            writer = csv.writer(f)
            writer.writerow([total_steps])
        
        np.savetxt(f"distance/{time_tag}_NaCl_exploring_traj.txt", PMg_dist[-1])

    print("Done")

    
    


            
            
    
    
    
    
    
    
