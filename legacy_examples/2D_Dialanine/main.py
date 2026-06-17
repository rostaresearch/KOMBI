#here we use the "explore_bias_NaCl_gen.py" style, explor the FES of theoretical 2D system
#by TW 9th Aug.
from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
from scipy.linalg import expm

import sys
import time
from tqdm import tqdm
from dham import DHAM

import openmm.app as omm_app
import openmm as omm
import openmm.unit as unit
import mdtraj

plt.rcParams.update({'font.size': 16})


platform = omm.Platform.getPlatformByName('CUDA')
pdb = omm_app.PDBFile('./dialaB.pdb')
T = 298.15      # temperature in K
fricCoef = 10   # friction coefficient in 1/ps
stepsize = 2    # MD integration step size in fs
dcdfreq = 100   # save coordinates at every 100 step
propagation_step = 100000
max_propagation = 10
num_bins = 20 #for qspace used in DHAM and etc.
kT = 0.5981
num_gaussian = 10

target_state = [-110/180*pi, 75/180*pi] # in radian space. [ϕ, ψ]

def propagate(context,
              gaussian_params,
              prop_index, 
              CV_total,
              steps=propagation_step,
              dcdfreq=100, 
              platform=platform, 
              stepsize=stepsize,
              num_bins=num_bins,
              ):
    """
    here we use the openmm context object to propagate the system.
    save the CV and append it into the CV_total.
    use the DHAM_it to process CV_total, get the partially observed Markov matrix from trajectory.
    return the current position, the CV_total, and the partially observed Markov matrix.
    """
    
    file_handle = open(f"./trajectories/dialanine_exploring_traj_{prop_index}.dcd", 'bw')
    dcd_file = omm_app.dcdfile.DCDFile(file_handle, pdb.topology, dt = stepsize)

    for _ in tqdm(range(int(steps/dcdfreq)), desc=f"Propagation {prop_index}"):
        integrator.step(dcdfreq)
        state = context.getState(getPositions=True)
        dcd_file.writeModel(state.getPositions(asNumpy=True))
    file_handle.close()

    #top = mdtraj.load_psf(psf_file)
    traj = mdtraj.load_dcd(f"./trajectories/dialanine_exploring_traj_{prop_index}.dcd", top = './dialaB.pdb')#, top=top)
    
    #get the psi and phi dihedral defined in dialaA.pdb.
    dihe_1 = mdtraj.compute_dihedrals(traj, np.array([[5, 7, 9, 15]])).ravel() #ϕ  (phi)
    dihe_2 = mdtraj.compute_dihedrals(traj, np.array([[7, 9, 15, 17]])).ravel() #ψ  (psi)

    coor = np.stack((dihe_1, dihe_2), axis=1)
    #now we save this into a text file for bookkeeping.
    np.savetxt(f"./trajectories/dialanine_exploring_traj_{prop_index}.txt", coor)
    

    #digitalize the dihedrals into np.meshgrid(np.linspace(-pi,pi, num_bins), np.linspace(-pi,pi, num_bins))
    # then we ravel this [num_bins, num_bins] into 1D index.

    #first we digitalize the dihedrals.
    dihe_1_digital = np.digitize(dihe_1, np.linspace(-pi,pi, num_bins))
    dihe_2_digital = np.digitize(dihe_2, np.linspace(-pi,pi, num_bins))

    #then we ravel it into 1D index.
    dihe_12_digital_ravelled = np.ravel_multi_index((dihe_1_digital, dihe_2_digital), (num_bins, num_bins), order='C')

    #plot the traj of dihedrals
    plt.figure()
    plt.scatter(dihe_1, dihe_2, alpha=0.3)
    plt.title(f"propagation {prop_index}, dihedral traj")
    plt.xlim(-pi, pi)
    plt.ylim(-pi, pi)
    plt.show()

    """ 
    plt.figure()
    plt.scatter(dihe_1_digital, dihe_2_digital, alpha=0.3, color='red')
    plt.show()
    """

    #we store the combined CV into CV_total.
    dihe_12_digital_ravelled_total = np.concatenate((CV_total[-1], dihe_12_digital_ravelled))
    CV_total.append(dihe_12_digital_ravelled_total)

    #here we use the DHAM.
    F_M, MM = DHAM_it(dihe_12_digital_ravelled_total.reshape(-1,1), gaussian_params, T=300, lagtime=1, numbins=num_bins)
    
    return dihe_12_digital_ravelled[-1], MM, CV_total #return the last position, the MM, CV traj list.


def get_working_MM(M):
    zero_rows = np.where(~M.any(axis=1))[0]
    zero_cols = np.where(~M.any(axis=0))[0]

    keep_indices = np.setdiff1d(range(M.shape[0]), np.union1d(zero_rows, zero_cols))
    M_work = M[np.ix_(keep_indices, keep_indices)]
    return M_work, keep_indices

def get_closest_state(qspace, target_state, working_indices):
    """
    usesage: qspace = np.linspace(2.4, 9, 150+1)
    target_state = 7 #find the closest state to 7A.
    """
    working_states = qspace[working_indices] #the NaCl distance of the working states.
    closest_state = working_states[np.argmin(np.abs(working_states - target_state))]
    return closest_state



def DHAM_it(CV, gaussian_params, T=300, lagtime=1, numbins=90):
    """
    intput:
    CV: the collective variable we are interested in. now it's 2d.
    gaussian_params: the parameters of bias potential. (in our case the 10-gaussian params)
     format: (a,bx, by,cx,cy)
    T: temperature 300

    output:
    the Markov Matrix
    Free energy surface probed by DHAM.
    """
    d = DHAM(gaussian_params)
    d.setup(CV, T)

    d.lagtime = lagtime
    d.numbins = numbins #num of bins, arbitrary.
    results = d.run(biased = True, plot=True)
    return results

def find_closest_index(working_indices, final_index, N):
    """
    returns the farest index in 1D.

    here we find the closest state to the final state.
    first we unravel all the index to 2D.
    then we use the lowest RMSD distance to find the closest state.
    then we ravel it back to 1D.
    note: for now we only find the first-encounted closest state.
          we can create a list of all the closest states, and then choose random one.
    """
    def rmsd_dist(a, b):
        return np.sqrt(np.sum((a-b)**2))
    working_x, working_y = np.unravel_index(working_indices, (N,N), order='C')
    working_states = np.stack((working_x, working_y), axis=1)
    final_state = np.unravel_index(final_index, (N,N), order='C')
    closest_state = working_states[0]
    for i in range(len(working_states)):
        if rmsd_dist(working_states[i], final_state) < rmsd_dist(closest_state, final_state):
            closest_state = working_states[i]
        
    closest_index = np.ravel_multi_index(closest_state, (N,N), order='C')
    return closest_index


def MDminimize(context):
    st = time.time()
    s = time.time()
    print("Setting up the simulation")

    # Minimizing step
    context.setPositions(pdb.positions)
    state = context.getState(getEnergy = True)
    energy = state.getPotentialEnergy()

    for _ in tqdm(range(50), desc="Minimization"):
        omm.openmm.LocalEnergyMinimizer.minimize(context, 1, 20)
        state = context.getState(getEnergy = True)
        energy = state.getPotentialEnergy()

    print("Minimization done in", time.time() - s, "seconds")
    s = time.time()
    return context, energy
###############################################
# here we start the main python process:
# propagate -> acess the Markov Matrix -> biasing -> propagate ...
###############################################

if __name__ == "__main__":
    
    #psf = omm_app.CharmmPsfFile('./dialaA.psf')
    pdb = omm_app.PDBFile('./dialaB.pdb')
    
    forcefield = omm_app.ForceField('amber14-all.xml')
    system = forcefield.createSystem(pdb.topology,
                                     nonbondedCutoff=1.0*unit.nanometers,
                                     constraints=omm_app.HBonds)
    platform = omm.Platform.getPlatformByName('CUDA')
    #### setup an OpenMM context
    integrator = omm.LangevinIntegrator(T*unit.kelvin, #Desired Integrator
                                        fricCoef/unit.picoseconds,
                                        stepsize*unit.femtoseconds) 
    #create discretized CV space, from -pi to pi, with num_bins.
    x,y = np.meshgrid(np.linspace(-pi,pi, num_bins), np.linspace(-pi,pi, num_bins))
    CV_total = [[]] #initialise the CV list.

    #note from now on, all index is in raveled 'flattened' form.
    for i_prop in range(max_propagation):
        if i_prop == 0:
            
            print("propagation number 0 STARTING.")
            gaussian_params = random_initial_bias_2d(initial_position = [-1.0, 1.0], num_gaussians=num_gaussian)

            #here we create CMAP_force object and will use this to update the gaussian_params later on.
            a = gaussian_params[:num_gaussian]
            bx = gaussian_params[num_gaussian:2*num_gaussian]
            by = gaussian_params[2*num_gaussian:3*num_gaussian]
            cx = gaussian_params[3*num_gaussian:4*num_gaussian]
            cy = gaussian_params[4*num_gaussian:]

            x,y = np.meshgrid(np.linspace(-pi,pi, num_bins), np.linspace(-pi,pi, num_bins))
            total_bias = get_total_bias_2d(x,y, gaussian_params) 
            #try transpose the totalbias.
            
            plt.figure()
            plt.contourf(x,y,(total_bias - total_bias.min()).reshape(num_bins,num_bins).T)
            plt.colorbar()
            plt.title("total_bias before rolling to 0-2pi")
            plt.show()
            
            total_bias = np.roll(total_bias, num_bins//2, axis=0) #now we shift the total_bias from (-pi, pi) to (0, 2pi)
            total_bias = np.roll(total_bias, num_bins//2, axis=1)
            total_bias = np.ravel(total_bias, order='F') #ravel it into 1D as required by addMap method.
            
            #plot the total_bias for visual check
            """x_2pi, y_2pi = np.meshgrid(np.linspace(0,2*pi, num_bins), np.linspace(0,2*pi, num_bins))
            plt.figure()
            plt.contourf(x_2pi, y_2pi,(np.reshape(total_bias - total_bias.min(), [num_bins, num_bins], order='F')))
            plt.colorbar()
            plt.title("total_bias after rolling to 0-2pi")
            plt.show()
            """
            cmap_force = omm.CMAPTorsionForce()
            map_idx = cmap_force.addMap(num_bins, total_bias)
            tor_idx = cmap_force.addTorsion(map_idx, 5,7,9,15, 7,9,15,17)

            system.addForce(cmap_force)
            context = omm.Context(system, integrator)
            cmap_force.updateParametersInContext(context)

            context, energy = MDminimize(context)

            cur_pos, M_reconstructed, CV_total  = propagate(context,
                                                            gaussian_params,
                                                            i_prop, 
                                                            CV_total,)
            
            #our cur_pos is flattened 1D index.
            working_MM, working_indices = get_working_MM(M_reconstructed) #we call working_index the small index. its part of the full markov matrix.
            
            #digitize the target state.
            
            target_state_digital = [np.digitize(target_state[0], np.linspace(-pi,pi, num_bins)),  np.digitize(target_state[1], np.linspace(-pi,pi, num_bins))]
            final_index = np.ravel_multi_index(target_state_digital, (num_bins, num_bins), order='C') #flattened.
            final_index_xy = np.unravel_index(final_index, (num_bins, num_bins), order='C') #2D index.
            #here we find the closest state to the final state.
            # first we unravel all the index to 2D.
            # then we use the lowest manhattan distance to find the closest state.
            # then we ravel it back to 1D.
            closest_index = find_closest_index(working_indices, final_index, num_bins) 
        else:
            print(f"propagation number {i_prop} STARTING.")
            #renew the gaussian params using returned MM.

            gaussian_params = try_and_optim_M(working_MM, 
                                              working_indices = working_indices,
                                              num_gaussian=10, 
                                              start_index=cur_pos, 
                                              end_index=closest_index,
                                              plot = True,
                                              num_bins=num_bins,
                                              )

            #renew the total bias.
            total_bias = get_total_bias_2d(x,y, gaussian_params)
            #note we actually transpose the total_bias. when random try.
            total_bias = total_bias.T
            closest_index_xy = np.unravel_index(closest_index, (num_bins, num_bins), order='C')
            cur_pos_xy = np.unravel_index(cur_pos, (num_bins, num_bins), order='C')
            
            #plot the total bias.
            plt.figure()
            plt.contourf(x,y,(total_bias - total_bias.min()).T)
            plt.plot(x[0][closest_index_xy[0]], y[closest_index_xy[1]][0], marker = 'x')
            plt.plot(x[0][cur_pos_xy[0]], y[cur_pos_xy[1]][0], marker = 'o') #this is local run current position.
            plt.plot(x[0][final_index_xy[0]], y[final_index_xy[1]][0], marker = 'o') #this is target state.
            plt.colorbar()
            plt.show()

            #renew the cmap_force.
            total_bias = np.roll(total_bias, num_bins//2, axis=0) #now we shift the total_bias from (-pi, pi) to (0, 2pi)
            total_bias = np.roll(total_bias, num_bins//2, axis=1)
            total_bias = np.ravel(total_bias, order='F') #ravel it into 1D as required by addMap method

            cmap_force.setMapParameters(map_idx, num_bins, total_bias)
            cmap_force.setTorsionParameters(tor_idx, map_idx, 5,7,9,15, 7,9,15,17)
            cmap_force.updateParametersInContext(context)
           
            #we propagate the system again.
            cur_pos, M_reconstructed, CV_total  = propagate(context,
                                                            gaussian_params,
                                                            i_prop, 
                                                            CV_total,)
            
            #our cur_pos is flattened 1D index.
            working_MM, working_indices = get_working_MM(M_reconstructed) #we call working_index the small index. its part of the full markov matrix.
            
            
            #here we find the closest state to the final state.
            # first we unravel all the index to 2D.
            # then we use the lowest manhattan distance to find the closest state.
            # then we ravel it back to 1D.
            closest_index = find_closest_index(working_indices, final_index, num_bins)

        if closest_index == final_index:
            print(f"we have sampled the final state point, stop propagating at number {i_prop}")
            #here we plot the trajectory. The CV_total[-1]
            pos = np.unravel_index(CV_total[-1].astype(int), (num_bins, num_bins), order='C')
            plt.figure()
            plt.plot(pos[0], pos[1], alpha=0.3)
            #plt.plot(state_start[0], state_start[1], marker = 'x') #this is starting point.
            plt.plot(target_state_digital[0], target_state_digital[1], marker = 'o') #this is ending point.
            plt.show()
            
            break
        else:
            print("continue propagating.")
            continue