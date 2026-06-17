#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023

import numpy as np
from MSM import *
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import openmm
import config

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 

##############################################################
# all MSM related functions now using MSM class in MSM.py
##############################################################

def try_and_optim_M(M, working_indices, num_gaussian=10, start_index=0, end_index=0, qspace = config.qspace, plot = False):
    #print("inside try and optim_M")
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params

    input:
    M: the working transition matrix, square matrix.
    working_indices: the indices of the working states.
    num_gaussian: number of gaussian functions to use.
    start_state: the starting state. note this has to be converted into the index space.
    end_state: the ending state. note this has to be converted into the index space.
    index_offset: the offset of the index space. e.g. if the truncated M (with shape [20, 20]) matrix starts from 13 to 33, then the index_offset is 13.
    """
    best_mfpt = np.inf

    start_state_working_index = np.argmin(np.abs(working_indices - start_index))
    end_state_working_index = np.argmin(np.abs(working_indices - end_index))
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    print("converted to xspace that's from:", qspace[working_indices[start_state_working_index]], "to", qspace[working_indices[end_state_working_index]])
    
    upper = qspace[working_indices[-1]]
    lower = qspace[working_indices[0]]
    print("upper bound:", upper, "lower bound:", lower)

    #initialize msm object
    msm = MSM()
    msm.qspace = qspace
    msm.build_MSM_from_M(M, dim=1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
    msm._compute_peq_fes_M()
    energy_gap = msm.free_energy.max() - msm.free_energy.min()
    qspace_low = qspace[0]
    qspace_high = qspace[-1]

    for try_num in range(1000): 
        #we initialize the msm object.
        msm.build_MSM_from_M(M, dim=1, time_step = config.dcdfreq_mfpt * config.stepsize.value_in_unit(openmm.unit.nanoseconds))
       
        rng = np.random.default_rng()
        a = np.ones(num_gaussian) * 0.6
        b = rng.uniform(qspace_low, qspace_high, num_gaussian)
        c = rng.uniform(0.7, 1, num_gaussian)
        
        total_bias = np.zeros_like(msm.qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, a[j], b[j], c[j])
        #project the total_bias into the pca space.
        working_bias = total_bias[working_indices]
        msm._bias_M(working_bias, method = "direct_bias")
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M()
        mfpts_biased = msm.mfpts

        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]
        #print(peq)
        #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
        if try_num % 100 == 0:
            print("random try:", try_num, "mfpt:", mfpt_biased)
            msm._kemeny_constant_check()
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (3*num_gaussian,)

    print("best mfpt:", best_mfpt)
    if False: 
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, best_params[j], best_params[j+num_gaussian], best_params[j+2*num_gaussian])
        working_bias = total_bias[working_indices]
        x = qspace[working_indices]
        plt.plot(qspace[working_indices], working_bias, label="working bias")
        unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
        #we take first quarter.
        unb_bins = unb_bins[:len(unb_bins)//4]
        unb_profile = unb_profile[:len(unb_profile)//4]
        plt.plot(unb_bins, unb_profile, label="unbiased F")
        #plot the total_bias
        plt.plot(qspace, total_bias, label="total bias", alpha=0.3)
        plt.xlim(2.0, 9)
        plt.legend()
        plt.savefig(f"./bias_plots_fes_best_param_prop{i}.png")
        plt.close()
    
    def mfpt_helper(params, M, start_state = start_index, end_state = end_index, kT=0.5981, working_indices=working_indices):
        msm.build_MSM_from_M(M, dim=1)

        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, a[j], b[j], c[j])
        working_bias = total_bias[working_indices]
        
        msm._bias_M(working_bias, method = "direct_bias")
        msm._compute_peq_fes_M()
        msm._build_mfpt_matrix_M()
        mfpts_biased = msm.mfpts
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        return mfpt_biased

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   #method='Nelder-Mead', 
                   method="L-BFGS-B",
                   bounds= [(0.1, 1.0)]*config.num_gaussian + [(config.qspace_low, config.qspace_high)]*config.num_gaussian + [(0.7, 2)]*config.num_gaussian, #add bounds to the parameters
                   tol=1e-6)
    return res.x    #, best_params

def apply_fes(system, particle_idx, gaussian_param=None, pbc = False, name = "FES", amp = 7, mode = "gaussian", plot = False, plot_path = "./fes_visualization.png"):
    """
    this function apply the bias given by the gaussian_param to the system.
    """
    pi = np.pi #we need convert this into nm.
    k = 5
    max_barrier = '1e3'
    offset = 0.4
    left_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp({k} * x - (-{offset}))))")
    right_pot = openmm.CustomExternalForce(f"{max_barrier} * (1 / (1 + exp(-{k} * (x - (2 * {pi} + {offset})))))")
    left_pot.addParticle(particle_idx)
    right_pot.addParticle(particle_idx)
    system.addForce(left_pot)
    system.addForce(right_pot)

    #unpack gaussian parameters
    if mode == "gaussian":
        num_gaussians = int(len(gaussian_param)/5)
        A = gaussian_param[0::5] * amp #*7
        x0 = gaussian_param[1::5]
        y0 = gaussian_param[2::5]
        sigma_x = gaussian_param[3::5]
        sigma_y = gaussian_param[4::5]

        #now we add the force for all gaussians.
        energy = "0"
        force = openmm.CustomExternalForce(energy)
        for i in range(num_gaussians):
            if pbc:
                energy = f"A{i}*exp(-periodicdistance(x,0,0, x0{i},0,0)^2/(2*sigma_x{i}^2) - periodicdistance(0,y,0, 0,y0{i},0)^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)
            else:
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2) - (y-y0{i})^2/(2*sigma_y{i}^2))"
                force = openmm.CustomExternalForce(energy)

            #examine the current energy term within force.

            print(force.getEnergyFunction())

            force.addGlobalParameter(f"A{i}", A[i])
            force.addGlobalParameter(f"x0{i}", x0[i])
            force.addGlobalParameter(f"y0{i}", y0[i])
            force.addGlobalParameter(f"sigma_x{i}", sigma_x[i])
            force.addGlobalParameter(f"sigma_y{i}", sigma_y[i])
            force.addParticle(particle_idx)
            #we append the force to the system.
            system.addForce(force)
        if plot:
            #plot the fes.
            x = np.linspace(0, 2*np.pi, 100)
            y = np.linspace(0, 2*np.pi, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.zeros_like(X)
            for i in range(num_gaussians):
                Z += A[i] * np.exp(-(X-x0[i])**2/(2*sigma_x[i]**2) - (Y-y0[i])**2/(2*sigma_y[i]**2))
            plt.figure()
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
            plt.xlabel("x")
            plt.xlim([-1, 2*np.pi+1])
            plt.ylim([-1, 2*np.pi+1])
            plt.ylabel("y")
            plt.title("FES mode = gaussian, pbc=False")
            plt.colorbar()
            plt.savefig(plot_path)
            plt.close()
            fes = Z

    if mode == "multiwell":
        """
        here we create a multiple well potential.
         essentially we deduct multiple gaussians from a flat surface, 
         with a positive gaussian acting as an additional barrier.
         note we have to implement this into openmm CustomExternalForce.
            the x,y is [0, 2pi]
         eq:
            U(x,y) = amp * (1                                                                   #flat surface
                            - A_i*exp(-(x-x0i)^2/(2*sigma_xi^2) - (y-y0i)^2/(2*sigma_yi^2))) ...        #deduct gaussians
                            + A_j * exp(-(x-x0j)^2/(2*sigma_xj^2) - (y-y0j)^2/(2*sigma_yj^2))       #add a sharp positive gaussian
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for multi-well potential.")
        else:
            num_hills = 9

            #here's the well params
            A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
            x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
            sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

            #now we add the force for all gaussians.
            #note all energy is in Kj/mol unit.
            energy = str(amp * 4.184) #flat surface
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            for i in range(num_hills):
                energy = f"A{i}*exp(-(x-x0{i})^2/(2*sigma_x{i}^2))"
                force = openmm.CustomExternalForce(energy)

                #examine the current energy term within force.
                print(force.getEnergyFunction())
                force.addGlobalParameter(f"A{i}", A_i[i] * 4.184) #convert kcal to kj
                force.addGlobalParameter(f"x0{i}", x0_i[i])
                force.addGlobalParameter(f"sigma_x{i}", sigma_x_i[i])
                force.addParticle(particle_idx)
                #we append the force to the system.
                system.addForce(force)
            
            if plot:
                #plot the fes.
                x = config.qspace
                Z = np.zeros_like(x)
                for i in range(num_hills):
                    Z += A_i[i] * np.exp(-(x-x0_i[i])**2/(2*sigma_x_i[i]**2))

                #add the x boundary barrier in plot
                Z += float(max_barrier) * (1 / (1 + np.exp(k * (x - (-offset))))) #left
                Z += float(max_barrier) * (1 / (1 + np.exp(-k * (x - (2 * pi + offset))))) #right

                plt.figure()
                plt.plot(x, Z - Z.min(), label="multiwell FES")
                plt.xlabel("x")
                plt.ylabel("FES (kcals/mol)")
                plt.ylim([0, 12])
                plt.xlim([0, 2*np.pi])
                plt.title("FES mode = 1D multiwell, pbc=False")
                plt.savefig(plot_path)
                plt.close()
                fes = Z
            
    if mode == "funnel":
        """
        this is funnel like potential.
        we start wtih a flat fes, then add/deduct sphrical gaussians
        eq:
            U = 0.7* amp * cos(2 * p * (sqrt((x-pi)^2 + (y-pi)^2))) #cos function. periodicity determines the num of waves.
            - amp exp(-((x-pi)^2+(y-pi)^2))
            + 0.4*amp*((x-pi/8)^2 + (y-pi/8)^2)
        """
        if pbc:
            raise NotImplementedError("pbc not implemented for funnel potential.")
        else:
            plot_3d = False
            periodicity = 8
            energy = f"0.7*{amp} * cos({periodicity} * (sqrt((x-{pi})^2 + (y-{pi})^2))) - 0.6* {amp} * exp(-((x-{pi})^2+(y-{pi})^2)) + 0.4*{amp}*((x-{pi}/8)^2 + (y-{pi}/8)^2)"
            
            force = openmm.CustomExternalForce(energy)
            force.addParticle(particle_idx)
            system.addForce(force)
            if plot:
                if plot_3d:
                    import plotly.graph_objs as go

                    # Define the x, y, and z coordinates
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.9* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z -= 0.6* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2)/0.5)
                    Z += 0.4*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    # Create the 3D contour plot
                    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, cmin = 0, cmax = amp *12/7)])
                    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
                    fig.update_layout(title='FES mode = funnel, pbc=False', autosize=True,
                                    width=800, height=800,
                                    scene = {
                                        "xaxis": {"nticks": 5},
                                        "yaxis": {"nticks": 5},
                                        "zaxis": {"nticks": 5},
                                        "camera_eye": {"x": 1, "y": 1, "z": 0.4},
                                        "aspectratio": {"x": 1, "y": 1, "z": 0.4}
                                    }
                                    )
                                    #margin=dict(l=65, r=50, b=65, t=90))
                    #save fig.
                    fig.write_image(plot_path)
                    fes = Z
                    
                else:
                    #plot the fes.
                    x = np.linspace(0, 2*np.pi, 100)
                    y = np.linspace(0, 2*np.pi, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = np.zeros_like(X)
                    Z += 0.4* amp * np.cos(periodicity * (np.sqrt((X-np.pi)**2 + (Y-np.pi)**2))) #cos function. periodicity determines the num of waves.
                    Z += 0.7* amp * np.exp(-((X-np.pi)**2/0.5+(Y-np.pi)**2/0.5))
                    Z += 0.2*amp*(((X-np.pi)/8)**2 + ((Y-np.pi)/8)**2)

                    plt.figure()
                    plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], vmin=0, vmax=amp *12/7, origin="lower")
                    plt.xlabel("x")
                    plt.xlim([-1, 2*np.pi+1])
                    plt.ylim([-1, 2*np.pi+1])
                    plt.ylabel("y")
                    plt.title("FES mode = funnel, pbc=False")
                    plt.colorbar()
                    plt.savefig(plot_path)
                    plt.close()
                    fes = Z

    return system, fes #return the system and the fes (2D array for plotting.)

def apply_bias(system, particle_idx, gaussian_param, pbc = False, name = "BIAS", num_gaussians = 20):
    """
    this applies a bias using customexternal force class. similar as apply_fes.
    note this leaves a set of global parameters Ag, x0g, sigma_xg
    as these parameters can be called and updated later.
    note this is done while preparing the system before assembling the context.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    energy = "0"
    force = openmm.CustomExternalForce(energy)
    for i in range(num_gaussians):
        if pbc:
            raise NotImplementedError("pbc not implemented for gaussian potential.")
            energy = f"Ag{i}*exp(-periodicdistance(x,0,0, x0g{i},0,0)^2/(2*sigma_xg{i}^2) - periodicdistance(0,y,0, 0,y0g{i},0)^2/(2*sigma_yg{i}^2))"
            force = openmm.CustomExternalForce(energy)
        else:
            energy = f"Ag{i}*exp(-(x-x0g{i})^2/(2*sigma_xg{i}^2))" #in openmm unit, kj/mol, nm.
            force = openmm.CustomExternalForce(energy)

        #examine the current energy term within force.

        print(force.getEnergyFunction())

        force.addGlobalParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        force.addGlobalParameter(f"x0g{i}", x0[i]) 
        force.addGlobalParameter(f"sigma_xg{i}", sigma_x[i])
        force.addParticle(particle_idx)
        #we append the force to the system.
        system.addForce(force)
    
    print("system added with bias.")
    return system

def update_bias(simulation, gaussian_param, name = "BIAS", num_gaussians = 20):
    """
    given the gaussian_param, update the bias
    note this requires the context object. or a simulation object.
    # the context object can be accessed by simulation.context.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we update the GlobalParameter for all gaussians. with num_gaussians terms. and update them in the system.
    #note globalparameter does NOT need to be updated in the context.
    for i in range(num_gaussians):
        simulation.context.setParameter(f"Ag{i}", A[i] * 4.184) #convert to kJ/mol
        simulation.context.setParameter(f"x0g{i}", x0[i])
        simulation.context.setParameter(f"sigma_xg{i}", sigma_x[i])
    
    print("system bias updated")
    return simulation

def get_total_bias(x, gaussian_param, num_gaussians = 20):
    """
    this function returns the total bias given the gaussian_param.
    note this is used for plotting the total bias.
    """
    assert len(gaussian_param) == 3 * num_gaussians, "gaussian_param should be in A, x0, sigma_x, format."

    #unpack gaussian parameters gaussian_params = np.concatenate((a, bx, cx))
    num_gaussians = len(gaussian_param)//3
    A = gaussian_param[:num_gaussians]
    x0 = gaussian_param[num_gaussians:2*num_gaussians]
    sigma_x = gaussian_param[2*num_gaussians:3*num_gaussians]

    #now we add the force for all gaussians. with num_gaussians terms.
    total_bias = np.zeros_like(x)
    for i in range(num_gaussians):
        total_bias += A[i] * np.exp(-(x-x0[i])**2/(2*sigma_x[i]**2))
    
    return total_bias
