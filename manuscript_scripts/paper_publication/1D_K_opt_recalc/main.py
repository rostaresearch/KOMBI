#By Tiejun Wei 05th Feb 2024.
#this use a array type of simulation.
#Given full K matrix, calculate optimal bias for each position (from 0, to target state)
#plot the optimal bias for each position, and the biased FES.
#note this is a recalc. we were using matlab version fes.

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from MSM import MSM

if __name__ == "__main__":
    N = 100

    #use FES from 1D_langevin
    amp = 6
    num_hills = 9
    A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

    x = np.linspace(0, 2 * np.pi, N)
    y = np.zeros(N)
    for i in range(num_hills):
        y += A_i[i] * np.exp(-((x - x0_i[i]) ** 2) / (2 * sigma_x_i[i] ** 2))
    y = y - np.min(y)

    #note we didn't apply sigmoid in 1D.
    #sigmoid of the FES
    k = 5
    max_barrier = '1e2' #note this is kJ/mol, we need to convert this to kcal/mol when plotting.
    offset = 0.4
    y += float(max_barrier)/4.184 * (1 / (1 + np.exp(k * (x - (-offset)))))
    y += float(max_barrier)/4.184 * (1 / (1 + np.exp(-k * (x - (2 * np.pi + offset)))))
    
    #we create K matrix based on y.
    kBT = 0.5981
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            u_ij = y[j] - y[i]
            K[i, j] = np.exp(u_ij/ (2*kBT))
            K[j, i] = np.exp(-u_ij/ (2*kBT))
        K[i, i] = -np.sum(K[:,i])

    #test K matrix
    msm = MSM()
    msm.K = K
    msm.num_states = N
    msm.num_dimensions = 1
    msm.qspace = np.linspace(0, 2 * np.pi, N)

    #msm._compute_peq_fes_K()
    #msm._plot_fes(filename='fes_K.png')

    #now we iterate through each position, and calculate the optimal bias.
    def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 
    
    
    
    scan_start = 2.0
    scan_end = 5.2 
    num_gaussian = 20
    time_step = 0.01
    msm.time_step = time_step
    #find state in qspace.
    state_start = np.argmin(np.abs(msm.qspace - scan_start))
    state_end = np.argmin(np.abs(msm.qspace - scan_end))
    
    optim_bias_param_list = []
    for i in range(state_start, state_end):
        print('start optimizing bias for state: ', i)
        #calculate
        optim_bias_param = None
        best_mfpt = np.inf
        #calculate the optimal bias for each position.
        for i_try in range(1000):
            msm.K = K
            rng = np.random.default_rng()
            a = np.ones(num_gaussian) * 2
            b = rng.uniform(0, 2 * np.pi, num_gaussian)
            c = rng.uniform(0, 0.5, num_gaussian)

            total_bias = np.zeros_like(msm.qspace)
            for j in range(num_gaussian):
                total_bias += gaussian(msm.qspace, a[j], b[j], c[j])
            
            msm._bias_K(total_bias)
            msm._compute_peq_fes_K()
            msm._build_mfpt_matrix_K()
            mfpts_biased = msm.mfpts
            mfpt_biased = mfpts_biased[i, state_end]

            if i_try % 100 == 0:
                print('i_try: ', i_try, 'best_mfpt: ', best_mfpt, 'mfpt_biased: ', mfpt_biased)
                if True:
                    plt.figure()
                    plt.plot(msm.qspace, total_bias)
                    plt.plot(msm.qspace, y - np.min(y))
                    plt.plot(msm.qspace, (y + total_bias) - np.min(y + total_bias),  alpha=0.5, linestyle='--')
                    plt.plot(msm.qspace, msm.free_energy- np.min(msm.free_energy),  alpha=0.5, linestyle='--')
                    plt.title('mfpt_biased: ' + str(mfpts_biased[i, state_end]))
                    plt.savefig('bias_' + str(i_try) + '.png')
                    plt.close()
            if best_mfpt > mfpt_biased:
                best_mfpt = mfpt_biased
                best_bias_param = np.concatenate((a, b, c))
            
        #we now optimize it using scipy.
        def mfpt_helper(params, K, start_state, end_state, kT=0.5981, ):
            msm.K = K
            a = params[:num_gaussian]
            b = params[num_gaussian:2*num_gaussian]
            c = params[2*num_gaussian:]
            total_bias = np.zeros_like(msm.qspace)
            for j in range(num_gaussian):
                total_bias += gaussian(msm.qspace, a[j], b[j], c[j])
            
            msm._bias_K(total_bias)
            msm._compute_peq_fes_K()
            msm._build_mfpt_matrix_K()
            mfpts_biased = msm.mfpts
            mfpt_biased = mfpts_biased[i, state_end]

            return mfpt_biased

        from scipy.optimize import minimize
        print("minimalizing for state: ", i)
        res = minimize(mfpt_helper, 
                       best_bias_param, 
                       args=(K,
                             i, 
                             state_end), 
                        method='Nelder-Mead',
                        bounds = [(0.1, 2.0)] * num_gaussian + [(0, 2 * np.pi)] * num_gaussian + [(0.6, 2)] * num_gaussian,
                        tol = 1e-4)
        
        optim_bias_param_list.append(res.x)
        print('optim_bias_param generated for state: ', i)

    #save
    optim_bias_param_list = np.array(optim_bias_param_list)
    np.savetxt('optim_bias_param_list.txt', optim_bias_param_list)
    print('optim_bias_param_list saved.')

    if True:
        plt.figure()
        plt.plot(x, y)
        plt.savefig('FES.png')
        plt.close()
