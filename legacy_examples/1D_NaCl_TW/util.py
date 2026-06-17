#util file for applying mfpt method on 1-d NaCl system.
#by TW 26th July 2023

import numpy as np
from scipy.linalg import logm, expm
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sys 
import config

def gaussian(x, a, b, c): #self-defined gaussian function
        return a * np.exp(-(x - b)**2 / ((2*c)**2)) 

def create_K_1D(fes, N=200, kT=0.5981):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    
    #input:
    #fes: the free energy profile, a 1D array.

    K = np.zeros((N,N), dtype=np.float64) #, dtype=np.float64
    for i in range(N-1):
        K[i, i + 1] = np.exp((fes[i+1] - fes[i]) / 2 / kT)
        K[i + 1, i] = np.exp((fes[i] - fes[i+1]) / 2 / kT)
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K

def kemeny_constant_check(N, mfpt, peq):
    kemeny = np.zeros((N, 1))
    for i in range(N):
        for j in range(N):
            kemeny[i] = kemeny[i] + mfpt[i, j] * peq[j]
    print("Performing Kemeny constant check...")
    print("the min/max of the Kemeny constant is:", np.min(kemeny), np.max(kemeny))
    """
    if np.max(kemeny) - np.min(kemeny) > 1e-6:
        print("Kemeny constant check failed!")
        raise ValueError("Kemeny constant check failed!")"""
    return kemeny

#define a function calculating the mean first passage time
def mfpt_calc(peq, K):
    """
    peq is the probability distribution at equilibrium.
    K is the transition matrix.
    N is the number of states.
    """
    N = K.shape[0] #K is a square matrix.
    onevec = np.ones((N, 1)) #, dtype=np.float64
    Qinv = np.linalg.inv(peq.T * onevec - K.T)

    mfpt = np.zeros((N, N)) #, dtype=np.float64
    for j in range(N):
        for i in range(N):
            #to avoid devided by zero error:
            if peq[j] == 0:
                mfpt[i, j] = 0
            else:
                mfpt[i, j] = 1 / peq[j] * (Qinv[j, j] - Qinv[i, j])
    
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def bias_K_1D(K, total_bias, kT=0.5981):
    """
    K is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix K_biased.
    """
    N = K.shape[0]
    K_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N-1):
        u_ij = total_bias[i+1] - total_bias[i]  # Calculate u_ij (Note: Indexing starts from 0)
        K_biased[i, i+1] = K[i, i+1] * np.exp(u_ij /(2*kT))  # Calculate K_biased
        K_biased[i+1, i] = K[i+1, i] * np.exp(-u_ij /(2*kT))

    for i in range(N):
        K_biased[i,i] = -np.sum(K_biased[:,i])
    return K_biased

def compute_free_energy(K, kT=0.5981):
    """
    K is the transition matrix
    kT is the thermal energy
    peq is the stationary distribution #note this was defined as pi in Simian's code.
    F is the free energy
    eigenvectors are the eigenvectors of K

    first we calculate the eigenvalues and eigenvectors of K
    then we use the eigenvalues to calculate the equilibrium distribution: peq.
    then we use the equilibrium distribution to calculate the free energy: F = -kT * ln(peq)
    """
    evalues, evectors = np.linalg.eig(K)

    #sort the eigenvalues and eigenvectors
    index = np.argsort(evalues) #sort the eigenvalues, the largest eigenvalue is at the end of the list
    evalues_sorted = evalues[index] #sort the eigenvalues based on index

    #calculate the equilibrium distribution
    peq = evectors[:, index[-1]].T #normalize the eigenvector
    peq = peq / np.sum(peq)
    peq = peq.real

    F = -kT * np.log(peq) #add a small number to avoid log(0)) # + 1e-16

    return [peq, F, evectors, evalues, evalues_sorted, index]

def compute_free_energy_power_method(K, kT=0.5981):
    """
    this use the power method to calculate the equilibrium distribution.
    num_iter is the number of iterations.
    """
    num_iter = 1000
    N = K.shape[0]
    peq = np.ones(N) / N #initialise the peq
    for i in range(num_iter):
        peq = np.dot(peq, K)
        peq = peq / np.sum(peq)
    F = -kT * np.log(peq)
    return [peq, F]

def try_and_optim(K=None, num_gaussian=10, start_state=0, end_state=0):
   #print("inside try_and_optim")
    """
    here we try different gaussian params 1000 times
    and use the best one (lowest mfpt) to local optimise the gaussian_params
    
    returns the best gaussian params
    """

    #we construct K from loading the fes in npy files.
    fes = np.load("Unbiased_Profile.npy")
    K = create_K_1D(fes, N=config.num_bins+1, kT=0.5981)

    best_mfpt = 1e12 #initialise the best mfpt np.inf
    for i in range(1000): 
        #print("in for loop")
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian)
        b = rng.uniform(1, 9, num_gaussian) #min/max of preloaded NaCl fes x-axis.
        c = rng.uniform(1, 5.0, num_gaussian) 
        
        qspace = np.linspace(2.0, 9, config.num_bins+1)
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            #print("adding params")
            total_bias += gaussian(qspace, a[j], b[j], c[j])
        
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]

        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        print(sum(peq))
        kemeny_constant_check(K.shape[0], mfpts_biased, peq)
        #print("random try:", i, "mfpt:", mfpt_biased)
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = (a, b, c)
            
    #print("best mfpt:", best_mfpt)
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, K, kT=0.5981, start_state = start_state, end_state = end_state):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros(50)
        for j in range(num_gaussian):
            total_bias += gaussian(np.arange(50), a[j], b[j], c[j])
        K_biased = bias_K_1D(K, total_bias, kT=0.5981)
        peq = compute_free_energy(K_biased, kT=0.5981)[0]
        mfpts_biased = mfpt_calc(peq, K_biased)
        mfpt_biased = mfpts_biased[start_state, end_state]
        return mfpt_biased

    res = minimize(mfpt_helper, 
                   best_params, 
                   args=(K,), 
                   method='Nelder-Mead', 
                   bounds= [(0, 1)]*10 + [(1, 9)]*10 + [(1.0, 5.0)]*10, #add bounds to the parameters
                   tol=1e-1)

    return res.x, best_params


def bias_M_1D(M, total_bias, kT=0.5981):
    """
    M is the unperturbed transition matrix.
    total_bias is the total biasing potential.
    kT is the thermal energy.
    This function returns the perturbed transition matrix M_biased.
    """
    N = M.shape[0]
    M_biased = np.zeros([N, N])#, #dtype=np.float64)

    for i in range(N):
        for j in range(N):
            u_ij = total_bias[j] - total_bias[i]
            M_biased[i, j] = M[i, j] * np.exp(-u_ij / 2*kT)
        M_biased[i, i] = M[i,i]

    """for i in range(N):
        if np.sum(M_biased[:, i]) != 0:
            M_biased[:, i] = M_biased[:, i] / np.sum(M_biased[:, i])
        else:
            M_biased[:, i] = 0"""
    #M_biased = M_biased/(np.sum(M_biased, axis=1)[:,None])# + 1e-15)
    for i in range(M_biased.shape[0]):
        row_sum = np.sum(M_biased[i, :])
        if row_sum > 0:
            M_biased[i, :] = M_biased[i, :] / row_sum
        else:
            M_biased[i, :] = 0
    return M_biased.real

#below Markov_mfpt_calc is provided by Sam M.
def Markov_mfpt_calc(peq, M):
    N = M.shape[0]
    onevec = np.ones((N, 1))
    Idn = np.diag(onevec[:, 0])
    A = (peq.reshape(-1, 1)) @ onevec.T #was peq.T @ onevec.T
    A = A.T
    Qinv = np.linalg.inv(Idn + A - M)
    mfpt = np.zeros((N, N))
    for j in range(N):
        for i in range(N):
            term1 = Qinv[j, j] - Qinv[i, j] + Idn[i, j]
            if peq[j] * term1 == 0:
                mfpt[i, j] = 1000000000000
            else:
                mfpt[i, j] = 1/peq[j] * term1
    #result = kemeny_constant_check(N, mfpt, peq)
    return mfpt


def try_and_optim_M(M, working_indices, num_gaussian=10, start_state=0, end_state=0, plot = False):
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

    #first we convert the big index into "index to the working indices".

    #start_state_working_index = np.where(working_indices == start_state)[0][0] #convert start_state to the offset index space.
    #end_state_working_index = np.where(working_indices == end_state)[0][0] #convert end_state to the offset index space.
    start_state_working_index = np.argmin(np.abs(working_indices - start_state))
    end_state_working_index = np.argmin(np.abs(working_indices - end_state))
    print("optimizing to get g_param from start state:", start_state_working_index, "to end state:", end_state_working_index, "in working indices.")
    
    #now our M/working_indices could be incontinues. #N = M.shape[0]
    qspace = np.linspace(2.0, 9, config.num_bins+1) #hard coded for now.
    best_mfpt = 1e20 #initialise the best mfpt np.inf

    for i in range(1000): 
        rng = np.random.default_rng()
        #we set a to be 1
        a = np.ones(num_gaussian) #* 1.5
        b = rng.uniform(0, 10, num_gaussian) #fix this so it place gaussian at the working indices. it has to be in angstrom. because we need return these.
        #b = rng.uniform(b_min, b_max, num_gaussian)
        c = rng.uniform(0.3, 0.7, num_gaussian) 
        
        #we convert the working_indices to the qspace.

        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        total_bias = total_bias[working_indices]
        
        M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(M_biased.T, kT=0.5981)
        #peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]
        #print(peq)
        #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
        if i % 100 == 0:
            print("random try:", i, "mfpt:", mfpt_biased)
            #kemeny_constant_check(M.shape[0], mfpts_biased, peq)
            #we plot the F.
            
        if best_mfpt > mfpt_biased:
            best_mfpt = mfpt_biased
            best_params = np.concatenate((a, b, c)) #we concatenate the params into a single array. in shape (30,)

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
    
    #now we use the best params to local optimise the gaussian params

    def mfpt_helper(params, M, start_state = start_state, end_state = end_state, kT=0.5981, working_indices=working_indices):
        a = params[:num_gaussian]
        b = params[num_gaussian:2*num_gaussian]
        c = params[2*num_gaussian:]
        total_bias = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias += gaussian(qspace, a[j], b[j], c[j])

        #now we need to convert the total_bias to the working_indices space.
        total_bias = total_bias[working_indices]

        M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(M_biased.T, kT=0.5981)
        #peq, F = compute_free_energy_power_method(M_biased, kT=0.5981)
        mfpts_biased = Markov_mfpt_calc(peq, M_biased)
        mfpt_biased = mfpts_biased[start_state_working_index, end_state_working_index]

        return mfpt_biased

    res = minimize(mfpt_helper, #minimize comes from scipy.
                   best_params, #gaussian params
                   args=(M,
                         start_state_working_index, 
                         end_state_working_index,
                         working_indices), 
                   #method='Nelder-Mead', 
                   method='L-BFGS-B',
                   bounds= [(0.1, 2)]*10 + [(0,10)]*10 + [(0.3, 1.5)]*10, #add bounds to the parameters
                   tol=1e-3)

    if plot:
        a = res.x[:num_gaussian]
        b = res.x[num_gaussian:2*num_gaussian]
        c = res.x[2*num_gaussian:]

        total_bias = np.zeros_like(qspace[working_indices])
        for j in range(num_gaussian):
            total_bias += gaussian(qspace[working_indices], a[j], b[j], c[j])
        
        M_biased = bias_M_1D(M, total_bias, kT=0.5981)
        [peq, F, evectors, evalues, evalues_sorted, index] = compute_free_energy(M_biased, kT=0.5981)

        total_bias_A = np.zeros_like(qspace)
        for j in range(num_gaussian):
            total_bias_A += gaussian(qspace, a[j], b[j], c[j])

        plt.title("Scipy optimized F with ground truth")
        plt.plot(qspace[working_indices], F, label = "reconstructed biased M FES")
        plt.plot(qspace, total_bias_A, label="total bias in A space.", linestyle="--", linewidth=0.5)
        plt.plot(qspace[working_indices], total_bias, label="total bias in working indices", linewidth=1)
        unb_bins, unb_profile = np.load("Unbiased_Profile.npy")
        plt.plot(unb_bins, unb_profile, label="unbiased FES (ground truth)")
        plt.plot(qspace[working_indices[start_state_working_index]], F[start_state_working_index], "o", label="start state")
        plt.plot(qspace[working_indices[end_state_working_index]], F[end_state_working_index], "x", label="end state")
        plt.legend()
        plt.show()

    return res.x    #, best_params
