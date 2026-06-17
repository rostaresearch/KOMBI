#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize
from util import gaussian
import config
from MSM import *

def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))

def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]

def count_transitions(b, numbins, lagtime, endpt=None, use_symmetry=False):
    """
    note the b is a 2D array, 
     row represents the trajectory, column represents the time.
    """
    if endpt is None:
        endpt = b
    Ntr = np.zeros(shape=(b.shape[0], numbins, numbins), dtype=np.int64)  # number of transitions
    for k in range(b.shape[0]):
        for i in range(lagtime, b.shape[1]):
            try:
                Ntr[k, b[k, i - lagtime], endpt[k, i]] += 1
            except IndexError:
                continue
    sumtr = np.sum(Ntr, axis=0)
    trvec = np.sum(Ntr, axis=2)
    if use_symmetry:
        print("Using symmetry in DHAM")
        sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym
    else:
        print("Not using symmetry in DHAM")
    # print("Degree of symmetry:",zz
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    return sumtr.real, trvec

class DHAM:
    
    def __init__(self, global_gaussian_params, num_bins):
        #gaussian_params comes in shape [prop_index + 1, num_gaussian, 3]
        self.global_gaussian_params = global_gaussian_params
        self.KbT = 0.001987204259 * 300  # energy unit: kcal/mol
        self.epsilon = 0.00001
        self.data = None
        self.vel = None
        self.datlength = None
        self.k_val = None
        self.constr_val = None
        self.qspace = None
        self.num_bins = num_bins
        self.lagtime = 1
        return

    def setup(self, CV, T, prop_index, time_tag):
        self.data = CV
        self.KbT = 0.001987204259 * T
        self.prop_index = prop_index
        self.time_tag = time_tag
        return

    def build_MM(self, sumtr, trvec, biased=False):
        MM = np.empty(shape=sumtr.shape, dtype=np.longdouble)
        if biased:
            MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
            #qsp = self.qspace[1] - self.qspace[0] #step size between bins
            for i in range(sumtr.shape[0]):
                for j in range(sumtr.shape[1]):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        for k in range(trvec.shape[0]):
                            #the k represents the trajectory index.

                            #for each chunk of trajectory, we load the gaussian parameters.
                            # and calculate total bias in u, used to unbias the traj later.
                            u = np.zeros_like(self.qspace)
                            for g in range(self.global_gaussian_params.shape[1]):
                                a,b,c = self.global_gaussian_params[k, g, :]
                                u += gaussian(self.qspace, a, b, c)
                            
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp(-(u[j] - u[i]) / (2*self.KbT))
                        #if sump1 > 0:
                        MM[i, j] = sumtr[i, j] / sump1
                    else:
                        MM[i, j] = 0
            for i in range(MM.shape[0]):
                row_sum = np.sum(MM[i,:])
                if row_sum > 0:
                    MM[i,:] = MM[i,:] / row_sum
                else:
                    MM[i,:] = 0
        else:
            raise NotImplementedError
        return MM

    def run(self, plot=False, adjust=True, biased=False, conversion=2E-13, use_symmetry=False, use_dynamic_bins=False):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        #given the data, we create dynamic qspace based on config.DHAM_num_bins
        #and discretize the data into bins.
        #for higher dimension, we discretize and then ravel it into 1D array.
        if use_dynamic_bins:
            minval = np.min(self.data[:, :])
            maxval = np.max(self.data[:, :])
            self.qspace = np.linspace(minval, maxval, self.num_bins)
            print("min/max of the data: ", minval, maxval)
            print("using dynamic Binning with {0:d} bins".format(self.num_bins))
        else:
            self.qspace = config.qspace
            print("using static Binning with {0:d} bins".format(config.qspace_num_bins))
            
        b = np.digitize(self.data[:, :], self.qspace)
        sumtr, trvec = count_transitions(b, self.num_bins, self.lagtime, use_symmetry=use_symmetry)

        MM = self.build_MM(sumtr, trvec, biased)
        msm = MSM()
        msm.M = MM.T
        msm._compute_peq_fes_M()
        print("sum of peq in dham reconstruction: ", sum(msm.peq))

        if False:
            plt.figure()
            plt.plot(self.qspace, msm.free_energy, label=" fes from reconstructed M by DHAMsym")
            plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.num_bins))
            plt.xlim(config.qspace_low, config.qspace_high)
            plt.savefig(f"./test_dham_{self.prop_index}.png")
            plt.close()        
        return self.qspace, msm.free_energy, MM.T # fes in unit kcal/mol.

    def bootstrap_error(self, size, iter=100, plotall=False, save=None):
        full = self.run(plot=False)
        results = []
        data = np.copy(self.data)
        for _ in range(iter):
            idx = np.random.randint(data.shape[0], size=size)
            self.data = data[idx, :]
            try:
                results.append(self.run(plot=False, adjust=False))
            except ValueError:
                print(idx)
        r = np.array(results).astype(np.float_)
        r = r[~np.isnan(r).any(axis=(1, 2))]
        r = r[~np.isinf(r).any(axis=(1, 2))]
        if plotall:
            f, a = plt.subplots()
            for i in range(r.shape[0]):
                a.plot(r[i, 0], r[i, 1])
       #     plt.show()
        # interpolation
        newU = np.empty(shape=r[:, 1, :].shape, dtype=np.float_)
        for i in range(r.shape[0]):
            newU[i, :] = np.interp(full[0], r[i, 0], r[i, 1])
            # realign
            offset = align(newU[i, :], full[1])
            newU[i, :] += offset
        stderr = np.std(newU, axis=0)
        f, a = plt.subplots()
        a.plot(full[0], full[1])
        a.fill_between(full[0], full[1] - stderr, full[1] + stderr, alpha=0.2)
     #   plt.title("lagtime={0:d} bins={1:d}".format(self.lagtime, self.numbins))
        if save is None:
     #       plt.show()
            print("saved is none")
        else:
     #       plt.savefig(save)
             print("save is none")
        self.data = data
        return
