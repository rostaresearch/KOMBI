#original code from github: https://github.com/rostaresearch/enhanced-sampling-workshop-2022/blob/main/Day1/src/dham.py
#modified by TW on 28th July 2023
#note that in this code we presume the bias is 10 gaussian functions added together.
#returns the Markov Matrix, free energy surface probed by DHAM. 


#note this is now in 2D.

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.linalg import eig
from scipy.optimize import minimize

from numpy import pi

def gaussian_2d(x, y, ax, bx, by, cx, cy):
    return np.exp(-((x-bx)**2/(2*cx**2) + (y-by)**2/(2*cy**2)))

def rmsd(offset, a, b):
    return np.sqrt(np.mean(np.square((a + offset) - b)))


def align(query, ref):
    offset = -10.0
    res = minimize(rmsd, offset, args=(query, ref))
    print(res.x[0], res.fun)
    return res.x[0]


def count_transitions(b, numbins, lagtime, endpt=None):
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
    sumtr = 0.5 * (sumtr + np.transpose(sumtr)) #disable for original DHAM, enable for DHAM_sym
    # anti = 0.5 * (sumtr - np.transpose(sumtr))
    # print("Degree of symmetry:",
    #       (np.linalg.norm(sym) - np.linalg.norm(anti)) / (np.linalg.norm(sym) + np.linalg.norm(anti)))
    #plt.contourf(sumtr.real)
    #plt.colorbar()
    return sumtr.real, trvec


class DHAM:
    KbT = 0.001987204259 * 300  # energy unit: kcal/mol
    epsilon = 0.00001
    data = None
    vel = None
    datlength = None
    k_val = None
    constr_val = None
    qspace = None
    numbins = 90
    lagtime = 1

    def __init__(self, gaussian_params):
        #unpack it to self.a, self.bx, self.by, self.cx, self.cy
        num_gaussian = len(gaussian_params)//5
        self.a = gaussian_params[:num_gaussian]
        self.bx = gaussian_params[num_gaussian:2*num_gaussian]
        self.by = gaussian_params[2*num_gaussian:3*num_gaussian]
        self.cx = gaussian_params[3*num_gaussian:4*num_gaussian]
        self.cy = gaussian_params[4*num_gaussian:5*num_gaussian]
        x,y = np.meshgrid(np.linspace(-pi,pi, self.numbins), np.linspace(-pi,pi, self.numbins))
        self.x = x
        self.y = y
        return

    def setup(self, CV, T):
        self.data = CV
        self.KbT = 0.001987204259 * T
        return

    def build_MM(self, sumtr, trvec, biased=False):
        #MM = np.empty(shape=sumtr.shape, dtype=np.longdouble)
        #X,Y = np.meshgrid(np.linspace(-pi,pi, self.numbins), np.linspace(-pi,pi, self.numbins), indexing='ij')
        if biased:
            MM = np.zeros(shape=sumtr.shape, dtype=np.longdouble)
            #compute the total bias u.
            u = np.zeros_like(self.x)
            for n in range(len(self.a)):
                u += gaussian_2d(self.x, self.y, self.a[n], self.bx[n], self.by[n], self.cx[n], self.cy[n])
            
            for i in range(sumtr.shape[0]):
                for j in range(sumtr.shape[1]):
                    if sumtr[i, j] > 0:
                        sump1 = 0.0
                        i_x, i_y = np.unravel_index(i, (self.numbins, self.numbins), order='C')
                        j_x, j_y = np.unravel_index(j, (self.numbins, self.numbins), order='C')

                        for k in range(trvec.shape[0]):
                            if trvec[k, i] > 0:
                                sump1 += trvec[k, i] * np.exp(-(u[j_x, j_y] - u[i_x, i_y]) / (2*self.KbT))
                        if sump1 > 0:
                            MM[i, j] = sumtr[i, j] / sump1
                        else:
                            MM[i, j] = 0
            #MM = MM.T
            #epsilon_offset = 1e-30
            #MM = MM / (np.sum(MM, axis=1)[:, None]+epsilon_offset)
            for i in range(MM.shape[0]):
                row_sum = np.sum(MM[i, :])
                if row_sum != 0:
                    MM[i, :] = MM[i, :] / row_sum
        else:
            raise NotImplementedError
        
        #plt.contourf(MM.real)
        return MM.real

    def run(self, plot=True, adjust=True, biased=False, conversion=2E-13):
        """

        :param plot:
        :param adjust:
        :param biased:
        :param conversion: from timestep to seconds
        :return:
        """
        #digitialize the data into 2D mesh.
        b = np.digitize(self.data, np.linspace(0, self.numbins*self.numbins+1, self.numbins*self.numbins+1))
        b = b.reshape(1,-1)
        #here we check the b trajectory.
        #unravel it.
        #b_test = b.reshape(-1)
        #b_test = np.unravel_index(b, (self.numbins, self.numbins), order='C')
        #plt.plot(b_test[0], b_test[1], 'o')
        #plt.show()

        sumtr, trvec = count_transitions(b, self.numbins * self.numbins, self.lagtime)

        MM = self.build_MM(sumtr, trvec, biased)

        from util_2d import compute_free_energy
        mU2 = compute_free_energy(MM.T, self.KbT)[1]

        #dG = np.max(mU2[:int(self.numbins)])
        #A = rate / np.exp(- dG / self.KbT)
        #note this is in kcal/mol. let's convert it to kj/mol
        conversion_factor = 4.184
        x,y = np.meshgrid(np.linspace(-pi,pi, self.numbins), np.linspace(-pi,pi, self.numbins))
        plt.figure()
        plt.contourf(x,y,(mU2.reshape(self.numbins, self.numbins).T)*conversion_factor) 
        plt.colorbar()
        plt.savefig("./figs/DHAM.png")
        plt.show()
        """
        from util_2d import compute_free_energy, Markov_mfpt_calc, kemeny_constant_check
        peq_M, F, evectors, evalues, evalues_sorted, index = compute_free_energy(MM, self.KbT)
        #mfpts = Markov_mfpt_calc(peq_M, MM)
        #kemeny_constant_check(mfpts, peq_M)
        if plot:
            F_2D = F.reshape(self.N, self.N).real
            F_2D_masked = np.ma.masked_invalid(F_2D)
            plt.contourf(self.x, self.y, F_2D_masked.T) #, label="reconstructed M by DHAMsym")
            plt.colorbar()
            plt.show()
            #plt.title("Lagtime={0:d} Nbins={1:d}".format(self.lagtime, self.numbins))
            #plt.show()
        """
        return (mU2)*conversion_factor, (MM.T)*conversion_factor

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
            plt.show()
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
        plt.title("lagtime={0:d} bins={1:d}".format(self.lagtime, self.numbins))
        if save is None:
            plt.show()
        else:
            plt.savefig(save)
        self.data = data
        return