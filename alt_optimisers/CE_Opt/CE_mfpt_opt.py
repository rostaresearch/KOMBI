#this is CE clas for mfpt optimization
import numpy as np
from scipy import stats
#from cross_entropy_method import CEM  


def CEM(test_function, dimensions, bounds, popsize, num_elite, sigma_init, seed_number, max_evals):
    np.random.seed(seed_number)
    eps = 1e-4
    bound_lower, bound_upper = np.asarray(bounds).T
    sigma = sigma_init * np.eye(dimensions)

    diff = np.fabs(bound_lower - bound_upper)
    n_evals = 0
    num_evals = []
    # mu = np.random.rand(dimensions) - (bound_upper + 1)
    mu = bound_lower + diff * np.random.rand(dimensions)
    generation_count = 0
    all_mu = []
    all_sigma = []
    all_offspring = []
    all_pops = []
    all_sigma = []
    all_elite = []
    all_fitness = []
    
    while True:
    # for i in range(10000):
        if n_evals > max_evals:
            break
        all_mu.append(mu)
        all_sigma.append(sigma)

        x = np.random.multivariate_normal(mu, sigma, popsize)
        all_pops.append(np.copy(x))
        # print(np.sum(x))
        all_offspring.append(x)
        fitness = np.array([test_function(x[i]) for i in range(popsize)])
        n_evals += popsize
        best_fitness = max(fitness) 
        all_fitness.append(best_fitness)
        # print(x)
        if best_fitness < eps or np.sum(x) > 1e150 or np.sum(x) < -1e150:
            break

        elite_idx = fitness.argsort()[:num_elite]
        all_elite.append(elite_idx)
        mu = np.mean(x[elite_idx], axis=0)

        sigma = np.zeros_like(sigma)
        for i in range(num_elite):
            z = x[elite_idx[i]] - mu
            z = z.reshape(-1, 1)
            # print(num_evals)
            # sigma += tf.matmul(z.T, z)
            # sigma += (z.T * z)
            sigma += (z.T @ z)

        all_sigma.append(sigma)
        sigma *= (1/num_elite)
        generation_count += 1
        num_evals.append(n_evals)

    all_mu.append(mu)
    best_results = mu.copy()
    best_fitness = test_function(mu)
    return all_pops, all_sigma, all_mu, all_fitness, num_evals, generation_count


class CEM_MFPT(CEM):
    """
    CEM for mfpt minimization problem
    """
    def __init__(self, *args, **kwargs):
        super(CEM_MFPT, self).__init__(*args, **kwargs)
        self.default_dist_titles = ['param_'+str(i+1) for i in range(30)]
        self.default_samp_titles = 'mfpt'
        self.num_gaussian = 10 # can be changed later.
    
    def do_sample(self):
        """
        the initial sampling of the parameters is set to be a uniform distribution of the gaussian parameter boarders.
        for gaussian amplitude (a): [0.1,1.5] unit in kcal/mol
        for gaussian means (b): [0,2pi] unit in nm
        for gaussian sigmas (c): [0.5,1.5] unit in nm
        
        note we have 3* num_gaussian parameters
        """
        return np.random.uniform(low=[0.1,0,0.5]*self.num_gaussian, high=[1.5,2*np.pi,1.5]*self.num_gaussian)

    def pdf(self, x):
        """
        the pdf of the parameters is set to be a uniform distribution.
        """
        return stats.uniform.pdf(x, loc=[0.1,0,0.5]*self.num_gaussian, scale=[1.5,2*np.pi,1.5]*self.num_gaussian)
    
    def update_sample_distribution(self, samples, weights):
        w = np.array(weights)
        s = np.array(samples)
