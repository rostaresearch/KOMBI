this folder is a Cross Entropy Optimization try on the 1D_langevin_sim script.

Inside the util function there's a warpped function:
try_and_optim_M()
we will replace the scipy.minimize part using the CE method.

The CE method is a population based optimization method, which is suitable for the optimization of non-convex, non-smooth, and high-dimensional problems. It is based on the cross-entropy method, which is a general Monte Carlo method for estimating rare event probabilities. 