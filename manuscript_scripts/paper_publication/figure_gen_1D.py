from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from os import path
plt.rcParams.update({'font.size': 20})

#here we load the precalculated 1D optim bias and zero it on 89 state.
# note all loaded data is gaussian parameters in 1D.
#state_start = 7 #note in this code this is 0-indexed. so 8 means state 9.
state_start = 5
state_end = 82
num_gaussian = 30
N = 100 #number of grid points, i.e. num of states.
kT = 0.5981
#qspace is from 0 to 99, with 100 states.
qspace = np.linspace(0, 99, 100)
cur_dir = path.dirname(path.realpath(__file__))
#filename = path.join(cur_dir, 'pos_bias_m.mat')
#filename = path.join(cur_dir, './pos_bias_m_langevin_system.mat')
filename = path.join(cur_dir, './pos_bias_1stMar_2.mat')
gaussian_params_pos = loadmat(filename)['pos_bias'][0]

#gaussian_params_pos = gaussian_params_pos[-1] #quick fix for edina's fig1_30Gaussian.mat file.


def create_K_1D(N, kT):
    #create the K matrix for 1D model potential
    #K is a N*N matrix, representing the transition rate between states
    #The diagonal elements are the summation of the other elements in the same row, i.e. the overall outflow rate from state i
    #The off-diagonal elements are the transition rate from state i to state j (or from j to i???)
    x = np.linspace(0, 5*np.pi, N) #create a grid of x values
    y1 = np.sin((x-np.pi))
    y2 = np.sin((x-np.pi)/2)
    amplitude = 10
    xtilt = 0.5
    y = xtilt*y1 - y2
    y = (xtilt*y1 + (1-xtilt)*y2)
    y = y*3
    
    K = np.zeros((N,N))
    for i in range(N-1):
        K[i, i + 1] = amplitude * np.exp((y[i+1] - y[i]) / 2 / kT)
        K[i + 1, i] = amplitude * np.exp((y[i] - y[i+1]) / 2 / kT) #where does this formula come from?
    for i in range(N):
        K[i, i] = 0
        K[i, i] = -np.sum(K[:, i])
    return K
def compute_free_energy(K, kT):
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
    peq = evectors[:, index[-1]].T/np.sum(evectors[:, index[-1]]) #normalize the eigenvector
    #take the real part of the eigenvector i.e. the probability distribution at equilibrium.
    #print('sum of the peq is:', np.sum(peq))

    #calculate the free energy
    F = -kT * np.log(peq + 1e-9) #add a small number to avoid log(0))

    return [peq, F, evectors, evalues, evalues_sorted, index]

def gaussian_1d(x, bx, cx):
    return 2*np.exp(-((x-bx)**2/(2*cx**2)))

def get_total_bias_1d(x, c_g, std_g):
    total_bias = np.zeros(x.shape[0])
    for i in range(num_gaussian):
        total_bias += gaussian_1d(x, c_g[i], std_g[i])
    return total_bias


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

def create_K_1D_langevin_system(N, kT):
    amp = 6
    num_hills = 9
    A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
    x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
    sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
    x = np.linspace(0, 2*np.pi, 100)
    y = np.zeros_like(x)
    for i in range(num_hills):
        y += A_i[i] * np.exp(-(x-x0_i[i])**2/(2*sigma_x_i[i]**2))
    y = y - y.min()
    
    #sigmoid on both end of x.
    k = 5
    max_barrier = '1e2'
    offset = 0.4
    y += float(max_barrier) * (1 / (1 + np.exp(k * (x - (-offset)))))
    y += float(max_barrier) * (1 / (1 + np.exp(-k * (x - (2 * np.pi + offset)))))

    kBT = 0.5981
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            u_ij = y[j] - y[i]
            K[i, j] = np.exp(u_ij/ (2*kBT))
            K[j, i] = np.exp(-u_ij/ (2*kBT))
        K[i, i] = -np.sum(K[:,i])
    return K

#test plot the K fes.
K = create_K_1D_langevin_system(N, kT)
F = compute_free_energy(K, kT)[1]
F_min = np.min(F)
F -= F_min

colormap = plt.cm.get_cmap('coolwarm', gaussian_params_pos.shape[0])

#only plot the F here for book keeping.
if True:
    increment = 10
    #plot the FES. up until 82 index. (state 83)
    plt.figure(figsize=(8,6))
    
    #plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
    plt.plot(F, color = 'black', linewidth = 2)

    #position legend on the top right with alpha=0.7
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color = 'black', alpha = 1))
    labels.append(f'unbiased FES')
    #get a marker for the current position.
    handles.append(plt.Line2D([0], [0], marker = 'o', color = colormap(0), alpha = 0.75, markersize = 10))
    labels.append(f'current position')
    #get a marker for target
    handles.append(plt.Line2D([0], [0], marker = '*', color = 'green', alpha = 1, markersize = 10))
    labels.append(f'target position')

    #then we create the legend.
    plt.legend(handles, labels, loc = 'upper right', fontsize = 15)
    plt.xlabel('state')
    plt.ylabel('FES (kcal/mol)')
    plt.tight_layout()
    #plt.show()
    #plt.xlim(state_start-2, state_end+2)
    plt.ylim(-1, 14)
    plt.plot(82, F[82], marker = '*', color = 'green', markersize = 18) #end
    plt.plot(32, F[32], marker = 'o', color = 'red', markersize = 18) #start
    plt.savefig(f'./1D_F.png')
    
    plt.close()



#here we plot the FES and the biased FES.
if True:
    increment = 10
    #plot the FES. up until 82 index. (state 83)
    plt.figure(figsize=(7,6))#(figsize=(8,6))
    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.2)
    #plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
    plt.plot(F, color = 'black', linewidth = 2)
    F_biased_total=[]
    pos_i_list=[]
    for pos_i in range(state_start, state_end, increment):
        pos_i_list.append(pos_i)
        #unpack all the gaussian params. 20 center_gaussian and 20 std gaussian.
        allparam = gaussian_params_pos[pos_i][0]

        c_g = allparam[:num_gaussian]
        std_g = allparam[num_gaussian:]

        #get the total gaussian bias.
        total_bias = get_total_bias_1d(qspace, c_g, std_g)

        #now we apply this bias on K and calculate FES.
        #K_biased = bias_K_1D(K, total_bias, kT)
        #F_biased = compute_free_energy(K_biased, kT)[1]

        F_biased = F + total_bias - total_bias[pos_i]

        #truncate the FES until 82
        F_biased = F_biased[:83]
        
        #give data point before pos_i as nan
        #F_biased[:pos_i] = np.nan
        #F_biased -= F_biased[82]
    
        #zero the F_biased on state 82.
        #F_biased -= F_biased[5]#np.min(F_biased)
        #print(F[82])
        #F_biased += F[5]
        F_biased_total.append(F_biased)

    #F_biased_total = F_biased_total[:-2]
    #pos_i_list = pos_i_list[:-2]


    for i in range(len(F_biased_total)):
        plt.plot(F_biased_total[i], color = colormap(i*increment), alpha = 1)

    
    j=0
    for i in pos_i_list:
        plt.plot(i, F[i], marker = 'o', color = colormap(i), markersize = 10)
        plt.plot(i, F_biased_total[j][i], marker = 'o', color = colormap(i), markersize = 10, alpha = 0.75)
        j += 1


    #position legend on the top right with alpha=0.7
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color = 'black', alpha = 1))
    labels.append(f'unbiased FES')
    handles.append(plt.Line2D([0], [0], color = colormap(0), alpha = 0.7))
    labels.append(f'biased FES')
    #get a marker for the current position.
    handles.append(plt.Line2D([0], [0], marker = 'o', color = colormap(0), alpha = 0.75, markersize = 10))
    labels.append(f'current position')
    #get a marker for target
    handles.append(plt.Line2D([0], [0], marker = '*', color = 'green', alpha = 1, markersize = 10))
    labels.append(f'target position')

    #then we create the legend.
    plt.legend(handles, labels, loc = 'lower left', fontsize = 15) #(handles, labels, loc = 'upper right', fontsize = 15)
    plt.xlabel('state')
    plt.ylabel('FES (kcal/mol)')
    plt.tight_layout()
    #plt.show()
    plt.xlim(state_start-2, state_end+2)
    #plt.ylim(-1, 20)
    plt.plot(82, F[82], marker = '*', color = 'green', markersize = 18) #end
    #plt.plot(32, F[32], marker = 'o', color = 'red', markersize = 18) #start
    plt.savefig(f'./1D_opt_biasedF_mat_overlap.png')
    
    plt.close()

    #import imageio
    if False:
        import os

        path = './1D_gif/'

        # Get all file names sorted by their creation time
        file_names = sorted(os.listdir(path), key=lambda x: int(x.split('-')[1].split('_')[0]) if '_overlap' in x and x.endswith('.png') else 999999)

        # Create a writer object
        writer = imageio.get_writer('./1D_gif/animation_2.gif', fps=3)

        # Add images to the writer object
        for file_name in file_names:
            if 'overlap' in file_name and file_name.endswith('.png'):
                writer.append_data(imageio.imread(os.path.join(path, file_name)))

        # Close the writer object
        writer.close()

        print('done')

##################################################
if False:
    #here we take the state 8 as example, show the decomposition of the gaussian bias.
    plt.figure(figsize=(8,6))
    plt.plot(88, F[88], marker = '*', color = 'green', markersize = 14)
    #plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
    plt.plot(F, color = 'black', linewidth = 2)

    pos_i = 8
    allparam = gaussian_params_pos[pos_i][0]
    c_g = allparam[:num_gaussian]
    std_g = allparam[num_gaussian:]
    for i in range(num_gaussian):
        individual_bias = gaussian_1d(qspace, c_g[i], std_g[i])
        #we cut until 88.
        individual_bias = individual_bias[:89]
        plt.plot(individual_bias, color = 'grey', alpha = 1, linewidth = 1.5, linestyle = '--')
    total_bias = get_total_bias_1d(qspace, c_g, std_g)
    total_bias = total_bias[:89]
    plt.plot(total_bias, color = colormap(pos_i), linewidth = 2, linestyle = '--')
    plt.plot(pos_i, F[pos_i], marker = 'o', color = colormap(pos_i), markersize = 10)
    F_biased = F + total_bias
    plt.plot(F_biased, color = colormap(pos_i), linewidth = 2)
    plt.plot(pos_i, F_biased[pos_i], marker = 'o', color = colormap(pos_i), alpha = 0.75, markersize = 10)

    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color = 'black', alpha = 1))
    labels.append(f'unbiased FES')
    handles.append(plt.Line2D([0], [0], color = colormap(pos_i), alpha = 1, linewidth = 2))
    labels.append(f'biased FES')
    handles.append(plt.Line2D([0], [0], color = 'blue', alpha = 1, linewidth = 1.5, linestyle = '--'))
    labels.append(f'total bias')
    handles.append(plt.Line2D([0], [0], color = 'grey', alpha = 1, linestyle = '--'))
    labels.append(f'individual bias')
    handles.append(plt.Line2D([0], [0], marker = 'o', color = colormap(pos_i), alpha = 1, markersize = 10))
    labels.append(f'current position')
    handles.append(plt.Line2D([0], [0], marker = '*', color = 'green', alpha = 1, markersize = 10))
    labels.append(f'target position')
    plt.legend(handles, labels, loc = 'upper right', fontsize = 15)

    plt.xlabel('state')
    plt.ylabel('FES (kcal/mol)')
    plt.tight_layout()
    plt.savefig(f'./1D_gif/optim-{pos_i}_bias_decomposition.png')
    plt.close()
    #plt.show()

##################################################

if True:
    increment = 10
    #we plot every 3 optimal bias, with dots
    plt.figure(figsize=(7,6)) # (figsize=(8,6))
    #plt.tight_layout(pad=2.0)
    #plt.subplots_adjust(bottom=0.2)
    #plt.plot(pos_i, F[pos_i], marker = 'o', color = 'red', markersize = 10)
    plt.plot(F, color = 'black', linewidth = 2)
    F_biased_total=[]
    pos_i_list=[]
    total_bias_list = []
    for pos_i in range(state_start, state_end, increment):
        pos_i_list.append(pos_i)
        #unpack all the gaussian params. 20 center_gaussian and 20 std gaussian.
        allparam = gaussian_params_pos[pos_i][0]
 
        c_g = allparam[:num_gaussian]
        std_g = allparam[num_gaussian:]

        #get the total gaussian bias.
        total_bias = get_total_bias_1d(qspace, c_g, std_g)

        #truncate the total bias.
        total_bias = total_bias[:83]
        #total_bias[:pos_i] = np.nan
        total_bias = total_bias - total_bias[pos_i] + F[pos_i]
        #total_bias -= total_bias[82]
        total_bias_list.append(total_bias)
    #ditch the last 2, not looking good.
    #total_bias_list = total_bias_list[:-2]
    #pos_i_list = pos_i_list[:-2]

    print(len(total_bias_list))
    #plot current total bias
    for i in range(len(total_bias_list)):
        plt.plot(total_bias_list[i], color = colormap(i*increment), alpha = 1, linewidth = 2, linestyle = '--')

    #plot the dot.
    j = 0
    """for i in range(1,pos_i,10):
        #plot agin the current position on F.
        plt.plot(i, F[i], marker = 'o', color = colormap(i), markersize = 10)
        #plot the current position on lastest F_biased.
        #plt.plot(i, F_biased_total[-1][pos_i], marker = 'o', color = 'red', markersize = 10, alpha = 0.5)
        plt.plot(i, total_bias_list[j][i], marker = 'o', color = colormap(i), markersize = 10, alpha = 0.75)
        j += 6"""
    for i in pos_i_list:
        plt.plot(i, F[i], marker = 'o', color = colormap(i), markersize = 10)
        plt.plot(i, total_bias_list[j][i], marker = 'o', color = colormap(i), markersize = 10, alpha = 0.75)
        j += 1

    #position legend on the top right with alpha=0.7
    handles, labels = [], []
    handles.append(plt.Line2D([0], [0], color = 'black', alpha = 1))
    labels.append(f'unbiased FES')
    handles.append(plt.Line2D([0], [0], color = colormap(0), alpha = 1,linewidth = 2, linestyle = '--'))
    labels.append(f'total bias')
    #get a marker for the current position.
    handles.append(plt.Line2D([0], [0], marker = 'o', color = colormap(0), alpha = 0.75, markersize = 10))
    labels.append(f'current position')
    #get a marker for target
    handles.append(plt.Line2D([0], [0], marker = '*', color = 'green', alpha = 1, markersize = 10))
    labels.append(f'target position')

    #then we create the legend.
    plt.legend(handles, labels, loc = 'lower left', fontsize = 15) #(handles, labels, loc = 'upper right', fontsize = 15)
    plt.xlabel('state')
    plt.ylabel('FES (kcal/mol)')
    plt.tight_layout()
    #plt.show()
    plt.xlim(state_start-2, state_end+2)
    #plt.ylim(-1, 14)
    plt.plot(82, F[82], marker = '*', color = 'green', markersize = 18) #end
    #plt.plot(32, F[32], marker = 'o', color = 'red', markersize = 18) #start
    plt.savefig(f'./1D_opt_bias_mat_overlap.png')
    plt.close()





print('done')