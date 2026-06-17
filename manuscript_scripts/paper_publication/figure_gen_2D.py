from math import pi
from matplotlib import pyplot as plt
import numpy as np
from util_2d import *
plt.rcParams.update({'font.size': 16})

#first we initialize some parameters.

N = 20 #number of grid points, i.e. num of states.
kT = 0.5981
t_max = 10**7 #max time
ts = 0.1 #time step

x,y = np.meshgrid(np.linspace(-3,3,N),np.linspace(-3,3,N))
state_start = (14, 14)
state_end = (4, 7)

#map it into xy mesh.

img = Image.open("./fes_digitize.png")
img = np.array(img)

img_greyscale = 0.8 * img[:,:,0] - 0.15 * img[:,:,1] - 0.2 * img[:,:,2]
img = img_greyscale
img = img/np.max(img)
img = img - np.min(img)

#get img square and multiply the amp = 7
min_dim = min(img.shape)
img = img[:min_dim, :min_dim]
img = 7 * img

plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.savefig("./figs/unbiased.png", dpi=600)
plt.show()


traj_6 = np.load("./data/20230927-135127_6_CV_total.npy")#np.load("./data/CV_total_20230921-114158_2.npy") #note this is ravelled.
traj_0 = np.load("./data/20230927-135127_0_CV_total.npy")
traj_1 = np.load("./data/20230927-135127_1_CV_total.npy")
traj_2 = np.load("./data/20230927-135127_2_CV_total.npy")
traj_3 = np.load("./data/20230927-135127_3_CV_total.npy")
traj_4 = np.load("./data/20230927-135127_4_CV_total.npy")
traj_5 = np.load("./data/20230927-135127_5_CV_total.npy")

#now we unravel the traj.
traj_0 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_0])
traj_1 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_1])
traj_2 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_2])
traj_3 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_3])
traj_4 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_4])
traj_5 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_5])
traj_6 = np.array([np.unravel_index(int(i), (N,N)) for i in traj_6])

#now traj_all is a list of array
# we unravel it element-wise
#for i, traj in enumerate(traj_all):
#    traj_all[i] = np.array([np.unravel_index(int(i), (N,N)) for i in traj])

#load gaussian parameters
gp0 = None
gp1 = np.load("./data/20230927-135127_1_gaussian_params.npy")
gp2 = np.load("./data/20230927-135127_2_gaussian_params.npy")
gp3 = np.load("./data/20230927-135127_3_gaussian_params.npy")
gp4 = np.load("./data/20230927-135127_4_gaussian_params.npy")
gp5 = np.load("./data/20230927-135127_5_gaussian_params.npy")
gp6 = np.load("./data/20230927-135127_6_gaussian_params.npy")
#gp7 = np.load("./data/20230927-135127_7_gaussian_params.npy")
#gp8 = np.load("./data/20230927-135127_8_gaussian_params.npy")
#gp9 = np.load("./data/20230927-135127_9_gaussian_params.npy")

#we define a funtion to plot.
#this is for CV_total data, where CV_total is a list of traj. length: [1001, 2002, etc.]
"""def plot_traj(traj_all, index, img, gp, state_start, state_end, title, save_name, first_plot = False, show_plot = False):
    plt.figure()
    for i, traj in enumerate(traj_list):
        traj_x_indices = traj[:, 0]
        traj_y_indices = traj[:, 1]
        traj_x_coords = x[traj_x_indices, traj_y_indices]
        traj_y_coords = y[traj_x_indices, traj_y_indices]
        if i != len(traj_list) - 1:
            plt.plot(traj_x_coords, -traj_y_coords, color="yellow", linewidth=1.8, alpha = 0.4,)
        else:
            plt.plot(traj_x_coords, -traj_y_coords, color="yellow", linewidth=1.8, alpha = 0.8,)
            if not first_plot:
                plt.plot(traj_x_coords[-1], -traj_y_coords[-1], marker = 'o', color = "red", markersize = 10)
    #plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12) #this is the unbiased FES.
    #plot the biased FES.
    x_img, y_img = np.meshgrid(np.linspace(-3,3,min_dim), np.linspace(-3,3,min_dim))
    total_bias = get_total_bias_2d(x_img, y_img, gp)
    img_biased = img + total_bias
    plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
    if first_plot:
        plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
    plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
    
    plt.title(title)
    if show_plot:
        plt.show()
    plt.savefig(save_name, dpi=600)
    plt.close()
"""
#this is for individual CV data.
def plot_traj(traj_list, img, gp, state_start, state_end, title, save_name, first_plot = False, show_plot = False):
    plt.figure()
    for i, traj in enumerate(traj_list):
        traj_x_indices = traj[:, 0]
        traj_y_indices = traj[:, 1]
        traj_x_coords = x[traj_x_indices, traj_y_indices]
        traj_y_coords = y[traj_x_indices, traj_y_indices]
        if i != len(traj_list) - 1:
            plt.plot(traj_x_coords, -traj_y_coords, color="yellow", linewidth=1.8, alpha = 0.4,)
        else:
            plt.plot(traj_x_coords, -traj_y_coords, color="yellow", linewidth=1.8, alpha = 0.8,)
            if not first_plot:
                plt.plot(traj_x_coords[-1], -traj_y_coords[-1], marker = 'o', color = "red", markersize = 10)
    #plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12) #this is the unbiased FES.
    #plot the biased FES.
    x_img, y_img = np.meshgrid(np.linspace(-3,3,min_dim), np.linspace(-3,3,min_dim))
    if gp is not None:
        total_bias = get_total_bias_2d(x_img, y_img, gp)
        img_biased = img + total_bias
        #apply the maximum filter to the biased img. threshold = 1.2 * amp, amp = 7
        img_biased = np.clip(img_biased, 0, 7*1.2)
    else:
        img_biased = img
    
    plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
    if first_plot:
        plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
    plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
    
    plt.title(title)
    if show_plot:
        plt.show()
    plt.savefig(save_name, dpi=600)
    plt.close()

traj_list_1 = [traj_0, traj_1]
traj_list_2 = [traj_0, traj_1, traj_2]
traj_list_3 = [traj_0, traj_1, traj_2, traj_3]
traj_list_4 = [traj_0, traj_1, traj_2, traj_3, traj_4]
traj_list_5 = [traj_0, traj_1, traj_2, traj_3, traj_4, traj_5]
traj_list_6 = [traj_0, traj_1, traj_2, traj_3, traj_4, traj_5, traj_6]

#plot the traj.
plot_traj([traj_0], img, gp0, state_start, state_end, title = "0th traj", save_name = "./figs/prod_figs/traj0.png", first_plot = True)
plot_traj(traj_list_1, img, gp1, state_start, state_end, "1st traj", "./figs/prod_figs/traj1.png")
plot_traj(traj_list_2, img, gp2, state_start, state_end, "2nd traj", "./figs/prod_figs/traj2.png")
plot_traj(traj_list_3, img, gp3, state_start, state_end, "3rd traj", "./figs/prod_figs/traj3.png")
plot_traj(traj_list_4, img, gp4, state_start, state_end, "4th traj", "./figs/prod_figs/traj4.png")
plot_traj(traj_list_5, img, gp5, state_start, state_end, "5th traj", "./figs/prod_figs/traj5.png")
plot_traj(traj_list_6, img, gp6, state_start, state_end, "6th traj", "./figs/prod_figs/traj6.png")
#plot_traj(traj_list_7, img, gp7, state_start, state_end, "7th traj", "./figs/prod_figs/traj7.png")
#plot_traj(traj_list_8, img, gp8, state_start, state_end, "8th traj", "./figs/prod_figs/traj8.png")
#plot_traj(traj_list_9, img, gp9, state_start, state_end, "9th traj", "./figs/prod_figs/traj9.png")
"""

#first we plot the unbiased FES and its traj.

traj_1_x_indices = traj_1[:, 0]
traj_1_y_indices = traj_1[:, 1]
traj_1_x_coords = x[traj_1_x_indices, traj_1_y_indices]
traj_1_y_coords = y[traj_1_x_indices, traj_1_y_indices]

plt.figure()
plt.imshow(img, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
#plt.title("")
plt.savefig("./figs/prod_figs/traj1.png")
#plt.show()
plt.close()

#next we plot the biased FES after traj1.
from util_2d import *
x_img, y_img = np.meshgrid(np.linspace(-3,3,min_dim), np.linspace(-3,3,min_dim))
total_bias = get_total_bias_2d(x_img, y_img, gp1)

#add the bias to the unbiased FES.
img_biased = img + total_bias

#process traj_2

traj_2_x_indices = traj_2[:, 0]
traj_2_y_indices = traj_2[:, 1]
traj_2_x_coords = x[traj_2_x_indices, traj_2_y_indices]
traj_2_y_coords = y[traj_2_x_indices, traj_2_y_indices]

plt.figure()
plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_2_x_coords, -traj_2_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
#plot the last position of traj_2
plt.plot(traj_2_x_coords[-1], -traj_2_y_coords[-1], marker = 'o', color = "red", markersize = 10)
#plt.show()
plt.savefig("./figs/prod_figs/traj2.png")
plt.close()


#next we plot the biased FES after traj2.
total_bias = get_total_bias_2d(x_img, y_img, gp2)
img_biased = img + total_bias
#process traj_3
traj_3_x_indices = traj_3[:, 0]
traj_3_y_indices = traj_3[:, 1]
traj_3_x_coords = x[traj_3_x_indices, traj_3_y_indices]
traj_3_y_coords = y[traj_3_x_indices, traj_3_y_indices]

plt.imshow(img_biased, cmap="coolwarm", extent=[-3,3,-3,3], vmin=0, vmax=12)
plt.plot(x[state_start], -y[state_start], marker = 'o', color = "red", markersize = 10) #this is starting point.
plt.plot(x[state_end], -y[state_end],marker = 'x', color = "red", markersize = 10) #this is ending point.
plt.plot(traj_1_x_coords, -traj_1_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_2_x_coords, -traj_2_y_coords, color="yellow", linewidth=0.8, alpha = 0.4,)
plt.plot(traj_3_x_coords, -traj_3_y_coords, color="yellow", linewidth=0.8, alpha = 0.8,)
plt.plot(traj_3_x_coords[-1], -traj_3_y_coords[-1], marker = 'o', color = "red", markersize = 10)
#plt.show()
plt.savefig("./figs/prod_figs/traj3.png")
plt.close()

"""
print("done")