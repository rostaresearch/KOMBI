#this code analysis the traj.
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
#first we get all traj

#font size
plt.rcParams.update({'font.size': 18})

plot_boxchart = True
plot_2d_traj = False
plot_1d_fes = False
plot_2d_fes = False

def nm_to_angstrom(x):
    return x * 10  # 1 nm = 10 angstroms
if __name__ == "__main__":
    if plot_boxchart:
        def get_data_from_csv(filename, set_label, method_label):
            df = pd.read_csv(filename, delimiter=',', header=None)
            df = df * 0.002
            cols = pd.MultiIndex.from_product([[set_label], [method_label], range(df.shape[1])])
            df.columns = cols
            return df
        
        set_names = ['1D', '2D', 'NaCl']
        folder_path = [os.path.join('./data/time2reach', set_name) for set_name in set_names]

        #here we load the data from all the folders into a dict
        df = None
        for folder in folder_path:
            set_label = folder.split('/')[-1]
            for file in os.listdir(folder):
                if file.endswith('.csv'):
                    if 'metaD' in file:
                        method_label = 'WT-MetaD'
                    elif 'mfpt' in file:
                        method_label = 'KOMBI' #MFPT-opt
                    else:
                        method_label = 'Classical MD'
                    #we concatenate the dataframes into one
                    if df is None:
                        df = get_data_from_csv(os.path.join(folder, file), set_label, method_label)
                    else:
                        df = pd.concat([df, get_data_from_csv(os.path.join(folder, file), set_label, method_label)], axis=1)
        
        print(df.head())

        #here we plot the group box chart, grouped by the set_names    
        fig, ax = plt.subplots(layout='constrained')
        ticks = ["1D", "2D", "NaCl"]
        
        df_long = df.stack(level=[0, 1]).reset_index()
        df_long.columns = ['Sample', 'Set', 'Method', 'Value']


        #to make MSM-opt label data appear the last
        unique_methods = df_long['Method'].unique().tolist()
        # Move 'MSM-opt' to the end
        unique_methods.append(unique_methods.pop(unique_methods.index('KOMBI')))

        #we print the avg, std of each method, and set.
        for set_name in set_names:
            for method in unique_methods:
                print("Set: ", set_name, "Method: ", method)
                print(df_long[(df_long['Set'] == set_name) & (df_long['Method'] == method)]['Value'].mean())
                print(df_long[(df_long['Set'] == set_name) & (df_long['Method'] == method)]['Value'].std())

        #change on 17th Apr. this line removes 1D/2D for classical MD.
        #df_long = df_long[~((df_long['Set'].isin(['1D', '2D'])) & (df_long['Method'] == 'Classical MD'))]

        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        #ax = sns.boxplot(x="Method", y="Value", hue="Set", data=df_long, palette="coolwarm", order=unique_methods)
        ax = sns.barplot(x="Method", y="Value", hue="Set", data=df_long, palette="coolwarm", order=unique_methods)
        ax.set_xlabel('Simulation method')
        ax.set_ylabel('Time to reach (ps)')
        ax.set_yscale('log')
        ax.set_title('Time to reach the target')
        for i, patch in enumerate(ax.patches):  # iterate over all bars
            method, set_name = df_long.iloc[i][['Method', 'Set']]
            if method == "Classical MD" and set_name in ["1D", "2D"]:
                if set_name == "1D":
                    classical_md_1d_x = patch.get_x() + patch.get_width() / 2  # Center of the bar
                elif set_name == "2D":
                    classical_md_2d_x = patch.get_x() + patch.get_width() / 2  # Center of the bar
                # Add a dashed overlay with the same dimensions and location as the existing bar
                ax.add_patch(
                    patches.Rectangle(
                        (patch.get_x(), patch.get_y()),  # position
                        patch.get_width(),              # width
                        patch.get_height(),             # height
                        fill=False,
                        linestyle='--',
                        linewidth=2,
                        edgecolor='black'
                    )
                )
        #we add few datapoints in 1D, 2D, in the Classical MD as little cross mark
        data_1d = np.array([8.30e+6]) #in ps.
        data_2d = np.array([1.67e+6, 6.23e+6, 3.42e+6]) # in ps.

        for value in data_1d:
            ax.plot(classical_md_1d_x, value, marker='x', markersize=10, color='black', label='1D Data Point')
        for value in data_2d:
            ax.plot(classical_md_2d_x, value, marker='x', markersize=10, color='black', label='2D Data Point')

        # Adding legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color='b', lw=1, linestyle='-'),
                        Line2D([0], [0], color='b', lw=1, linestyle='-')]
        
        # Saving the figure
        #plt.savefig("./figs/box_plot_log.png")
        plt.savefig("./figs/bar_plot_log.png")
        plt.close()

    if plot_2d_traj:
        #first we load the traj.
        import mdtraj as md
        
        mfpt_traj_paths = [file for file in os.listdir('./data/traj/2D/MSMopt') if file.endswith('.dcd')]
        metad_traj_path = './data/traj/2D/metad/20231129-1922082_metaD_traj.dcd'
        top_path = './data/traj/2D/system.pdb'
        print("loading mfpt trajs: ", mfpt_traj_paths)
        print("loading metad traj: ", metad_traj_path)

        mfpt_traj = None
        for mfpt_traj_path in mfpt_traj_paths:
            if mfpt_traj == None:
                mfpt_traj = md.load(os.path.join('./data/traj/2D/MSMopt', mfpt_traj_path), top=top_path)
            else:
                mfpt_traj = md.join([mfpt_traj, md.load(os.path.join('./data/traj/2D/MSMopt', mfpt_traj_path), top=top_path)])
        #metad_traj = md.load(metad_traj_path, top=top_path)
        metad_traj = np.load('./data/traj/2D/metad/20231129-1922082_metaD_pos_traj_.npy')
        
        #we get coor of index 0 atom.
        mfpt_traj = mfpt_traj.xyz[::1, 0, :2] # [frame, atom, xyz] we take xy
        #metad_traj = metad_traj.xyz[::10, 0, :2]
        metad_traj = metad_traj[metad_traj[:,0] != 0]
        metad_traj = metad_traj[::1, :2]

        #then we construct the fes for plotting.
        amp = 6
        num_wells = 9
        num_barrier = 1
        k = 5  # Steepness of the sigmoid curve
        max_barrier = "1e2"  # Scaling factor for the potential maximum
        offset = 0.7 #the offset of the boundary energy barrier.
        A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
        x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
        y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
        sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
        sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

        A_j = np.array([0.3]) * amp
        x0_j = [np.pi]
        y0_j = [np.pi]
        sigma_x_j = [3]
        sigma_y_j = [0.3]

        x = np.linspace(0, 2*np.pi, 100)
        y = np.linspace(0, 2*np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        Z += amp * 4.184 #flat surface
        for i in range(num_wells):
            Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
        for i in range(num_barrier):
            Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))
        total_energy_barrier = np.zeros_like(X)
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * np.pi + offset))))) #right
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * np.pi + offset)))))
        Z += total_energy_barrier
        fes_min = Z.min()
        Z = Z - fes_min

        render_2D = True
        render_3D = False

        if render_3D:
            #we add Z values to the traj.
            mfpt_traj = np.concatenate((mfpt_traj, np.zeros((mfpt_traj.shape[0], 1))), axis=1)
            metad_traj = np.concatenate((metad_traj, np.zeros((metad_traj.shape[0], 1))), axis=1)
            
            def calculate_Z(x,y):
                Z_val = np.zeros_like(x)
                Z_val += amp * 4.184 #flat surface
                for i in range(num_wells):
                    Z_val -= A_i[i] * np.exp(-(x-x0_i[i])**2/(2*sigma_x_i[i]**2) - (y-y0_i[i])**2/(2*sigma_y_i[i]**2))
                for i in range(num_barrier):
                    Z_val += A_j[i] * np.exp(-(x-x0_j[i])**2/(2*sigma_x_j[i]**2) - (y-y0_j[i])**2/(2*sigma_y_j[i]**2))
                total_energy_barrier = np.zeros_like(x)
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (x - (-offset)))))
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (x - (2 * np.pi + offset)))))
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (y - (-offset)))))
                total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (y - (2 * np.pi + offset)))))
                Z_val += total_energy_barrier
                Z_val = Z_val - fes_min + 0.05
                return Z_val

            for point in mfpt_traj:
                point[2] = calculate_Z(point[0], point[1])
            for point in metad_traj:
                point[2] = calculate_Z(point[0], point[1])


        #plotting.
        if render_2D:
            
            plt.figure()
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(bottom=0.2)
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi,0, 2*np.pi], origin="lower")
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

            #we get the 2D kernel density plot using seaborn of the mfpt traj and metad traj.
            #sns.kdeplot(x=mfpt_traj[:,0], y=mfpt_traj[:,1], color="yellow", shade=True, shade_lowest=False, alpha=0.7)
            #sns.kdeplot(x=metad_traj[:,0], y=metad_traj[:,1], color="grey", shade=True, shade_lowest=False, alpha=0.7, levels = [0.00001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5])
            #sns.lineplot(x=mfpt_traj[:,0], y=mfpt_traj[:,1], color="yellow", alpha=0.7, linewidth=2, sort=False)
            #sns.lineplot(x=metad_traj[:,0], y=metad_traj[:,1], color="grey", alpha=0.7, linewidth=1.25, sort=False)
            #sns.scatterplot(x=mfpt_traj[::50,0], y=mfpt_traj[::50,1], color="yellow", alpha=0.7, s=2.5)
            sns.scatterplot(x=metad_traj[:,0], y=metad_traj[:,1], color="grey", alpha=0.7, s=2.5)

            #we plot the start (5.0, 4.0) and end (1.0, 1.5) points.
            plt.plot(5.0, 4.0, marker='o', markersize=12, color="red")
            plt.plot(1.0, 1.5, marker='*', markersize=12, color="green")

            #plot setting.
            plt.xlabel("x (Å)")
            plt.xlim([0, 2*np.pi])
            plt.ylim([0, 2*np.pi])
            plt.ylabel("y (Å)")
            #plt.title("FES mode = multiwell, pbc=False")
            cbar=plt.colorbar()
            cbar.set_label("U (kcal/mol)")
            #plt.savefig('./figs/2Dtraj_visual_scatter_mfpt.png', dpi=800)
            plt.savefig('./figs/2Dtraj_visual_scatter_metad.png', dpi=800)
            plt.close()


        if render_3D:
            print("plotting")
            plt.figure()
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(bottom=0.2, left=-0.2)
            surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0.2, rstride=5, cstride=5, alpha=0.5)
            #ax.contourf(X, Y, Z, zdir='z', offset=0, cmap="coolwarm", alpha=0.8)

            #plot the traj. thin line every 1000 frames connected with dots.
            ax.plot3D(mfpt_traj[:,0], mfpt_traj[:,1], mfpt_traj[:,2], color="red", marker='o',markersize=3.5, linestyle='-', linewidth=2, alpha=1)
            ax.plot3D(metad_traj[:,0], metad_traj[:,1], metad_traj[:,2], color="blue", marker='o',markersize=3.5, linestyle='dashed', alpha=0.5)
            
            ax.view_init(55, -45) #45 -45
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

            cbar_ax = fig.add_axes([0.75, 0.15, 0.02, 0.7]) #left, bottom, width, height
            cbar = fig.colorbar(surf, shrink=0.5, aspect=10, cax=cbar_ax)
            cbar.set_label("U (kcal/mol)")
            plt.savefig('./2Dtraj_visual_3D_traj.png', dpi=800)
            plt.close()

    if plot_1d_fes:
        amp = 6
        num_hills = 9
        A_i = np.array([0.9, 0.3, 0.7, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
        x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 4.75, 6, 1] # this is in nm.
        sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]

        render_1D = False
        render_3D = True

        if render_1D:
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

            #plotting.
            plt.figure()
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(bottom=0.2)

            fig, ax = plt.subplots(layout='constrained')
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

            plt.plot(x, y, color="blue", alpha=0.7, linewidth=2)
            plt.xlabel("x (Å)")
            plt.xlim([0, 2*np.pi])
            #plt.ylim([0, 10])
            plt.ylabel("U (kcal/mol)")
            plt.savefig('./1D_fes_visual_render1D.png', dpi=800)
            plt.close()

        if render_3D:
            X,Y = np.meshgrid(np.linspace(0, 2*np.pi, 100), np.linspace(-1, 1, 1000))
            Z = np.zeros_like(X)
            for i in range(num_hills):
                Z += A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2))
            Z = Z - Z.min()

            #sigmoid on both end of x.
            k = 5
            max_barrier = '1e3'
            offset = 0.4
            Z += float(max_barrier)/4.184 * (1 / (1 + np.exp(k * (X - (-offset)))))
            Z += float(max_barrier)/4.184 * (1 / (1 + np.exp(-k * (X - (2 * np.pi + offset)))))
            
            #oscillator like constraint on y and z.
            os_k = 1e3/4.184
            Z += os_k * (Y)**2
            
            #plotting.
            plt.figure()
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(bottom=0.4)
            fig, ax = plt.subplots(layout='constrained', subplot_kw={"projection": "3d"})
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

            surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0.2, rstride=5, cstride=5, alpha=1)
            ax.view_init(55, -45) #45 -45

            plt.xlabel("x (Å)")
            plt.xlim([0, 2*np.pi])
            #plt.ylim([-0.2, 0.2])
            plt.ylabel("y (Å)")

            cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7]) #left, bottom, width, height
            cbar = fig.colorbar(surf, shrink=0.5, aspect=10, cax=cbar_ax)
            cbar.set_label("U (kcal/mol)")
            plt.savefig('./1D_fes_visual_render3D.png', dpi=800)
            plt.close()

    if plot_2d_fes:
        amp = 6
        num_wells = 9
        num_barrier = 1
        k = 5  # Steepness of the sigmoid curve
        max_barrier = "1e2"  # Scaling factor for the potential maximum
        offset = 0.7 #the offset of the boundary energy barrier.
        A_i = np.array([0.9, 0.3, 0.5, 1, 0.2, 0.4, 0.9, 0.9, 0.9]) * amp #this is in kcal/mol.
        x0_i = [1.12, 1, 3, 4.15, 4, 5.27, 5.5, 6, 1] # this is in nm.
        y0_i = [1.34, 2.25, 2.31, 3.62, 5, 4.14, 4.5, 1.52, 5]
        sigma_x_i = [0.5, 0.3, 0.4, 2, 0.9, 1, 0.3, 0.5, 0.5]
        sigma_y_i = [0.5, 0.3, 1, 0.8, 0.2, 0.3, 1, 0.6, 0.7]

        A_j = np.array([0.3]) * amp
        x0_j = [np.pi]
        y0_j = [np.pi]
        sigma_x_j = [3]
        sigma_y_j = [0.3]

        x = np.linspace(0, 2*np.pi, 100)
        y = np.linspace(0, 2*np.pi, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        Z += amp * 4.184 #flat surface
        for i in range(num_wells):
            Z -= A_i[i] * np.exp(-(X-x0_i[i])**2/(2*sigma_x_i[i]**2) - (Y-y0_i[i])**2/(2*sigma_y_i[i]**2))
        for i in range(num_barrier):
            Z += A_j[i] * np.exp(-(X-x0_j[i])**2/(2*sigma_x_j[i]**2) - (Y-y0_j[i])**2/(2*sigma_y_j[i]**2))
        total_energy_barrier = np.zeros_like(X)
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (X - (-offset))))) #left
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (X - (2 * np.pi + offset))))) #right
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(k * (Y - (-offset)))))
        total_energy_barrier += float(max_barrier) * (1 / (1 + np.exp(-k * (Y - (2 * np.pi + offset)))))
        Z += total_energy_barrier
        fes_min = Z.min()
        Z = Z - fes_min

        render_2D = True
        render_3D = True

        #plotting.
        if render_2D:
            plt.figure()
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(bottom=0.2)
            plt.imshow(Z, cmap="coolwarm", extent=[0, 2*np.pi, 0, 2*np.pi], origin="lower")

            
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))

            plt.xlabel("x (Å)")
            plt.xlim([0, 2*np.pi])
            plt.ylim([0, 2*np.pi])
            plt.ylabel("y (Å)")
            #plt.title("FES mode = multiwell, pbc=False")
            cbar=plt.colorbar()
            cbar.set_label("U (kcal/mol)")
            plt.savefig('./figs/2D_fes_visual_render2D.png', dpi=800)
            plt.close()
        if render_3D:
            print("plotting")
            plt.figure()
            plt.tight_layout(pad=4.0)
            plt.subplots_adjust(bottom=0.2, left = -0.3)

            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax = plt.gca()
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: nm_to_angstrom(x)))
            surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0.2, rstride=5, cstride=5, alpha=1)
            #ax.contourf(X, Y, Z, zdir='z', offset=0, cmap="coolwarm", alpha=0.8)

            ax.view_init(55, -45)
            plt.xlabel("x (Å)")
            plt.xlim([0, 2*np.pi])
            plt.ylim([0, 2*np.pi])
            plt.ylabel("y (Å)")

            cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) #left, bottom, width, height
            cbar = fig.colorbar(surf, shrink=0.4, aspect=10, cax=cbar_ax)
            cbar.set_label("U (kcal/mol)")
            ax.xaxis.labelpad=20
            ax.yaxis.labelpad=20
            plt.savefig('./figs/2D_fes_visual_render3D.png', dpi=800)
            plt.close()
