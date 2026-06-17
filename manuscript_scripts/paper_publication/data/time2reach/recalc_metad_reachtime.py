#load traj, check the first reaching time.
import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

traj_folder_path = './data/traj/2D/metad/'
import os
all_files = os.listdir(traj_folder_path)
#find only the dcd files
dcd_files = [f for f in all_files if f.endswith('.dcd')]
traj_path_list = [os.path.join(traj_folder_path, f) for f in dcd_files]

#we iter all traj file and save the first reaching time in a csv file.
first_reach_time_list = []
for traj_path in traj_path_list:
    traj = md.load(traj_path, top='./data/traj/2D/metad/explore_traj.pdb')
    print("total frames: ", traj.n_frames)
    #check the frame used to reach coor [1.0, 1.5, 0], use distance criteria < 0.1
    for idx, frame in enumerate(traj):
        #print(frame.xyz[0][0])
        x, y = frame.xyz[0][0][0], frame.xyz[0][0][1]

        dist = np.sqrt((x - 1.0)**2 + (y - 1.5)**2)
        if dist < 0.1:
            print(f'frame {idx} reached the target coor at {frame.time}')
            first_reach_time_list.append(frame.time[0]*500)
            break

#save it to csv
with open('./data/time2reach/2D/total_steps_metaD.csv', 'w') as f:
    for time in first_reach_time_list:
        f.write(f'{time}\n')
        


"""traj = md.load(traj_path, top='./data/traj/2D/metad/explore_traj.pdb')
print("total frames: ", traj.n_frames)
#check the frame used to reach coor [1.0, 1.5, 0], use distance criteria < 0.1
for idx, frame in enumerate(traj):
    #print(frame.xyz[0][0])
    x, y = frame.xyz[0][0][0], frame.xyz[0][0][1]

    dist = np.sqrt((x - 1.0)**2 + (y - 1.5)**2)
    if dist < 0.1:
        print(f'frame {idx} reached the target coor at {frame.time}')
        break"""
