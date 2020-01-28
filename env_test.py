import numpy as np
import math
import os, sys

import matplotlib.pyplot as plt

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

projectFile = './Project/' 
sumoBinary = "sumo"
seeds = 20190709
traci.start([sumoBinary, '-c', projectFile + 'ramp.sumo.cfg','--start','--seed',\
                 str(seeds), '--emergencydecel.warning-threshold', '1.1', '--quit-on-end'], label='training')
scenario = traci.getConnection('training')
lane = ['ramp_0', 'mainline_control_0', 'mainline_control_1', 'mainline_control_2']

for step in range(7500):
    position = np.zeros((1, 10, 201), dtype=np.float32)
    velocity = np.zeros((1, 10, 201), dtype=np.float32)
    traci.simulationStep()
    x_min =95
    y_min =1141
    if step > 300:
        '''if step > 6000:
            traci.edge.setMaxSpeed('mainline_control', 8.33)
            print('New speed limit applied')'''
        for idx in lane:
            curent_veh = traci.lane.getLastStepVehicleIDs(idx)
            for veh in curent_veh:
                pos = traci.vehicle.getPosition(veh)
                y = pos[0]
                x = pos[1]
                if x_min > x:
                    x_min = x
                if y_min > y:
                    y_min = y
                #print(x,y)
                '''if position[0][x][y] == 0:
                    velocity[0][x][y] += traci.vehicle.getSpeed(veh)
                else:
                    velocity[0][x][y] = (velocity[0][x][y] * position[0][x][y] + traci.vehicle.getSpeed(veh)) / (position[0][x][y] + 1)
                position[0][x][y] += 1.0'''
        '''if step > 6000:
            print(np.max(velocity))'''
        #state = np.concatenate((position, velocity), axis=0)
        #print(state.shape)
        '''if step  == 6000:
            fig = plt.figure(figsize=(40,8))
            ax1 = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
            ax1.imshow(position[0], cmap='gray',origin='lower')
            ax1.set_title('Position')
            ax2.imshow(velocity[0], cmap='gray',origin='lower')
            ax2.set_title('Velocity')
            plt.savefig('./test_state/state.svg')
            print('Done!')'''
print(x_min,y_min)

