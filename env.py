import gym
import numpy as np
import os,sys,time
import math
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import xml.etree.ElementTree as ET
from xml.dom import minidom

import torch
from lib import seeding
from collections import deque
import traci
from sumolib import checkBinary

#Environment Constants    
WARM_UP_TIME = 300
TOTAL_TIME = 9000
VEHICLE_MEAN_LENGTH = 5
EDGE=[146, 45, 2346, 145] #LX, LY, RX, RY

class SumoEnv(gym.Env):
    '''Sumo Environment is a simulation environment which provides necessary parameters for training. On-ramp simulation
    environment could be modified in xml files in project.'''
    #Memory Organization
    __slots__ = ['frameskip', 'run_step', 'scenario',\
        'lanearea_ob', 'action_set', 'evaluation',\
            'sumoBinary', 'projectFile', 'observation_space', 'action_space', 'shape',\
                'return_reward', 'downsample']
    def __init__(self, frameskip=15, downsamples=10, device='cpu', evaluation=False):
        super(SumoEnv, self).__init__()
        #create environment
        if isinstance(frameskip, int):
            self.frameskip = frameskip
        else:
            self.frameskip = np.random.randint(frameskip[0], frameskip[1])
        self.run_step = 0
        self.lanearea_ob = list()
        self.downsample = downsamples
        self.shape = (int((EDGE[3]-EDGE[1])/self.downsample), int((EDGE[2]-EDGE[0])/self.downsample))
        self.evaluation = evaluation
        if self.evaluation:
            self.return_reward = 0.0

        # initialize sumo path
        self.projectFile = './Project/'    
 
        # initialize observation space
        #self.observation_space = (2 * self.downsample, self.shape[0], self.shape[1]) #(samples, h, w)
        self.observation_space = (2, self.shape[0], self.shape[1])

        # initialize action set
        self.action_set = [11.11, 12.51, 13.89, 15.28, 16.67, 18.05, 19.44, 20.83, 22.22] 
        #self.action_set = [8.33, 11.11, 13.89, 16.67, 19.44, 22.22]  # possible actions collection
        self.action_space = len(self.action_set)

        # initialize lanearea_dec_list
        dec_tree = ET.parse("./Project/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            if 'obs' in lanearea_dec.attrib["id"]:
                self.lanearea_ob.append(lanearea_dec.attrib["id"])

    def seed(self, seed= None):
        _, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    def is_episode(self):
        if self.run_step > TOTAL_TIME:
            print('Scenario finished.')
            self.close()
            return True
        '''if self.evaluation == False and self._getmainlinespeed() < 14:
            print('Traffic jammed! at phase %d' % (self.run_step / 1800 + 1))
            traci.close()
            return True'''
        return False

    def warm_up_simulation(self):
        # Warm up simulation.
        warm_step=0
        while warm_step <= WARM_UP_TIME:
            traci.simulationStep()
            warm_step += 1

  
        
    def get_state(self):
        position = np.zeros(self.shape, dtype=np.float32)
        velocity = np.zeros(self.shape, dtype=np.float32)
        
        #First get vehicle info on mainline
        lane = ['ramp_0', 'mainline_up_0', 'mainline_up_1', 'mainline_up_2']
        for idx in lane:
            curent_veh = traci.lane.getLastStepVehicleIDs(idx)
            for veh in curent_veh:
                pos = traci.vehicle.getPosition(veh)
                y = math.floor((pos[0]-146)/self.downsample)
                x = math.floor((pos[1]-45)/self.downsample)
                #print(x,y)
                if position[x][y] == 0:
                    velocity[x][y] += traci.vehicle.getSpeed(veh)
                else:
                    velocity[x][y] = (velocity[x][y] * position[x][y] + traci.vehicle.getSpeed(veh)) / (position[x][y] + 1)
                position[x][y] += 1.0
        #state = np.concatenate((np.stack(np.hsplit(position,self.downsample)), np.stack(np.hsplit(velocity,self.downsample))), axis=0)
        state = np.stack((position, velocity))
        return state

  
    def _getmergingspeed(self):
        ms = list()
        for lanearea in self.lanearea_ob:
            if "merging" in lanearea:
                ms.append(traci.lanearea.getLastStepMeanSpeed(lanearea))
        return np.mean(ms)
    
    def _getmainlinespeed(self):
        ms = list()
        for lanearea in self.lanearea_ob:
            if "mainline" in lanearea:
                ms.append(traci.lanearea.getLastStepMeanSpeed(lanearea))
        return np.mean(ms)

    def _gettraveltime(self):
        mainline=['ramp', 'merging', 'mainline_down']
        ramp=['mainline_up', 'merging', 'mainline_down']
        ttr=0.0
        ttm=0.0
        for item in ramp:
            ttr += traci.edge.getTraveltime(item)
        for item in mainline:
            ttm += traci.edge.getTraveltime(item)
        return np.mean([ttr, ttm])
    
    def _gettotalvehiclelength(self):
        vl = 0.0
        for lanearea in self.lanearea_ob:
            vl += traci.lanearea.getJamLengthVehicle(lanearea)
        return vl
    
    def _getvarmainline(self):
        mls = list()
        for lanearea in self.lanearea_ob:
            if "mainline" in lanearea:
                mls.append(traci.lanearea.getLastStepMeanSpeed(lanearea))
        var = np.var(mls)
        return var

    def _getflow(self):
        ms = list()
        vn = list()
        l = list()
        for lanearea in self.lanearea_ob:
            if "mainline" in lanearea:
                ms.append(traci.lanearea.getLastStepMeanSpeed(lanearea))
                vn.append(traci.lanearea.getLastStepVehicleNumber(lanearea))
                l.append(traci.lanearea.getLength(lanearea))
        return np.mean(ms) * np.mean(vn) / np.mean(l) * 3600
        
    
    def step_reward(self):
        #Reward = w1 * var(mainline) + w2 * queue + w3 * speed
        queue = self._gettotalvehiclelength()
        speed = self._getmergingspeed()
        #flow = self._getflow()
        #tt = self._gettraveltime()
        #maxdec = self._getvehdec()
        #var = self._getvarmainline()
        reward =  .1 * speed - .4 * queue
        #reward = .03* flow - .07 * queue
        #print("queue: {}, speed: {}".format(queue,speed))
        return reward

    def _GetMergingOccupancy(self):
        vn = list()
        l = list()
        for lanearea in self.lanearea_ob:
            if "merging" in lanearea:
                vn.append(traci.lanearea.getLastStepVehicleNumber(lanearea))
                l.append(traci.lanearea.getLength(lanearea))
        return VEHICLE_MEAN_LENGTH * np.mean(vn) / np.mean(l) * 100

    def status_report(self):
        num_arrow = int(self.run_step * 50 / TOTAL_TIME)
        num_line = 50 - num_arrow
        percent = self.run_step * 100.0 / TOTAL_TIME
        process_bar = 'Scenario Running... [' + '>' * num_arrow + '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
    
    def step(self, a):
        '''Conduct an action, update observation and collect reward.'''
        reward = 0.0
        info = dict()
        if a <= self.action_space:
            action = self.action_set[a]
        else:
            action = a
        #print(action)
        traci.edge.setMaxSpeed('mainline_up', action)
        for _ in range(self.frameskip):
            traci.simulationStep()
            reward += self.step_reward()
            self.status_report()
            self.run_step += 1
        observation = self.get_state()
        # Update observation of environment state.
        #reward = self.step_reward()
        #reward = reward / self.frameskip
        #print(reward)
        if self.evaluation:
            info = {'flow':self._getflow(),'speed': self._getmainlinespeed(), 'ms':self._getmergingspeed(), \
                'tt': self._gettraveltime(),'occ':self._GetMergingOccupancy(), 'var': self._getvarmainline()}
        #print(info)  
        done = self.is_episode()   
        return observation, reward, done, info

    def reset(self, label=None, eval_seed=None):
        # Reset simulation with the random seed randomly selected the pool.
        if self.evaluation:
            self.sumoBinary = "sumo"
            seed = eval_seed
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg', '--start',\
                '--seed', str(seed), '--netstate-dump', self.projectFile + \
                    'Output/Traj/traj{}.xml'.format(label), \
                    '--emergencydecel.warning-threshold', '1.2', '--quit-on-end'], label='evaluation')
            self.scenario = traci.getConnection('evaluation')
        else:
            self.sumoBinary = "sumo"
            seeds = self.seed()[1]
            traci.start([self.sumoBinary, '-c', self.projectFile + 'ramp.sumo.cfg','--start','--seed',\
                 str(seeds), '--emergencydecel.warning-threshold', '1.2', '--quit-on-end'], label='training')
            self.scenario = traci.getConnection('training')
        self.warm_up_simulation()
        obs = self.get_state()
        self.run_step = 0
        return obs
    
    def close(self):
        self.scenario.close()



#*************************************************#

    '''
    # initialize lane_list and edge_list
        net_tree = ET.parse("./Project/ramp.net.xml")
        for lane in net_tree.iter("lane"):
            self.lane_list.append(lane.attrib["id"])
        self.observation_space = (2, self.shape[0], self.shape[1])

    # initialize lanearea_dec_list
        dec_tree = ET.parse("./Project/ramp.add.xml")
        for lanearea_dec in dec_tree.iter("laneAreaDetector"):
            if 'obs' in lanearea_dec.attrib["id"]:
                self.lanearea_ob.append(lanearea_dec.attrib["id"])
            if 'vsl' in lanearea_dec.attrib["id"]:
                self.lanearea_dec_list.append(lanearea_dec.attrib["id"])
                self.lanearea_max_speed[lanearea_dec.attrib["id"]] = 22.22

    #Original state extractor  
    def update_observation(self):

        state = np.zeros((1, 3*len(self.lane_list), self.maxlen), dtype = np.float32)

        vehicle_position = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)
        vehicle_speed = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)
        vehicle_acceleration = np.zeros((len(self.lane_list),self.maxlen),dtype = np.float32)

        #originally set -1 on no road sections (abandoned)
        for lane in self.lane_list:
            lane_index = self.lane_list.index(lane)
            lane_len = traci.lane.getLength(lane)
            lane_stop = int (lane_len / VEHICLE_MEAN_LENGTH/self.downsample) 
            for i in range(lane_stop, self.maxlen):
                vehicle_position[lane_index][i] = -1.0

        current_step_vehicle = list()
        for lane in self.lane_list:
            current_step_vehicle += (traci.lane.getLastStepVehicleIDs(lane))

        for vehicle in current_step_vehicle:
            vehicle_in_lane = traci.vehicle.getLaneID(vehicle)
            lane_index = self.lane_list.index(vehicle_in_lane)
            vehicle_pos= traci.vehicle.getPosition(vehicle)
            lane_shape = traci.lane.getShape(vehicle_in_lane)
            vehicle_index = abs(int((vehicle_pos[0]-lane_shape[0][0])/VEHICLE_MEAN_LENGTH))
            vehicle_index = round(vehicle_index / self.downsample)

            vehicle_position[lane_index][vehicle_index] += 1.0
            vehicle_speed[lane_index][vehicle_index] += traci.vehicle.getSpeed(vehicle) 
            vehicle_acceleration[lane_index][vehicle_index] += traci.vehicle.getAcceleration(vehicle)

        for lane_num in range(len(self.lane_list)):
            for vehicle_num in range(len(vehicle_position[lane_num])):
                if vehicle_position[lane_num][vehicle_num] == 0 or vehicle_position[lane_num][vehicle_num] == -1:
                    continue
                vehicle_speed[lane_num][vehicle_num] /= vehicle_position[lane_num][vehicle_num]
                vehicle_acceleration[lane_num][vehicle_num] /= vehicle_position[lane_num][vehicle_num]
        
        state = np.concatenate((vehicle_position, vehicle_speed, vehicle_acceleration), axis= 0)
        return np.expand_dims(state, 0)

    #Original action function
    def reset_vehicle_maxspeed(self):
        for lane in self.lane_list:
            max_speed = 22.22
            for vehicle in traci.lane.getLastStepVehicleIDs(lane):
                traci.vehicle.setMaxSpeed(vehicle,max_speed)
        
        for dec_lane in self.lanearea_dec_list:
            vehicle_list = traci.lanearea.getLastStepVehicleIDs(dec_lane)
            max_speed = self.lanearea_max_speed[dec_lane]
            for vehicle in vehicle_list:
                traci.vehicle.setMaxSpeed(vehicle,max_speed)'''