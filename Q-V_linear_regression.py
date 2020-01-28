import torch
import torchvision
from torch import nn, optim
import numpy as np

import gym
from gym.utils import seeding

import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt 

import traci
import os, sys
from sumolib import checkBinary
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

#Linear Regression model
class LinearRegression(nn.Module):
    '''Create model for linear regression to simulate the relationship between Q & v'''

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
    
    def forward(self, x):
        out = self.linear(x)
        return out

def _getseed(seed= None):
    _, seed1 = seeding.np_random(seed)
    # Derive a random seed. This gets passed as a uint, but gets
    # checked as an int elsewhere, so we need to keep it below
    # 2**31.
    seed2 = seeding.hash_seed(seed1 + 1) % 2**31
    return [seed1, seed2]

def traci_input():
    Total_time = 9000
    projectFile = './project/'
    sumoBinary = "sumo"
    seed = _getseed()[1]
    traci.start([sumoBinary, '-c', projectFile + 'ramp.sumo.cfg', '--start', '--emergencydecel.warning-threshold', '1.1', \
        '--seed', str(seed), '--quit-on-end'], label='training')
    scenario = traci.getConnection('training')
    # initialize lane_list and edge_list
    lanearea_dec_list = list()
    lanearea_max_speed = dict()
    lanearea_ob = list()
    lane_list = list()

    # initialize lanearea_dec_list
    net_tree = ET.parse("./project/ramp.net.xml")
    for lane in net_tree.iter("lane"):
        lane_list.append(lane.attrib["id"])
    dec_tree = ET.parse("./project/ramp.add.xml")
    for lanearea_dec in dec_tree.iter("laneAreaDetector"):
        lanearea_ob.append(lanearea_dec.attrib["id"])
        if lanearea_dec.attrib["freq"] == '60':
            lanearea_dec_list.append(lanearea_dec.attrib["id"])
            lanearea_max_speed[lanearea_dec.attrib["id"]] = 22.22
    
    #warm up simulation
    warm_step=0
    while warm_step <= 3 * 1e2:
        traci.simulationStep()
        warm_step += 1
    
    #get density and speed
    des = np.zeros((Total_time, 1), dtype=np.float32)
    speed = np.zeros((Total_time,1), dtype=np.float32)

    #run and record
    for i in range(Total_time):
        j = 0
        lane_speed = 0.0
        lane_density = 0.0
        for lane in lane_list:
            if "mainline_up_0" in lane:
                lane_speed += traci.lane.getLastStepMeanSpeed(lane)
                lane_density += traci.lane.getLastStepVehicleNumber(lane)
                j += 1
        des[i] = lane_density / j
        speed[i] = lane_speed / j
        traci.simulationStep()
    traci.close(False)

    return des, speed

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    model = LinearRegression().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-4)
    epoch = 0
    inputs, target = traci_input()
    inputs = torch.tensor(inputs).to(device)
    target = torch.tensor(target).to(device)
    
    plt.style.use("dark_background")
    plt.ion()
    plt.show()
    while True:
        epoch += 1

        #forward
        out = model(inputs)
        loss = criterion(out, target)
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #info
        if (epoch + 1) % 2000 ==0:
            model.eval()
            prediction = model(inputs)
            prediction = prediction.data.cpu().numpy()
            x = inputs.data.cpu().numpy()
            y = target.data.cpu().numpy()
            plt.cla()
            plt.scatter(x, y, label='Original Data')
            plt.plot(x, prediction, 'r-', lw=5, label="Linear Regression")
            print('Epoch: %d' % epoch, 'Loss=%.4f' % loss.cpu().data.numpy(), "k->0, vf={}".format(model(torch.tensor([0.0]).to(device))))
            '''for name, parm in model.named_parameters():
                print("{}: {}".format(name, parm))'''
            plt.pause(0.1)
            model.train()

if __name__ == "__main__":
    train()