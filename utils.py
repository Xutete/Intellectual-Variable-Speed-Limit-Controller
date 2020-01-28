import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from xml.dom.minidom import parse
import xml.dom.minidom
import math

from lib import param

import traci
from sumolib import checkBinary

INTERVAL = [param.START_TIME, param.END_TIME]

def LaunchSim(file, label, seed, mode='sumo', trajFile=None):
    startSetting = [checkBinary(mode), "-c", file, "--start", "--quit-on-end", "--step-length", str(param.STEP_SIZE), "--seed", str(seed), "--no-warnings", "true", "--collision.action", "none"]

    if trajFile is None:
        traci.start(startSetting, label=label)
    else:
        traci.start([*startSetting, "--netstate-dump", trajFile], label=label)
        
    return traci.getConnection(label)

def GetEvaluation(detectorGroup, outputFile, metric, interval=INTERVAL):
    # Index for detector
    x_detector = dict()
    y_detector = dict()
    for x in range(len(detectorGroup)):
        for y in range(len(detectorGroup[x])):
            x_detector[detectorGroup[x][y]] = x
            y_detector[detectorGroup[x][y]] = y

    # Initialize list of evaluation
    evaluation = [[[] for j in range(len(detectorGroup[i]))] for i in range(len(detectorGroup))]

    for item in xml.dom.minidom.parse(outputFile).documentElement.getElementsByTagName('interval'):
        if interval[0] <= float(item.getAttribute('end')) <= interval[1]:
            key = item.getAttribute('id')
            if key in x_detector:
                evaluation[x_detector[key]][y_detector[key]].append(float(item.getAttribute(metric)))

    return evaluation


def GetAvgRunningTime(outputFile, interval=INTERVAL):
    totalVeh, totalRunningTime = 0, 0
    for item in xml.dom.minidom.parse(outputFile).documentElement.getElementsByTagName('interval'):
        if interval[0] <= float(item.getAttribute('end')) <= interval[1]:
            totalVeh += int(item.getAttribute('vehicleSum'))
            totalRunningTime += int(item.getAttribute('vehicleSum')) * float(item.getAttribute('meanTravelTime'))
    
    return totalRunningTime / totalVeh


def GetFlowVersusOcc(outputFile, interval=INTERVAL):
    flow, occ = [], []
    for item in xml.dom.minidom.parse(outputFile).documentElement.getElementsByTagName('interval'):
        if interval[0] <= float(item.getAttribute('end')) <= interval[1]:
            flow.append(float(item.getAttribute('flow')))
            occ.append(float(item.getAttribute('occupancy')))
    
    return (flow, occ)


def PlotSpeedContour(speed):
    distance = [0, 1200, 1200, 1400, 50, 70, 200]
    meanSpeed = np.mean(np.array(speed), axis=(0, 2))

    plt.figure(figsize=(12, 4))
    norm = matplotlib.colors.Normalize(vmin=8, vmax=20)
    plt.contourf(range(meanSpeed.shape[1]), np.cumsum(distance), meanSpeed, cmap='jet_r', norm=norm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Speed(m/s)', param.FONT_SETTING)
    
    plt.contour(range(meanSpeed.shape[1]), np.cumsum(distance), meanSpeed)

    xticks = np.linspace(0, 60, 7)    
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(['{}:{}0'.format(8+int(i==6), i*int(i!=6)) for i in range(7)], param.FONT_SETTING)

    plt.ylabel('Distance (m)', param.FONT_SETTING)
    yticks = np.linspace(0, 4000, 9, dtype=np.int16)
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels([str(i) for i in yticks], param.FONT_SETTING)

    plt.show()


def PlotRampQueue(rampQueue):
    rampQueue = np.array(rampQueue).squeeze()
    if len(rampQueue.shape) == 1:
        rampQueue = np.array([rampQueue])
    
    plt.figure(figsize=(12, 4))
    
    PlotTimeSeriesData(rampQueue)
    
    plt.ylabel('Queue length (m)', param.FONT_SETTING)
    #plt.gca().set_yticks(np.linspace(0, 20, 5))
    #plt.gca().set_yticklabels([str(int(i)) for i in np.linspace(0, 20, 5)], FONTSETTING)


    plt.grid()
    plt.show()


def PlotRunningTime(runningTime):
    runningTime = np.array(runningTime).squeeze() / 60
    if len(runningTime.shape) == 1:
        runningTime = np.array([runningTime])
    
    plt.figure(figsize=(12, 4))

    PlotTimeSeriesData(runningTime, 25, 75)
    
    plt.ylabel('Running time (min)', param.FONT_SETTING)
    plt.gca().set_yticks(np.linspace(1.5, 3.5, 5))
    plt.gca().set_yticklabels([str(i) for i in np.linspace(1.5, 3.5, 5)], param.FONT_SETTING)

    plt.grid()
    plt.show()



def PlotTimeSeriesData(data, lowerPercentile=25, upperPercentile=75):

    plt.plot(range(data.shape[-1]), np.mean(data, axis=0), 'b')
    plt.fill_between(range(data.shape[-1]), np.percentile(data, lowerPercentile, axis=0), np.percentile(data, upperPercentile, axis=0), alpha=0.3, color='b', interpolate=True)

    xticks = np.linspace(0, 60, 7)    
    plt.gca().set_xticks(xticks)
    plt.gca().set_xticklabels(['{}:{}0'.format(8+int(i==6), i*int(i!=6)) for i in range(7)], param.FONT_SETTING)


def GetSpeedReward(traci, mergingLoop, speedWeight):
    mergingSpeed = []
    
    for loop in mergingLoop:
        if traci.inductionloop.getLastStepMeanSpeed(loop) > 0:
            mergingSpeed.append(traci.inductionloop.getLastStepMeanSpeed(loop))
    
    meanMergingSpeed = np.mean(mergingSpeed) if len(mergingSpeed) > 0 else 0
    
    return meanMergingSpeed * speedWeight


def GetQueueReward(traci, rampArea, queueWeight):
    return traci.lanearea.getLastStepHaltingNumber(rampArea[0][0]) * queueWeight


def GetTraj(trajFile):
    traj_xml = xml.dom.minidom.parse(trajFile)
    traj_dict = dict()
    counter = 0

    for timestep in traj_xml.getElementsByTagName('timestep'):
        time = float(timestep.getAttribute('time'))

        if time % 3 == 0:
            continue

        
        for edge in timestep.getElementsByTagName('edge'):
            edge_id = param.EDGE_CODING[edge.getAttribute('id')]
            for lane in edge.getElementsByTagName('lane'):
                lane_id = param.LANE_CODING[lane.getAttribute('id')]

                for vehicle in lane.getElementsByTagName('vehicle'):

                    veh_id = vehicle.getAttribute('id')

                    pos = float(vehicle.getAttribute('pos'))
                    speed = float(vehicle.getAttribute('speed'))

                    if veh_id not in traj_dict.keys():
                        traj_dict[veh_id] = []
                    
                    traj_dict[veh_id].append(pd.DataFrame({'vehicle': veh_id, 'time': time, 'edge': edge_id, 'lane': lane_id, 'pos': pos, 'pos_abs': pos+MILESTONE[edge_id], 'speed': speed}, index=[str(counter)]))
                    counter += 1


    traj_df_dict = dict() 
    for key in traj_dict.keys():
        traj_df_dict[key] = pd.concat(traj_dict[key])

        trajSeg = [0]
        for i in range(traj_df_dict[key].shape[0]-1):
            trajSeg.append(int(traj_df_dict[key]['lane'][i] != traj_df_dict[key]['lane'][i+1]))
        traj_df_dict[key]['trajSeg'] = np.cumsum(trajSeg)
        traj_df_dict[key]['isLaneChanging'] = traj_df_dict[key]['trajSeg'] != 0
    
    traj_df = pd.concat([traj_df_dict[key] for key in traj_df_dict.keys()])

    return traj_df


def MoveVeh(time, sumo):
    if time % param.PHASE_LEN == 0:
        vehIDs = sumo.lane.getLastStepVehicleIDs('merging_0')
        if len(vehIDs) > 0:
            vehID = vehIDs[-1]
            if sumo.vehicle.getWaitingTime(vehID) > 0:
                sumo.vehicle.moveTo(vehID, 'merging_1', 125) # Move vehicle "vehID" to lane "merging_1" at position "125"
