import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from xml.dom import minidom
from lib import parameters as params

def load_data(filepath, name):
    traj_xml = minidom.parse(filepath)
    traj_dict = dict()
    counter = 0

    for timestep in traj_xml.getElementsByTagName('timestep'):
        time = float(timestep.getAttribute('time'))

        if time < 5400 or time > 7200 or time % 5 != 0:
            continue

        
        for edge in timestep.getElementsByTagName('edge'):
            edge_id = params.EDGE_CODING[edge.getAttribute('id')]
            for lane in edge.getElementsByTagName('lane'):
                lane_id = lane.getAttribute('id')
                if "mainline_down_1" in lane_id or "mainline_down_3" in lane_id:
                    continue
                
                lane_id = params.LANE_CODING[lane_id]

                for vehicle in lane.getElementsByTagName('vehicle'):

                    veh_id = vehicle.getAttribute('id')

                    pos = float(vehicle.getAttribute('pos'))
                    speed = float(vehicle.getAttribute('speed'))

                    if veh_id not in traj_dict.keys():
                        traj_dict[veh_id] = []
                    
                    traj_dict[veh_id].append(pd.DataFrame({'vehicle': veh_id, 'time': time, \
                        'edge': edge_id, 'lane': lane_id, 'pos': pos, 'pos_abs': pos + params.MILE_STONE[edge_id], \
                            'speed': speed}, index=[str(counter)]))
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

    #FileStore
    h5 = pd.HDFStore('./benchmark/data/{}.h5'.format(name), 'w')
    h5['data'] = traj_df
    h5.close()

    return traj_df

def plot(df, name=None):
    plt.figure(figsize=(16, 8), dpi=80)
    fontSetting = {'family': 'Times New Roman', 'size': 12}
    for group in df.groupby(['vehicle', 'trajSeg']):
        if group[1].lane[0] != 0:
            continue
        #print(type(group[1].speed))
        if 'ramp' in group[1].vehicle[0]:
            if group[1].isLaneChanging[0]:
                plt.scatter(group[1].time.astype('float'), group[1].pos_abs, c=group[1].speed, cmap='Blues')
            else:
                plt.scatter(group[1].time.astype('float'), group[1].pos_abs, c=group[1].speed, cmap='Blues')
        else:
            if group[1].isLaneChanging[0]:
                plt.scatter(group[1].time.astype('float'), group[1].pos_abs, c=group[1].speed, cmap='Reds')
            else:
                plt.scatter(group[1].time.astype('float'), group[1].pos_abs, c=group[1].speed, cmap='Reds')
    plt.clim(0.0, 22.22)
    plt.ylim([1800, 2500])
    plt.xlim([5800, 6400])
    plt.xlabel('Time (sec)', fontSetting)
    plt.ylabel('Distance (m)', fontSetting)
    plt.suptitle('Space-time Diagram', fontname=fontSetting['family'], fontsize=fontSetting['size'])
    #plt.legend()
    plt.tight_layout()
    plt.savefig('./benchmark/graph/{}.svg'.format(name))

if __name__ == "__main__":
    nct_pt = "./Project/Output/Traj/traj_ori_"
    cus_pt = "./Project/Output/Traj/traj_cus_"
    nn_pt = "./Project/Output/Traj/traj_nn_"
    iters = 5
    print("Duty begins...")
    for i in range(iters):
        #ori
        ori_path = './benchmark/data/{}.h5'.format("traj_ori_" + str(i+1))
        if os.path.isfile(ori_path):
            df = pd.read_hdf(ori_path,key='data')
        else:
            df = load_data(nct_pt + str(i+1) + ".xml", "traj_ori_" + str(i+1))
        plot(df, name="traj_ori_" + str(i+1))
        #cus
        cus_path = './benchmark/data/{}.h5'.format("traj_cus_" + str(i+1))
        if os.path.isfile(cus_path):
            df = pd.read_hdf(cus_path,key='data')
        else:
            df = load_data(cus_pt + str(i+1) + ".xml", "traj_cus_" + str(i+1))
        plot(df, name="traj_cus_" + str(i+1))
        #nn
        nn_path = './benchmark/data/{}.h5'.format("traj_nn_" + str(i+1))
        if os.path.isfile(nn_path):
            df = pd.read_hdf(nn_path,key='data')
        else:
            df = load_data(nn_pt + str(i+1) + ".xml", "traj_nn_" + str(i+1))
        plot(df, name="traj_nn_" + str(i+1))
    print('Done!')
