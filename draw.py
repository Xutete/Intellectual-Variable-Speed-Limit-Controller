import numpy as np
import matplotlib.pyplot as plt
from xml.dom import minidom

if __name__ == "__main__":
    tree = minidom.parse("traj13-57-12.xml")
    collection = tree.documentElement
    vehicle_list = list()
    ramp_vehicle = list()
    vehicle_position = dict()
    timestep = collection.getElementsByTagName('timestep')
    for time in timestep:
        time_index = float(time.getAttribute("time"))
        print(time_index)
        vehicles = time.getElementsByTagName("vehicle")
        for vehicle in vehicles:
            vehicle_id = vehicle.getAttribute("id")
            vehicle_edge = vehicle.parentNode.parentNode.getAttribute("id")
            if vehicle_id not in vehicle_list:
                if vehicle_edge == 'ramp':
                    if vehicle_id in ramp_vehicle:
                        continue
                    ramp_vehicle.append(vehicle.getAttribute("id"))
                elif (vehicle_edge == 'mainline_up' or vehicle_edge == 'merging' or vehicle_edge == 'mainline_down'):
                    vehicle_list.append(vehicle_id)
                    vehicle_position[vehicle_id] = {'time':list(), 'pos':list()}

            else:
                if time_index < 5400:
                    continue
                vehicle_pos = float(vehicle.getAttribute("pos"))
                if vehicle_edge == 'merging':
                    vehicle_pos = vehicle_pos + 2195
                elif vehicle_edge == 'mainline_down':
                    vehicle_pos = vehicle_pos + 2195 + 123
                vehicle_position[vehicle_id]['time'].append(time_index)
                vehicle_position[vehicle_id]['pos'].append(vehicle_pos)

    ###
    plt.subplot(1, 1, 1)
    
    for vehicle in vehicle_list:
        isRamp = vehicle in ramp_vehicle
        if isRamp:
            plt.plot(np.array(vehicle_position[vehicle]['time']), np.array(vehicle_position[vehicle]['pos']), color="blue")
        else:
            plt.plot(np.array(vehicle_position[vehicle]['time']), np.array(vehicle_position[vehicle]['pos']), color="red")

    plt.show()


                    