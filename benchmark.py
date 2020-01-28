import matplotlib
import matplotlib.pyplot as plt
from lib import agent, action, experience, parameters
from torch.autograd import Variable
import env as benv
import xml.etree.ElementTree as ET
from xml.dom import minidom
import torch.optim as optim
import torch.nn as nn
import torch
import collections
import numpy as np
import os
import sys
import time
import copy
sys.path.append("./lib")
sys.path.append("./common")

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci

# Global Variables
frameskip = 180
TOTAL_TIME = 9000
VEHICLE_MEAN_LENGTH = 5
Total_Time = int(TOTAL_TIME / frameskip) + 2
params = parameters.Constants
fontSetting = {'family': 'Times New Roman', 'size': 12}

# Build Up Neural Network


class DuelingNetwork(nn.Module):
    """
    Create a neural network to convert image data
    """
    def __init__(self, input_shape, n_actions):
        super(DuelingNetwork, self).__init__()

        self.convolutional_Layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fully_connected_adv = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )
        self.fully_connected_val = nn.Sequential(
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_out(self, shape):
        #print('1, *shape: ', torch.zeros(1, *shape).size())
        o = self.convolutional_Layer(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = x.float()
        conv_out = self.convolutional_Layer(x).view(x.size()[0], -1)
        val = self.fully_connected_val(conv_out)
        adv = self.fully_connected_adv(conv_out)
        return val + adv - adv.mean()


def bench_load_model(net, path_net):
    state_dict = torch.load(path_net)
    net.load_state_dict(state_dict['state_dict'])
    frame = state_dict['frame']
    print("Having pre-trained %d frames." % frame)
    net.eval()
    return net, frame


def custom_opt(obs, vc) -> float:
    best_occ = 14.0
    K = .15
    vsl = vc + K*(best_occ - obs['occ'])
    return np.clip(vsl, 8.33, 22.22)

def Florida(obs) -> float:
    if obs['flow'] > 1750:
        if obs['speed'] <= 12.5:
            return 11.11
        elif obs['speed'] <= 15.28:
            return 13.89
        elif obs['speed'] <= 18.06:
            return 16.67
        elif obs['speed'] <= 20.83:
            return 19.44
        else:
            return 22.22
    else:
        if obs['occ'] <= 23:
            return 22.22
        else:
            if obs['speed'] <=12.5:
                return 11.11
            elif obs['speed'] <= 15.28:
                return 13.89
            elif obs['speed'] <=18.06:
                return 16.67
            elif obs['speed'] <= 20.83:
                return 19.44
            else:
                return 22.22

'''def Florida(obs, vc) -> float:
    if obs['flow'] > 1750:
        if obs['speed'] <= vc - 1.39:
            act = vc - 2.78
        elif obs['speed'] >= vc + 1.39:
            act = vc + 2.78
        else:
            act = vc
    else:
        if obs['occ'] <= 16:
            return 22.22
        else:
            if obs['speed'] <= vc - 1.39:
                act = vc - 2.78
            elif obs['speed'] >= vc + 1.39:
                act = vc + 2.78
            else:
                act = vc
    return np.clip(act, 11.11, 22.22)'''

#Speed Performance
def parse_evaluation(detector_group, output_file, performance, period):

    # Index for list of performance
    x_detector = dict()
    y_detector = dict()

    # Initialize list of performance
    performance_list = []
    for i in range(len(detector_group)):
        performance_list.append([])
        for j in range(len(detector_group[i])):
            performance_list[i].append([])
            
            key = detector_group[i][j].split('_')[-2] + detector_group[i][j].split('_')[-1]
            x_detector[key] = i
            y_detector[key] = j

    DOMTree = minidom.parse(output_file)
    collection = DOMTree.documentElement.getElementsByTagName('interval')

    for item in collection:
        if float(item.getAttribute('begin')) >= period[0] and float(item.getAttribute('begin')) <= period[1]:
            
            key = item.getAttribute('id').split('_')[-2] + item.getAttribute('id').split('_')[-1]

            if x_detector.__contains__(key):
                performance_list[x_detector[key]][y_detector[key]].append(float(item.getAttribute(performance)))
    
    return performance_list

def speed_contourf(speed_list, name, fontSetting):
    distance = [0, 400, 400, 250, 50, 210, 90]

    mainline_speed_np = np.array(speed_list)
    #pint(mainline_speed_np.shape)
    average_mainline_speed_np = np.mean(mainline_speed_np, axis=(0, 2))
    #print(average_mainline_speed_np.shape)

    plt.figure(figsize=(12, 4))
    norm = matplotlib.colors.Normalize(vmin=8, vmax=22)
    plt.contourf(range(average_mainline_speed_np.shape[1]), np.cumsum(distance), average_mainline_speed_np, cmap="jet_r", norm=norm)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Speed (m/s)', fontSetting)


    plt.contour(range(average_mainline_speed_np.shape[1]), np.cumsum(distance), average_mainline_speed_np)
    plt.xlabel('Time (min)', fontSetting)
    plt.ylabel('Induction Loop ID', fontSetting)

    xTicks = np.linspace(0, 301, 60)
    xTickslabels=['8:00','8:30','9:00', '9:30', '10:00', '10:30']
    plt.gca().set_xticks(xTicks)
    plt.gca().set_xticklabels(xTickslabels, fontSetting)
    
    plt.gca().set_yticks(np.cumsum(distance))
    plt.gca().set_yticklabels(['0', '1', '2', '3', '4', '5', '6'], fontSetting)

    plt.title('Speed along the mainline ({})'.format(name), fontSetting)
    plt.savefig("./graph/{}.svg".format(name))



def runtime(i):
    num_arrow = int(i * 50 / Total_Time)
    num_line = 50 - num_arrow
    percent = i * 100.0 / Total_Time
    process_bar = 'Scenario Running... [' + '>' * num_arrow + \
        '-' * num_line + ']' + '%.2f' % percent + '%' + '\r'
    sys.stdout.write(process_bar)
    sys.stdout.flush()
    if percent == 100:
        sys.stdout.write('\n')


class BenchmarkEpsilonTracker:
    def __init__(self, epsilon_greedy_selector):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = 0


def benchmark(iters, env, agents=None, net=None):
    slices = int(Total_Time*frameskip/60) + 2

    # data containers
    data_ori = np.zeros((4, slices), dtype=np.float32)
    mainline_speed_ori = []
    var_ori = np.zeros(slices, dtype=np.float32)
    action_ori = np.zeros(Total_Time, dtype=np.float32)

    data_cus = np.zeros((4, slices), dtype=np.float32)
    mainline_speed_cus = []
    var_cus = np.zeros(slices, dtype=np.float32)
    action_cus = np.zeros(Total_Time, dtype=np.float32)

    data_nn = np.empty((4, slices), dtype=np.float32)
    mainline_speed_nn = []
    var_nn = np.zeros(slices, dtype=np.float32)
    action_nn = np.zeros(Total_Time, dtype=np.float32)
    reward_list = list()
    seed = np.load('./logs/seed_20.npy')
    act_list = env.action_set



    # Env initialization
    if net is None:
        print("No network found!")
        net = DuelingNetwork(env.observation_space, env.action_space)
        path_net = os.path.join('./savednetwork/', 'network_checkpoint.pth')
        if path_net:
            if os.path.isfile(path_net):
                print("=> Loading checkpoint '{}'".format(path_net))
                net, _ = bench_load_model(net, path_net)
                net.to(device)
                print("Checkpoint loaded successfully! ")
        if next(net.parameters()).is_cuda:
            print("Now using {} for benchmark".format(
                torch.cuda.get_device_name(torch.cuda.current_device())))
    else:
        net = net.to(device)
    
    selector = action.EpsilonGreedyActionSelector(epsilon=0)
    epsilon_tracker = BenchmarkEpsilonTracker(selector)
    agents = agent.DQNAgent(net, selector, device=device)

    for timestep in range(iters):
        print("\n******stage {}*******".format(timestep+1))
        
        # First run a scenario without VSL
        env.reset(label='_ori_{}'.format(timestep + 1), eval_seed = seed[timestep])
        i = 0
        reward_list.clear()
        while True:
            action_ori[i] = 22.22
            _, reward, done, info = env.step(22.22)
            i += 1
            j = env.run_step
            reward_list.append(reward)
            if j % 60 == 0:
                var_ori[int(j / 60) -1 ] = info['var']
            if done:
                break
            # runtime(i)
        print("reward:{}".format(np.sum(reward_list)))
        tree = minidom.parse("./Project/Output/multi_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = int(float(interval.getAttribute("begin"))-300)
            if begin <= 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                data_ori[0][int(begin / 60)-1] += float(interval.getAttribute("meanSpeed"))
                data_ori[1][int(begin / 60)-1] += float(interval.getAttribute("meanTravelTime"))
        tree = minidom.parse("./Project/Output/mainline_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "mainline_eval_6" in interval.getAttribute("id"):
                    data_ori[2][int(begin / 60)-1] += float(interval.getAttribute("flow"))
        mainline_speed_ori.append(parse_evaluation(parameters.induction_loop_mainline, \
            "./Project/Output/mainline_eval.out", 'speed', [300, 9300]))
        speed_contourf(mainline_speed_ori, 'No Control', fontSetting)
        tree = minidom.parse("./Project/Output/eval_obs.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "ramp" in interval.getAttribute("id"):
                    data_ori[3][int(begin / 60)-1] += float(interval.getAttribute("jamLengthInMetersSum"))

        mainline_speed_ori.append(parse_evaluation(parameters.induction_loop_mainline, \
            "./Project/Output/mainline_eval.out", 'speed', [300, 9300]))
        speed_contourf(mainline_speed_ori, 'No Control', fontSetting)
        #print(np.array(mainline_speed_ori).shape)
        #continue
        
        # Then run a scenario with custom VSL regulations
        env.reset(label='_cus_{}'.format(timestep + 1), eval_seed = seed[timestep])
        act = 22.22
        i = 0
        reward_list.clear()
        while True:
            action_cus[i] = act
            _, reward, done, info = env.step(act)
            i += 1
            j = env.run_step
            reward_list.append(reward)
            act = Florida(info)
            #print(act)
            if j % 60 == 0:
                var_cus[int(j / 60) - 1] = info['var']
            if done:
                break
            # runtime(i)
            # print("A scenario finished!")
        print("reward:{}".format(np.sum(reward_list)))
        tree = minidom.parse("./Project/Output/multi_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = int(float(interval.getAttribute("begin"))-300)
            if begin <= 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                data_cus[0][int(begin / 60)-1] += float(interval.getAttribute("meanSpeed"))
                data_cus[1][int(begin / 60)-1] += float(interval.getAttribute("meanTravelTime"))
        tree = minidom.parse("./Project/Output/mainline_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "mainline_eval_6" in interval.getAttribute("id"):
                    data_cus[2][int(begin / 60)-1] += float(interval.getAttribute("flow"))
        tree = minidom.parse("./Project/Output/eval_obs.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "ramp" in interval.getAttribute("id"):
                    data_cus[3][int(begin / 60)-1] += float(interval.getAttribute("jamLengthInMetersSum"))
        
        mainline_speed_cus.append(parse_evaluation(parameters.induction_loop_mainline, \
            "./Project/Output/mainline_eval.out", 'speed', [300, 9300]))
        speed_contourf(mainline_speed_cus, 'Rule-based VSL', fontSetting)
        
        # Finally run a scenario with neural network guided VSL
        state = env.reset(label='_nn_{}'.format(timestep+1), eval_seed = seed[timestep])
        i = 0
        reward_list.clear()
        while True:
            epsilon_tracker.frame(i)
            act = agents(np.expand_dims(state,0))[0][0]
            #action_nn[i] = np.clip(env.action+act_list[act], 11.11,22.22)
            action_nn[i] = act_list[act]
            state, reward, done, info = env.step(act)
            i += 1
            j = env.run_step
            reward_list.append(reward)
            if j % 60 == 0:
                var_nn[int(j / 60) - 1] = info['var']
            if done:
                break
            # runtime(i)
        # print("A scenario finished!")
        time.sleep(10)
        print("reward:{}".format(np.sum(reward_list)))
        tree = minidom.parse("./Project/Output/multi_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = int(float(interval.getAttribute("begin"))-300)
            if begin <= 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                data_nn[0][int(begin / 60)-1] += float(interval.getAttribute("meanSpeed"))
                data_nn[1][int(begin / 60)-1] += float(interval.getAttribute("meanTravelTime"))
        tree = minidom.parse("./Project/Output/mainline_eval.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "mainline_eval_6" in interval.getAttribute("id"):
                    data_nn[2][int(begin / 60)-1] += float(interval.getAttribute("flow"))
        tree = minidom.parse("./Project/Output/eval_obs.out")
        collection = tree.documentElement
        intervals = collection.getElementsByTagName('interval')
        for interval in intervals:
            begin = float(interval.getAttribute("begin")) - 300
            if begin < 0 or begin > 9000:
                continue
            if begin % 60 == 0:
                if "ramp" in interval.getAttribute("id"):
                    data_nn[3][int(begin / 60)-1] += float(interval.getAttribute("jamLengthInMetersSum"))
        
        mainline_speed_nn.append(parse_evaluation(parameters.induction_loop_mainline, \
            "./Project/Output/mainline_eval.out", 'speed', [300, 9300]))
        speed_contourf(mainline_speed_nn, 'HDQN-based VSL', fontSetting)

    # save data
    data_ori = data_ori / iters
    data_cus = data_cus / iters
    data_nn = data_nn / iters
    np.save("logs/data.npy",np.stack((data_ori,data_cus,data_nn)))
    np.save("logs/var.npy",np.stack((var_ori,var_cus,var_nn)))
    np.save("logs/act.npy",np.stack((action_ori,action_cus,action_nn)))
    print("Data saved!")
    
    dom = minidom.Document()
    root_node = dom.createElement('root')
    dom.appendChild(root_node)
    ori_node = dom.createElement('ori')
    cus_node = dom.createElement('cus')
    nn_node = dom.createElement('nn')
    root_node.appendChild(ori_node)
    root_node.appendChild(cus_node)
    root_node.appendChild(nn_node)

    
    for i in range(len(data_ori[0])):
        o_node = dom.createElement('data')
        o_node.setAttribute('ms', str(data_ori[0][i]))
        o_node.setAttribute('tt', str(data_ori[1][i]))
        o_node.setAttribute('flow', str(data_ori[2][i]))
        o_node.setAttribute('queue', str(data_ori[3][i]))
        o_node.setAttribute('var', str(var_ori[i]))
        ori_node.appendChild(o_node)

        c_node = dom.createElement('data')
        c_node.setAttribute('ms', str(data_cus[0][i]))
        c_node.setAttribute('tt', str(data_cus[1][i]))
        c_node.setAttribute('flow', str(data_cus[2][i]))
        c_node.setAttribute('queue', str(data_cus[3][i]))
        c_node.setAttribute('var', str(var_cus[i]))
        cus_node.appendChild(c_node)

        n_node = dom.createElement('data')
        n_node.setAttribute('ms',str(data_nn[0][i]))
        n_node.setAttribute('tt',str(data_nn[1][i]))
        n_node.setAttribute('flow', str(data_nn[2][i]))
        n_node.setAttribute('queue', str(data_nn[3][i]))
        n_node.setAttribute('var', str(var_nn[i]))
        nn_node.appendChild(n_node)
    

    for i in range (len(action_ori)):
        o_a = dom.createElement('action')
        o_a.setAttribute('action', str(action_ori[i]))
        ori_node.appendChild(o_a)

        c_a = dom.createElement('action')
        c_a.setAttribute('action', str(action_cus[i]))
        cus_node.appendChild(c_a)

        n_a = dom.createElement('action')
        n_a.setAttribute('action', str(action_nn[i]))
        nn_node.appendChild(n_a)

    with open('./logs/benchmark.xml','w',encoding='UTF-8') as fh:
        dom.writexml(fh,indent='',addindent='\t',newl='\n',encoding='UTF-8')




    # Draw benchmark
    x = range(slices)
    x_t = range(52)
    # plt.style.use('dark_background')
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
    # plt.rcParams['axes.unicode_minus'] = False

    speed_contourf(mainline_speed_ori, 'No Control', fontSetting)
    speed_contourf(mainline_speed_cus, 'Rule-based VSL', fontSetting)
    speed_contourf(mainline_speed_nn, 'HDQN-based VSL', fontSetting)

    plt.figure(figsize=(10, 5))
    plt.plot(x, data_ori[1] / 60, 'g-', label='No Control')
    plt.plot(x, data_cus[1] / 60, 'r-.', label='Rule-based VSL')
    plt.plot(x, data_nn[1] / 60, 'b--', label='HDQN-based VSL')
    plt.xlim = ((0,150))
    plt.ylim = ((0, 5))
    plt.xlabel('Time (min)', fontSetting)
    plt.ylabel('Travel Time (sec)', fontSetting)
    plt.suptitle('Travel Time', y=1, fontname=fontSetting['family'], fontsize=fontSetting['size'])
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.savefig("./graph/Travel_Time.svg")

    plt.figure(figsize=(10, 5))
    plt.plot(x, data_ori[0] * 3.6, 'g-', label='No Control')
    plt.plot(x, data_cus[0] * 3.6, 'r-.', label='Rule-based VSL')
    plt.plot(x, data_nn[0] * 3.6, 'b--', label='HDQN-based VSL')
    plt.xlim = ((0,150))
    plt.ylim = ((30,80))
    plt.xlabel('Time (min)', fontSetting)
    plt.ylabel('Average Speed (m/s)', fontSetting)
    plt.suptitle('Average Speed', y=1, fontname=fontSetting['family'], fontsize=fontSetting['size'])
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.savefig("./graph/Average_speed.svg")

    plt.figure(figsize=(10, 5))
    plt.plot(x, data_ori[2], 'g-', label='No Control')
    plt.plot(x, data_cus[2], 'r-.', label='Rule-based VSL')
    plt.plot(x, data_nn[2], 'b--', label='HDQN-based VSL')
    plt.xlim = ((0,150))
    plt.ylim = ((3500,6500))
    plt.xlabel('Time (min)', fontSetting)
    plt.ylabel('Flow (veh/h)', fontSetting)
    plt.suptitle('Merge Flow', y=1, fontname=fontSetting['family'], fontsize=fontSetting['size'])
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.savefig("./graph/Merge_Flow.svg")
    
    plt.figure(figsize=(12, 5))
    plt.plot(x_t, action_ori * 3.6, 'g-', label='No Control')
    plt.plot(x_t, action_cus * 3.6, 'r-.', label='Rule-based VSL')
    plt.plot(x_t, action_nn * 3.6, 'b--', label='HDQN-based VSL')
    # ax1.plot(x, data_ori[2], 'g-', label='Without VSL')
    # ax1.plot(x, data_cus[2], 'r-.', label='Rule Based VSL')
    # ax1.plot(x, data_nn[2], 'b--', label='With NN')
    plt.xlim = ((0,600))
    plt.ylim = ((40,80))
    plt.xticks(range(0,601,120), ['8:00','8:30','9:00', '9:30', '10:00','10:30'])
    plt.xlabel('Time (sec)', fontSetting)
    plt.ylabel('Speed Limit (km/h)', fontSetting)
    # ax1.set_title('Action')
    plt.suptitle('Variable Speed Limits', y=1, fontname=fontSetting['family'], fontsize=fontSetting['size'])
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.legend()
    plt.savefig("./graph/VSL.svg")

    # print("Mean reward: %.1f" % np.mean(data_ori[2]), ", %.1f" % np.mean(data_nn[2]))


if __name__ == "__main__":
    iters = 20
    print("CUDA™ is " + ("AVAILABLE" if torch.cuda.is_available() else "NOT AVAILABLE"))
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        print("Now using CPU for benchmark")
    env = benv.SumoEnv(frameskip=frameskip, device=device, evaluation=True)
    state_size = env.observation_space
    action_size = env.action_space
    net = DuelingNetwork(state_size, action_size)
    path_net = os.path.join('./savednetwork/', 'HDQN.SUCCESS.pth')
    if path_net:
        if os.path.isfile(path_net):
            print("=> Loading checkpoint '{}'".format(path_net))
            net, _ = bench_load_model(net, path_net)
            print("Checkpoint loaded successfully! ")
    if next(net.parameters()).is_cuda:
        print("Now using {} for benchmark".format(
            torch.cuda.get_device_name(torch.cuda.current_device())))
    benchmark(iters,env, net=net)
    print("Benchmark Done!")