# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 13:36:09 2019

@author: ChocolateDave
"""

# Import Modules
import argparse
import collections
import copy
import numpy as np
import os,sys
sys.path.append("./lib")
sys.path.append("./common")
import torch
import torch.nn as nn
import torch.optim as optim

import env as Env
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from lib import agent, action, experience, parameters, saver, tracker, wrappers

# Global Variable:
params = parameters.Constants
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

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

# Training
def Train(cuda, name='3DQN', path=None, frameskip=15):
    # CUDA Assistant
    print("CUDA™ is " + ("AVAILABLE" if torch.cuda.is_available() else "NOT AVAILABLE"))
    device = torch.device("cuda" if torch.cuda.is_available() and cuda else "cpu")
    if cuda:
        torch.backends.cudnn.benchmark = True

    writer = SummaryWriter(comment =name)
    if not os.path.exists('./savednetwork/') and path == None:
        os.makedirs('./savednetwork/')
    else:
        path_net = path

    # environment initialization
    env = Env.SumoEnv(frameskip=frameskip, device=device)
    #env = wrappers.wrap_dqn(env, reward_clipping=False)
    state_size = env.observation_space
    action_size = env.action_space
    
    #Networ initialization
    net = DuelingNetwork(state_size, action_size).to(device)
    if cuda:
        assert next(net.parameters()).is_cuda, "CUDA ERROR!"
    #print("Observation space: {}, Action size:{}".format(state_size, action_size))
    #Load previously trained network
    if os.path.isfile(path_net):
        print("=> Loading checkpoint '{}'".format(path_net))
        net, pre_frame = saver.load_model(net, path_net)
        print("Checkpoint loaded successfully! ")
        net.train()
    else:
        pre_frame = 0
        print("=> No such checkpoint at '{}'".format(path_net))
        path_net = os.path.join('./savednetwork/', name+'.pth')
    
    #Training initialization
    selector = action.EpsilonGreedyActionSelector(epsilon=params['epsilon_start']) 
    epsilon_tracker = tracker.EpsilonTracker(selector, params)
    tgt_net = agent.TargetNet(net)
    agents = agent.DQNAgent(net, selector, device=device)
    exp_source = experience.ExperienceSourceFirstLast(env, agents, gamma=params['gamma'], steps_count=1)
    buffer = experience.PrioritizedReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    #Initialize
    frame_idx = 0
    beta = BETA_START
    t = True

    with tracker.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue
            
            if t:
                print("Training begins...")
                t = False
            
            optimizer.zero_grad()
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            loss_v, sample_prios_v = experience.calc_loss(batch, batch_weights, net, tgt_net.target_model,
                                               params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['checkpoint'] == 0:
                    saver.save_model(net, path_net, frame_idx + pre_frame - params['replay_initial'])
                    print("=> Checkpoint reached.\n=>Network saved at %s" % path_net)

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()
    net.eval()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default='network_checkpoint', type=str, dest='name', help='train period name')
    parser.add_argument("--resume", default = "checkpoint.pth", type = str, dest='path', help= 'path to latest checkpoint')
    parser.add_argument("--skip", default=180, type=int, dest='frameskip', help='frameskip of env')
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    Train(args.cuda, args.name, args.path, args.frameskip)