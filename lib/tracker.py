import sys
import time
import numpy as np
import torch
import torch.nn as nn

class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", epsilon %.4f" % epsilon
        print("%d: Just done %d episode(s), mean reward %d, speed %.2f f/s epsilon %s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("Interaction/epsilon", epsilon, len(self.total_rewards))
        self.writer.add_scalar("Interaction/speed", speed, len(self.total_rewards))
        self.writer.add_scalar("Interaction/mean_reward", mean_reward, len(self.total_rewards))
        self.writer.add_scalar("Interaction/episodic_reward", reward, len(self.total_rewards))
        if mean_reward > self.stop_reward:
            print("Training finished in episode %d!" % len(self.total_rewards))
            return True
        return False


class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon = params['epsilon_start']
        self.epsilon_start = params['epsilon_start']
        self.epsilon_frames = params['epsilon_frames']
        self.epsilon_final = params['epsilon_final']
        self.frame(0)

    def frame(self, frame):
        '''self.epsilon_greedy_selector.epsilon = \
            self.epsilon_final + (self.epsilon_start - self.epsilon_final) *\
                 np.exp(-self.epsilon_rate * frame)'''
        self.epsilon_greedy_selector.epsilon = \
            max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)