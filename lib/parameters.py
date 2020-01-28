import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Define Constants that could be modified
Constants = {
        'training_name':'training',
        'evaluate_name':'evaluation',

        
        ### MEMORY HYPERPARAMETERS
        'replay_size':      200000, # number of previous transitions to remember
        'replay_initial':   200000,  #Timesteps to observe before training

        # EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
        'epsilon_frames':   400000, # epsilon decay factor
        'epsilon_start':    1.0,    # exploration probability at start
        'epsilon_final':    0.01,  # minimum exploration probability
        'learning_rate':    0.0001,# exponential decay rate for exploration prob

        # Q LEARNING hyperparameters
        'gamma':            0.99,  # Discountings rate

        ### TRAINING HYPERPA-RAMETERS
        'stop_reward':      100000,   # Maximum reward to stop training
        'batch_size':       64,
        'target_net_sync':  1000,
        'checkpoint':       1000

        } 

EDGE_LEN = {'mainline_load': 371.95, ':mainline_start_0': 18.06, 'mainline_up': 2195.18, ':merging_start_1': 9.96, \
        'ramp_load': 257.80, ':ramp_start_0': 0.3, 'ramp': 1182.87, ':merging_start_0': 9.52, \
                'merging': 123.26, ':merging_end_0': 8.12, 'mainline_down': 339.28}


POS_MAINLINE_UP = 0
POS_MAINLINE_START = POS_MAINLINE_UP -EDGE_LEN[':mainline_start_0']
POS_MAINLINE_LOAD = POS_MAINLINE_START -EDGE_LEN['mainline_load']
POS_MERGING_START_1 = POS_MAINLINE_UP + EDGE_LEN['mainline_up']
POS_MERGING = POS_MERGING_START_1 + EDGE_LEN[':merging_start_1']
POS_RAMP_LOAD = POS_MERGING - EDGE_LEN[':merging_start_0'] - EDGE_LEN['ramp'] - EDGE_LEN[':ramp_start_0'] - EDGE_LEN['ramp_load']
POS_RAMP_START = POS_RAMP_LOAD + EDGE_LEN['ramp_load']
POS_RAMP = POS_RAMP_START + EDGE_LEN[':ramp_start_0']
POS_MERGING_START_0 = POS_RAMP + EDGE_LEN['ramp']
POS_MERGING_END = POS_MERGING + EDGE_LEN['merging']
POS_MAINLINE_DOWN = POS_MERGING_END + EDGE_LEN[':merging_end_0']

MILE_STONE = {0: POS_MAINLINE_LOAD, 1: POS_MAINLINE_START, 2: POS_MAINLINE_UP, 3: POS_MERGING_START_1, 
              4: POS_RAMP_LOAD, 5: POS_RAMP_START, 6: POS_RAMP, 7: POS_MERGING_START_0,
              8: POS_MERGING, 9: POS_MERGING_END, 10: POS_MAINLINE_DOWN}

EDGE_CODING = {'mainline_load': 0, ':mainline_start_0': 1, 'mainline_up': 2, ':merging_start_1': 3, 
               'ramp_load': 4, ':ramp_start_0': 5, 'ramp': 6, ':merging_start_0': 7,
               'merging': 8, ':merging_end_0': 9, 'mainline_down': 10}


LANE_CODING = {'mainline_load_0': 0, 'mainline_load_1': 1, 'mainline_load_2': 2,
               ':mainline_start_0_0': 0, ':mainline_start_0_1': 1, ':mainline_start_0_2': 2,
               'mainline_up_0': 0, 'mainline_up_1': 1, 'mainline_up_2': 2,
               ':merging_start_1_0': 0, ':merging_start_1_1': 1, ':merging_start_1_2': 2,
               'ramp_load_0': -1, 
               ':ramp_start_0_0': -1, 
               'ramp_0': -1, 
               ':merging_start_0_0': -1,
               'merging_0': -1, 'merging_1': 0, 'merging_2': 1, 'merging_3': 2,
               ':merging_end_0_0': -1, ':merging_end_0_1': 0, ':merging_end_0_2': 1, ':merging_end_0_3': 2,
               'mainline_down_0': 0, 'mainline_down_2': 1, 'mainline_down_4': 2}

induction_loop_mainline = [['mainline_eval_0_0', 'mainline_eval_0_1', 'mainline_eval_0_2'], 
                           ['mainline_eval_1_0', 'mainline_eval_1_1', 'mainline_eval_1_2'],
                           ['mainline_eval_2_0', 'mainline_eval_2_1', 'mainline_eval_2_2'],
                           ['mainline_eval_3_1', 'mainline_eval_3_2', 'mainline_eval_3_3'],
                           ['mainline_eval_4_1', 'mainline_eval_4_2', 'mainline_eval_4_3'],
                           ['mainline_eval_5_0', 'mainline_eval_5_1', 'mainline_eval_5_2'], 
                           ['mainline_eval_6_0', 'mainline_eval_6_1', 'mainline_eval_6_2']]
           


### MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
#training = False

## TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
#episode_render = False

# FIXED Q TARGETS HYPERPARAMETERS
#        'max_tau':          1000   #Tau is the C step where we sync our target network 
#Prio-learning factors
#        'PRIO_REPLAY_ALPHA': 0.6,
#        'BETA_START':        0.4,
#        'BETA_FRAMES':       1000000,