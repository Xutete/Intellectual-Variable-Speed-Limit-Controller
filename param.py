import numpy as np

PROJECT = "./SimulationProject/"

FONT_SETTING = {'family': 'Cambria', 'size': 12}

STEP_SIZE = 1

START_TIME, END_TIME = 600, 4200

PHASE_LEN = 4

MAIN_LOOP = [["mainline_eval_0_0", "mainline_eval_0_1", "mainline_eval_0_2"], 
            ["mainline_eval_1_0", "mainline_eval_1_1", "mainline_eval_1_2"],
            ["mainline_eval_2_0", "mainline_eval_2_1", "mainline_eval_2_2"],
            ["mainline_eval_3_1", "mainline_eval_3_2", "mainline_eval_3_3"],
            ["mainline_eval_4_1", "mainline_eval_4_2", "mainline_eval_4_3"],
            ["mainline_eval_5_0", "mainline_eval_5_1", "mainline_eval_5_2"], 
            ["mainline_eval_6_0", "mainline_eval_6_1", "mainline_eval_6_2"]]
RAMP_AREA = [["ramp_eval_0_0"]]
MERGING_AREA = [["ramp_eval_1_0"]]
EE_DETECTOR = [["area_eval_0_0"]] # Entry&Exit Detector


MERGING_LOOP = ["mainline_eval_3_0", "mainline_eval_3_1", "mainline_eval_3_2", "mainline_eval_3_3"] # Induction loop for measuring merging speed
QUEUE_AREA = ["ramp_eval_0_0", "ramp_eval_1_0"] # Lanearea for measuring queue length

RAMP_LOOP = ["ramp_eval_2_0"] # Induction loop for queue detection in ALINEA

RAMP_METER = "merging_start"

np.random.seed(42)

TEST_SEEDS = np.random.randint(1, 10**7, 20)

TRAIN_SEEDS = np.random.randint(1, 10**7, 10**4)

SPEED_WEIGHT, QUEUE_WEIGHT = 0.5, -0.01

BUFFER_SIZE = 10 ** 6 
INITIAL_BUFFER_SIZE = 5 * 10 ** 5

SYN = 10 ** 4

LOSS_INTERVAL = 200

BATCH_SIZE = 32

LEARNING_RATE = 0.00025

MAX_EPS, MIN_EPS, DECAY_RATE = 1.0, 0.1, 0.99


EDGE_LEN = {'mainline_load': 316.71, ':mainline_start_0': 0.3, 'mainline_up': 3868.42, ':merging_start_1': 8.03, 'ramp_load': 257.80, ':ramp_start_0': 0.3, 'ramp': 1181.87, ':merging_start_0': 8.91, 'merging': 125.2, ':merging_end_0': 8.20, 'mainline_down': 339.16}


POS_MAINLINE_LOAD = 0
POS_MAINLINE_START = POS_MAINLINE_LOAD + EDGE_LEN['mainline_load']
POS_MAINLINE_UP = POS_MAINLINE_START + EDGE_LEN[':mainline_start_0']
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
               'mainline_down_0': 0, 'mainline_down_1': 1, 'mainline_down_2': 2}