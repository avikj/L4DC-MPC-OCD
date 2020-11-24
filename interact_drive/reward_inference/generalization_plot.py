from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import seaborn as sns
import os
import sys
import random
from interact_drive.reward_inference import optimize_weights
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign
import itertools
import argparse
import pandas as pd

NUM_CMAES_EVALS = 85

file_dicts = {
    'finite_horizon': {
        (1, 1): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_1_opt_seed_1434742766_sigma_0.05.pkl',
        (1, 2): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_2_opt_seed_1846665217_sigma_0.05.pkl',
        (1, 3): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_3_opt_seed_2732669186_sigma_0.05.pkl',
        (1, 4): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_4_opt_seed_629181476_sigma_0.05.pkl',
        (1, 5): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_5_opt_seed_3845961838_sigma_0.05.pkl',
        (1, 6): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_6_opt_seed_4246928334_sigma_0.05.pkl',
        (1, 7): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_7_opt_seed_1216228145_sigma_0.05.pkl',
        (3, 1): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_1_opt_seed_3803522406_sigma_0.05.pkl',
        (3, 2): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_2_opt_seed_3414830645_sigma_0.05.pkl',
        (3, 3): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_3_opt_seed_537909881_sigma_0.05.pkl',
        (3, 4): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_4_opt_seed_404410072_sigma_0.05.pkl',
        (3, 5): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_5_opt_seed_1247211064_sigma_0.05.pkl',
        (3, 6): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_6_opt_seed_1096122360_sigma_0.05.pkl',
        (3, 7): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init_seed_7_opt_seed_3886655297_sigma_0.05.pkl',
        (5, 10): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_10_opt_seed_1721333845_sigma_0.05.pkl',
        (5, 1): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_1_opt_seed_2124184117_sigma_0.05.pkl',
        (5, 2): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_2_opt_seed_2750296751_sigma_0.05.pkl',
        (5, 3): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_3_opt_seed_2257270358_sigma_0.05.pkl',
        (5, 4): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_4_opt_seed_2277198133_sigma_0.05.pkl',
        (5, 5): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_5_opt_seed_43427611_sigma_0.05.pkl',
        (5, 6): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_6_opt_seed_1865573723_sigma_0.05.pkl',
        (5, 7): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_7_opt_seed_2048804620_sigma_0.05.pkl',
        (5, 8): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_8_opt_seed_3127109526_sigma_0.05.pkl',
        (5, 9): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__5_init_seed_9_opt_seed_564498324_sigma_0.05.pkl',
        (7, 10): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_10_opt_seed_356638632_sigma_0.05.pkl',
        (7, 1): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_1_opt_seed_898861217_sigma_0.05.pkl',
        (7, 2): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_2_opt_seed_5418685_sigma_0.05.pkl',
        (7, 3): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_3_opt_seed_3940351816_sigma_0.05.pkl',
        (7, 4): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_4_opt_seed_595027811_sigma_0.05.pkl',
        (7, 5): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_5_opt_seed_1834196677_sigma_0.05.pkl',
        (7, 6): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_6_opt_seed_1800540646_sigma_0.05.pkl',
        (7, 7): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_7_opt_seed_907618373_sigma_0.05.pkl',
        (7, 8): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_8_opt_seed_3835039258_sigma_0.05.pkl',
        (7, 9): 'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__7_init_seed_9_opt_seed_1207683396_sigma_0.05.pkl',
    },
    'replanning': {
        (1,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_1_opt_seed_1190916145_sigma_0.05.pkl',
        (1,2): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_2_opt_seed_2553789545_sigma_0.05.pkl',
        (1,3): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_3_opt_seed_946726374_sigma_0.05.pkl',
        (1,4): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_4_opt_seed_4003208099_sigma_0.05.pkl',
        (1,5): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_5_opt_seed_2092103804_sigma_0.05.pkl',
        (1,6): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_6_opt_seed_1905304730_sigma_0.05.pkl',
        (1,7): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_7_opt_seed_2671657731_sigma_0.05.pkl',
        (3,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init__seed_1_sigma_0.05.pkl',
        (3,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_1_opt_seed_2756028173_sigma_0.05.pkl',
        (3,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_1_opt_seed_3163346402_sigma_0.05.pkl',
        (3,2): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_2_opt_seed_776072588_sigma_0.05.pkl',
        (3,3): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_3_opt_seed_1867788114_sigma_0.05.pkl',
        (3,4): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_4_opt_seed_117634039_sigma_0.05.pkl',
        (3,5): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_5_opt_seed_3712927087_sigma_0.05.pkl',
        (3,6): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_6_opt_seed_4199044508_sigma_0.05.pkl',
        (3,7): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_7_opt_seed_4215107909_sigma_0.05.pkl',
        (5,10): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_10_opt_seed_487136777_sigma_0.05.pkl',
        (5,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_1_opt_seed_1775163370_sigma_0.05.pkl',
        (5,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_1_opt_seed_454987000_sigma_0.05.pkl',
        (5,2): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_2_opt_seed_2371686105_sigma_0.05.pkl',
        (5,3): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_3_opt_seed_2466516868_sigma_0.05.pkl',
        (5,4): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_4_opt_seed_1390102526_sigma_0.05.pkl',
        (5,5): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_5_opt_seed_3096487240_sigma_0.05.pkl',
        (5,6): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_6_opt_seed_812967300_sigma_0.05.pkl',
        (5,7): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_7_opt_seed_4077614066_sigma_0.05.pkl',
        (5,8): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_8_opt_seed_3864491032_sigma_0.05.pkl',
        (5,9): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_9_opt_seed_253430489_sigma_0.05.pkl',
        # (7,10): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_10_opt_seed_3513503694_sigma_0.05.pkl',
        # (7,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_162996332_sigma_0.05.pkl',
        (7,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_319611122_sigma_0.05.pkl',
        # (7,1): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_509800738_sigma_0.05.pkl',
        (7,2): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_2_opt_seed_2951468509_sigma_0.05.pkl',
        (7,3): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_3_opt_seed_3281406377_sigma_0.05.pkl',
        (7,4): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_4_opt_seed_2582331046_sigma_0.05.pkl',
        (7,5): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_5_opt_seed_3196603962_sigma_0.05.pkl',
        (7,6): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_6_opt_seed_4272125608_sigma_0.05.pkl',
        (7,7): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_7_opt_seed_300143191_sigma_0.05.pkl',
        (7,8): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_8_opt_seed_809033453_sigma_0.05.pkl',
        # (7,9): 'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_9_opt_seed_812376507_sigma_0.05.pkl',
    },
    'local_opt': {
        (1, 1): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_1_opt_seed_775562188_sigma_0.05.pkl',
        (1, 2): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_2_opt_seed_2618053792_sigma_0.05.pkl',
        (1, 3): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_3_opt_seed_974891352_sigma_0.05.pkl',
        (1, 4): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_4_opt_seed_511037338_sigma_0.05.pkl',
        (1, 5): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_5_opt_seed_2995848051_sigma_0.05.pkl',
        (1, 6): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_6_opt_seed_2905213039_sigma_0.05.pkl',
        (1, 7): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_7_opt_seed_3991352654_sigma_0.05.pkl',
        (3, 10): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_10_opt_seed_140975998_sigma_0.05.pkl',
        (3, 1): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_1_opt_seed_866805202_sigma_0.05.pkl',
        (3, 2): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_2_opt_seed_3530346049_sigma_0.05.pkl',
        (3, 3): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_3_opt_seed_2525870846_sigma_0.05.pkl',
        (3, 4): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_4_opt_seed_4081522144_sigma_0.05.pkl',
        (3, 5): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_5_opt_seed_1395296186_sigma_0.05.pkl',
        (3, 6): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_6_opt_seed_1062591525_sigma_0.05.pkl',
        (3, 7): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_7_opt_seed_4053917964_sigma_0.05.pkl',
        (3, 8): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_8_opt_seed_142287638_sigma_0.05.pkl',
        (3, 9): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__3_init_seed_9_opt_seed_3093197839_sigma_0.05.pkl',
        (5, 1): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_1_opt_seed_323424106_sigma_0.05.pkl',
        (5, 2): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_2_opt_seed_1122283449_sigma_0.05.pkl',
        (5, 3): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_3_opt_seed_2116509752_sigma_0.05.pkl',
        (5, 4): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_4_opt_seed_3044059000_sigma_0.05.pkl',
        (5, 5): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_5_opt_seed_3894639458_sigma_0.05.pkl',
        (5, 6): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_6_opt_seed_2980771891_sigma_0.05.pkl',
        (5, 7): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__5_init_seed_7_opt_seed_3719217169_sigma_0.05.pkl',
        (7, 10): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_10_opt_seed_2015880917_sigma_0.05.pkl',
        (7, 1): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_1_opt_seed_1227543455_sigma_0.05.pkl',
        (7, 2): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_2_opt_seed_3855063872_sigma_0.05.pkl',
        (7, 3): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_3_opt_seed_2803001119_sigma_0.05.pkl',
        (7, 4): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_4_opt_seed_859456155_sigma_0.05.pkl',
        (7, 5): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_5_opt_seed_1253922391_sigma_0.05.pkl',
        (7, 6): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_6_opt_seed_451879240_sigma_0.05.pkl',
        (7, 7): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_7_opt_seed_3930626616_sigma_0.05.pkl',
        (7, 8): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_8_opt_seed_3153788001_sigma_0.05.pkl',
        (7, 9): 'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__7_init_seed_9_opt_seed_1592528860_sigma_0.05.pkl',
    }
}

if __name__ == '__main__':

	"""parser = argparse.ArgumentParser()
	parser.add_argument('scenario', type=str, choices=['finite_horizon', 'replanning', 'local_opt'])
	args = parser.parse_args()"""

	matplotlib.rc('font',family='serif', serif=['Palatino'])
	sns.set_style('white')
	def set_style():
		sns.set(font='serif', font_scale=1.8, rc={'text.usetex':True})
	  	# Make the background white, and specify the
		# specific font family
		sns.set_style("white", {
			"font.family": "serif",
			"font.weight": "normal",
			"font.serif": ["Times", "Palatino", "serif"],
			'axes.facecolor': 'white',
			'lines.markeredgewidth': 1})
	set_style()

	quantiles = True



	def standard_error(a):
		print("computing standard error", np.std(a)/np.sqrt(len(a)))
		return np.std(a)/np.sqrt(len(a))

	training_sizes = [1,3,5,7]

	fig, axs = plt.subplots(1, 3)


	for scenario, ax in zip(['finite_horizon', 'local_opt', 'replanning'], axs):
		print(scenario)
		fns = [f for f in os.listdir() if f'{scenario}_gen_test_results_test_init_' in f and 'baseline' not in f]
		baseline_fns = [f for f in os.listdir() if f'{scenario}_gen_test_results_test_init_' in f and 'baseline' in f]



		all_inits_results = [pickle.load(open(f, 'rb')) for f in fns]
		test_inits_results = [d for d in all_inits_results if len(d) == (28 if scenario != 'local_opt' else 24)]
		true_weight_baseline_inits_results = [-pickle.load(open(f, 'rb')) for f in baseline_fns] # negative sign to convert reward to cost
		training_returns = []

		test_results_for_training_size = defaultdict(list)
		train_results_for_training_size = defaultdict(list)
		for training_size in training_sizes:
			for seed in [1,2,3,4,5,6,7] if scenario != 'local_opt' else [2,3,4,5,6,7]:
				train_results_for_training_size[training_size].append(
						-max([e[1] for e in pickle.load(open(file_dicts[scenario][(training_size, seed)], 'rb'))[:NUM_CMAES_EVALS]])/training_size
					)
					# negative sign to convert reward to cost
				test_results_for_training_size[training_size].append(
					-np.mean([results_for_init[(training_size, seed)] for results_for_init in test_inits_results]))
					

		for size in training_sizes:
			print("Size",size)
			print(test_results_for_training_size[size])

		# list of data points (true returns) for each x-value (training size)
		y = [test_results_for_training_size[size] for size in training_sizes] 

		print('means', [np.mean(a) for a in y], 'medians', [np.median(a) for a in y])

		def compute_error(y_vals):
			if quantiles:
				return np.abs(np.array([np.quantile(y_vals, [0.25, 0.75])-np.mean(y_vals) for size in training_sizes]).T)
			return np.array([standard_error(test_results_for_training_size[size]) for size in training_sizes])

		"""plt.xlabel('Training set size')
		plt.ylabel('Cost under true cost function')
		plt.errorbar(training_sizes, [np.mean(a) for a in y], yerr=compute_error(test_results_for_training_size[size]), label='Test cost')

		plt.errorbar(training_sizes, [np.mean(true_weight_baseline_inits_results)]*len(training_sizes), yerr=compute_error([true_weight_baseline_inits_results]*len(training_sizes)), color='xkcd:gray', label='Test cost when optimizing true weights')
		plt.errorbar(training_sizes, [np.mean(train_results_for_training_size[size]) for size in training_sizes], yerr=compute_error(train_results_for_training_size[size]), label='Train Cost')
		plt.legend()
		plt.title(f'{scenario}, use_quantiles_not_stderr={quantiles}')
		plt.show()"""


		test_xs = []
		train_xs = []
		test_ys = []
		train_ys = []
		baseline_ys = []
		baseline_xs = []
		for size in training_sizes:
		    print("Size", size)
		    print(test_results_for_training_size[size])
		    test_ys.extend(test_results_for_training_size[size])
		    train_ys.extend(train_results_for_training_size[size])
		    baseline_ys.extend(true_weight_baseline_inits_results)
		    test_xs.extend([size]*len(test_results_for_training_size[size]))
		    train_xs.extend([size]*len(train_results_for_training_size[size]))
		    baseline_xs.extend([size]*len(true_weight_baseline_inits_results))




		sns.lineplot(ax=ax, x=test_xs, y=test_ys, color='xkcd:dark orange', label=r"MPC with $C_{\textrm{CMA-ES}}'$ (test inits)" if scenario =='replanning' else None)
		sns.lineplot(ax=ax, x=train_xs, y=train_ys, color='xkcd:blue', label=r"MPC with $C_{\textrm{CMA-ES}}'$ (train inits)" if scenario =='replanning' else None)
		sns.lineplot(ax=ax, x=baseline_xs, y=baseline_ys, color='xkcd:gray', label='MPC with $C$' if scenario =='replanning' else None)

		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)

		ax.set_xlabel(r"$n_{{init}}$")
		ax.set_ylabel('True cost $C$ of rollout')

		scen_n = {"finite_horizon":1, "local_opt":2, "replanning":3}[scenario]
		ax.set_title(f'Scenario {scen_n}')

	plt.legend()

	plt.tight_layout()

	plt.show()
	plt.savefig(f'{scenario}_generalization_plot.png')
