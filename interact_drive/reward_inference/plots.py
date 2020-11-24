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

def standard_error(a):
    print("computing standard error", np.std(a)/np.sqrt(len(a)))
    return np.std(a)/np.sqrt(len(a))

inits_files = {
    'finite_horizon': [os.path.join('./finite_horizon_single_inits_random', f) for f in os.listdir('./finite_horizon_single_inits_random')], # sorted list of init files
    'local_opt': [f for f in os.listdir('.') if f.startswith('random_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]')], # sorted list of init files
    'replanning': [os.path.join('./random_replanning_single_inits', f) for f in os.listdir('./random_replanning_single_inits')]
}


cmaes_inits_files = {
    'local_opt': [os.path.join('./local_opt_cmaes_results_many', f) for f in os.listdir('./local_opt_cmaes_results_many')], # sorted list of init files
    'finite_horizon': [os.path.join('./finite_horizon_cmaes_results_many', f) for f in os.listdir('./finite_horizon_cmaes_results_many')], # sorted list of init files
    'replanning': [os.path.join('./replanning_cmaes_results_many', f) for f in os.listdir('./replanning_cmaes_results_many')], # sorted list of init files
}

"""single_cmaes_files = {
    'finite_horizon':'cmaes_results/cmaes_finite_horizon__designer_weights_[ -5. 0. 0. 0. -6. -50. -50.]__1_init__seed_1_sigma_0.2.pkl',
    'local_opt':'cmaes_results/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init__seed_1_sigma_0.2.pkl',
    'replanning':'cmaes_results/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init__seed_1_sigma_0.2.pkl'
}"""
single_cmaes_files = {
    # 'finite_horizon': ['cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__11_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__3_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[ 0.00395948 -0.8930889 0.76676498 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[ 0.02108118 -0.90441358 0.76493624 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[ 0.04562645 -0.90710203 0.74180907 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[ 0.04687965 -0.89904293 0.805356 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[ 0.07085891 -0.90340992 0.80742664 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.00522086 -0.89908118 0.83322118 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.01888388 -0.8873841 0.82800926 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.01906651 -0.9086438 0.77029901 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.02789485 -0.89092949 0.84900212 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.041299 -0.89813824 0.79764547 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__[-0.04675225 -0.89113145 0.80960917 1.57079633]_init__seed_1_sigma_0.05.pkl',] ,
    'local_opt': [#['cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08001094 -0.90681984 1.00123886 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08004084 -0.89808586 1.00089344 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08004329 -0.91420406 0.99026783 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08011665 -0.90882716 0.99414796 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08020427 -0.88617781 0.99445355 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08026644 -0.89816236 1.00554414 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08038121 -0.8747682 1.00467362 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08038293 -0.91728759 0.99504388 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08047276 -0.88185898 1.00818541 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08063469 -0.89627649 0.99960724 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__[-0.08071021 -0.88226289 1.00160295 1.57079633]_init__seed_1_sigma_0.05.pkl'],
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_1_opt_seed_775562188_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_2_opt_seed_2618053792_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_3_opt_seed_974891352_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_4_opt_seed_511037338_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_5_opt_seed_2995848051_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_6_opt_seed_2905213039_sigma_0.05.pkl',
        'final_cmaes_results/local_opt/cmaes_local_opt__designer_weights_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1_init_seed_7_opt_seed_3991352654_sigma_0.05.pkl',
    ],
    'replanning':  [# ['cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 0.00234398 -0.89808586 1.00089344 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 1.05405902e-03 -9.08827159e-01 9.94147960e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 1.97973863e-04 -8.86177805e-01 9.94453547e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[-1.39474257e-03 -8.81858978e-01 1.00818541e+00 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[-9.53325629e-04 -9.17287593e-01 9.95043875e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl']
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_1_opt_seed_1190916145_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_2_opt_seed_2553789545_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_3_opt_seed_946726374_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_4_opt_seed_4003208099_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_5_opt_seed_2092103804_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_6_opt_seed_1905304730_sigma_0.05.pkl',
        'final_cmaes_results/replanning/cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_7_opt_seed_2671657731_sigma_0.05.pkl',
    ],
    'finite_horizon':  [# ['cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 0.00234398 -0.89808586 1.00089344 1.57079633]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 1.05405902e-03 -9.08827159e-01 9.94147960e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[ 1.97973863e-04 -8.86177805e-01 9.94453547e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[-1.39474257e-03 -8.81858978e-01 1.00818541e+00 1.57079633e+00]_init__seed_1_sigma_0.05.pkl', 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__[-9.53325629e-04 -9.17287593e-01 9.95043875e-01 1.57079633e+00]_init__seed_1_sigma_0.05.pkl']
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_2_opt_seed_1846665217_sigma_0.05.pkl',
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_3_opt_seed_2732669186_sigma_0.05.pkl',
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_4_opt_seed_629181476_sigma_0.05.pkl',
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_5_opt_seed_3845961838_sigma_0.05.pkl',
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_6_opt_seed_4246928334_sigma_0.05.pkl',
        'final_cmaes_results/finite_horizon/cmaes_finite_horizon__designer_weights_[-0.07028325 0. 0. 0. -0.0843399 -0.7028325 -0.7028325 ]__1_init_seed_7_opt_seed_1216228145_sigma_0.05.pkl',
    ],

}

final_cmaes_results_replanning = {
    (1,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_1_opt_seed_1190916145_sigma_0.05.pkl',
    (1,2): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_2_opt_seed_2553789545_sigma_0.05.pkl',
    (1,3): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_3_opt_seed_946726374_sigma_0.05.pkl',
    (1,4): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_4_opt_seed_4003208099_sigma_0.05.pkl',
    (1,5): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_5_opt_seed_2092103804_sigma_0.05.pkl',
    (1,7): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_6_opt_seed_1905304730_sigma_0.05.pkl',
    (1,7): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__1_init_seed_7_opt_seed_2671657731_sigma_0.05.pkl',
    (3,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init__seed_1_sigma_0.05.pkl',
    (3,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_1_opt_seed_2756028173_sigma_0.05.pkl',
    (3,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_1_opt_seed_3163346402_sigma_0.05.pkl',
    (3,2): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_2_opt_seed_776072588_sigma_0.05.pkl',
    (3,3): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_3_opt_seed_1867788114_sigma_0.05.pkl',
    (3,4): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_4_opt_seed_117634039_sigma_0.05.pkl',
    (3,5): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_5_opt_seed_3712927087_sigma_0.05.pkl',
    (3,6): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_6_opt_seed_4199044508_sigma_0.05.pkl',
    (3,7): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__3_init_seed_7_opt_seed_4215107909_sigma_0.05.pkl',
    (5,10): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_10_opt_seed_487136777_sigma_0.05.pkl',
    (5,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_1_opt_seed_1775163370_sigma_0.05.pkl',
    (5,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_1_opt_seed_454987000_sigma_0.05.pkl',
    (5,2): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_2_opt_seed_2371686105_sigma_0.05.pkl',
    (5,3): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_3_opt_seed_2466516868_sigma_0.05.pkl',
    (5,4): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_4_opt_seed_1390102526_sigma_0.05.pkl',
    (5,5): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_5_opt_seed_3096487240_sigma_0.05.pkl',
    (5,6): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_6_opt_seed_812967300_sigma_0.05.pkl',
    (5,7): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_7_opt_seed_4077614066_sigma_0.05.pkl',
    (5,8): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_8_opt_seed_3864491032_sigma_0.05.pkl',
    (5,9): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__5_init_seed_9_opt_seed_253430489_sigma_0.05.pkl',
    (7,10): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_10_opt_seed_3513503694_sigma_0.05.pkl',
    (7,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_162996332_sigma_0.05.pkl',
    (7,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_319611122_sigma_0.05.pkl',
    (7,1): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_1_opt_seed_509800738_sigma_0.05.pkl',
    (7,2): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_2_opt_seed_2951468509_sigma_0.05.pkl',
    (7,3): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_3_opt_seed_3281406377_sigma_0.05.pkl',
    (7,4): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_4_opt_seed_2582331046_sigma_0.05.pkl',
    (7,5): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_5_opt_seed_3196603962_sigma_0.05.pkl',
    (7,6): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_6_opt_seed_4272125608_sigma_0.05.pkl',
    (7,7): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_7_opt_seed_300143191_sigma_0.05.pkl',
    (7,8): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_8_opt_seed_809033453_sigma_0.05.pkl',
    (7,9): 'cmaes_replanning__designer_weights_[-0.20555662 0. 0. -0.13703774 -0.6851887 -0.6851887 ]__7_init_seed_9_opt_seed_812376507_sigma_0.05.pkl',
}

def load_random_search_results(scenario='finite_horizon', num_weight_samples=400):
    # TODO maybe filter out the first true weight entry
    inits_results = []
    for fn in os.listdir('final_random_results'):
        if scenario in fn:
            inits_results.append(pickle.load(open('final_random_results/'+ fn, 'rb'))[:num_weight_samples])
            assert len(inits_results[-1]) == num_weight_samples, f'{fn} only has {len(inits_results[-1])} entries.'

    return inits_results

def load_cmaes_results(scenario='finite_horizon', num_weight_samples=35):
    # TODO maybe filter out the first true weight entry
    print("Loading CMAES results from file", cmaes_inits_files[scenario])
    inits_results = []
    for fn in cmaes_inits_files[scenario]:
        inits_results.append(pickle.load(open(fn, 'rb'))[:num_weight_samples])
        assert len(inits_results[-1]) == num_weight_samples, f'{fn} only has {len(inits_results[-1])} entries.'

    return inits_results

def load_single_init_cmaes_results(scenario='finite_horizon', num_weight_samples=90):
    # TODO maybe filter out the first true weight entry
    print("Loading CMAES results from file", single_cmaes_files[scenario])
    inits_results = []
    for fn in single_cmaes_files[scenario]:
        inits_results.append(pickle.load(open(fn, 'rb'))[:num_weight_samples])
        assert len(inits_results[-1]) == num_weight_samples, f'{fn} only has {len(inits_results[-1])} entries.'

    return inits_results

def get_test_inits(scenario, n, seed):
    np.random.seed(seed)
    car, world = optimize_weights.envs[scenario]['make_env']()
    lo, hi = optimize_weights.envs[scenario]['init_offset_range']
    lo, hi = np.array(lo), np.array(hi)
    return [car.state+lo+(hi-lo)*np.random.random() for _ in range(n)]

def get_bar_plot_data(inits_results):
    num_weight_samples = len(inits_results[0])
    assert all([len(r) == num_weight_samples for r in inits_results])

    baseline_rewards = [results[0][1] for results in inits_results]
    print('baseline_rewards', baseline_rewards)
    print('baseline_rewards sum', sum(baseline_rewards))

    # find best training reward for each individual init (individual maxes)
    best_init_training_rewards = [max([r for w, r in results]) for results in inits_results]
    print('best training rewards per init', best_init_training_rewards)
    print('sum', sum(best_init_training_rewards))
    # find best training reward across inits (max of sum)
    best_weight_index_across_inits = max(range(num_weight_samples), key=lambda w_i: sum([results[w_i][1] for results in inits_results]))
    best_training_reward_across_inits = [results[best_weight_index_across_inits][1] for results in inits_results]

    print("Best reward weight across training examples: ", inits_results[0][best_weight_index_across_inits][0])
    print('rewards from best reward function across weights', best_training_reward_across_inits)
    print('sum', sum(best_training_reward_across_inits))


    return baseline_rewards, best_init_training_rewards, best_training_reward_across_inits # third output is the weights we'll use in generalization tests




"""
NUM_WEIGHT_SAMPLES_PER_TRAINING_SIZE = 20 # 20
NUM_TEST_INITS = 15
def plot_generalization(scenario='finite_horizon', num_weight_samples=200, seed=1):

    save_filename = f'{scenario}_generalization_plot_results_saved_RANDOMIZED_seed_{seed}.pkl'
    print('save filename:', save_filename)
    np.random.seed(seed)
    plt.xticks(np.arange(0, 20))

    total_inits = 3 if scenario == 'replanning' else 11

    def which_inits(n):
        # randomly return n ints on [1, total_inits]
        r = list(np.random.choice(np.arange(1, total_inits+1), size=n))
        return r

    print("SCENARIO", scenario)
    # max_eleven = {1: [8], 3: [1, 3, 7], 5: [1, 5, 6, 8, 9],7: [1,2,3,5,6,7,9], 9: [1,2,3,4,6,7,8,10,11], 11:  [1,2,3,4,5,6,7,8,9,10,11]} # accidentally 1-indexed and too lazy to change
    # max_three = {1: [3], 2: [2,3], 3: [1,2,3]}# {1: [2], 3: [1,2,3]}
    # which_inits = {
    #   'finite_horizon': max_eleven,
    #    'local_opt': max_eleven,
    #    'replanning': max_three
    #}[scenario]
    print("WHICH INITS", which_inits)
    inits_results = load_random_search_results(scenario, num_weight_samples)
    test_baseline = [results[0][1] for results in inits_results]
    std_test_baseline = standard_error(test_baseline)
    mean_test_baseline = np.mean(test_baseline)
    if  os.path.isfile(save_filename):
        print("LOADING SAVED EVAL RESULTS FROM PICKLE")
        training_sizes, means, standard_errors = pickle.load(open(save_filename, 'rb'))
    else:
        print("NO SAVED EVAL RESULTS; SAMPLING TEST EVALS")
        # def eval_training_with_n(n, reps=500):
        best_weight_indices_for_training_size = {}
        weight_indices_to_test = []
        training_sizes = [1, 3] if scenario == 'replanning' else [1,3,5,7,9,11]
        for n in training_sizes:
            print("Training Size:",n)
            assert n % 2 == 1 # must include center init
            # select n results evenly spaced; assert 
            best_weight_indices_for_training_size[n] = []
            for _ in range(NUM_WEIGHT_SAMPLES_PER_TRAINING_SIZE):
                training_inits_results = [inits_results[i-1] for i in which_inits(n)]  # see 1-index comment^
                best_weight_indices_for_training_size[n].append(max(range(num_weight_samples), key=lambda w_i: sum([results[w_i][1] for results in training_inits_results])))

            print(f'best_weight_indices_for_training_size[{n}] = {best_weight_indices_for_training_size[n]}')
            weight_indices_to_test.extend(best_weight_indices_for_training_size[n])

            #print(f'Training with {n} inits.')
            #print("Mean training baseline: ", np.mean([results[0][1] for results in training_inits_results]))
            #print('Mean training rewards:', np.mean([results[best_weight_index_across_inits][1] for results in training_inits_results]))
            #print("Best reward weight across training examples: ", training_inits_results[0][best_weight_index_across_inits][0])
            #print("Mean test baselines", mean_test_baseline)
        test_evals_for_weight_indices = {} # maps weight index to list of test results
        env_config = optimize_weights.envs[scenario]
        car, world = env_config['make_env']()
        bord = BayesOptRewardDesign(world, car, 
            [], env_config['eval_horizon'], num_samples=env_config['num_eval_samples'], save_path='finite_horizon_test_results.pkl')
        print("weights to test:", weight_indices_to_test)
        for wi in set(weight_indices_to_test):
            test_evals_for_weight_indices[wi] = []
            for init in get_test_inits(scenario, n=NUM_TEST_INITS, seed=99999):
                print(f"Evaluating weights {inits_results[0][wi][0]} at init {init}")
                test_evals_for_weight_indices[wi].append(bord.eval_weights_for_init(init, inits_results[0][wi][0], render=False))

        means, standard_errors = [], []
        for n in training_sizes:
            best_weight_indices_across_inits = best_weight_indices_for_training_size[n] 
            results_from_weight_indices = list(itertools.chain.from_iterable(
                [test_evals_for_weight_indices[wi] for wi in best_weight_indices_across_inits])) # concatenate different samples for each training size

            means.append(np.mean(results_from_weight_indices))
            standard_errors.append(standard_error(results_from_weight_indices))
            print(n, means[-1], standard_errors[-1])
        with open(f'{scenario}_generalization_plot_results_saved.pkl', 'wb') as file:
            pickle.dump((training_sizes, means, standard_errors), file)
                # print("\n")
            # return np.mean(mean_test_rewards), (np.min(mean_test_rewards), np.max(mean_test_rewards))



    # plt.title('Higher average test reward is achieved by a reward function which is selected according to more training examples.')
    plt.title(scenario.replace('_', ' '))
    plt.ylabel('Value of $C^*$ when MPC optimizes $C\'$')
    plt.xlabel('Number of initializations used to select $C\'$')

    plt.errorbar(training_sizes, r_to_c(means), yerr=standard_errors, elinewidth=2, capsize=3, label='MPC optimizes chosen $C\'$', color='xkcd:dark orange')
    plt.errorbar(training_sizes, r_to_c([mean_test_baseline]*len(training_sizes)), yerr=[std_test_baseline]*len(training_sizes), label='MPC optimizes $C^*$', color='xkcd:gray')
    plt.legend()

    plt.show()
    plt.savefig(f'{scenario}_generalization_plot.png')
    plt.clf()
"""

def cmaes_bar_plot_data(scenarios, num_weight_samples):
    for scenario, nws in zip(scenarios, num_weight_samples):
        data = pickle.load(open(single_cmaes_files[scenario], 'rb'))[:nws]
        assert len(data) == nws
        yield max([r for w, r in data])

def r_to_c(arr):
    return [-a for a in arr]

def make_bar_plots(scenarios=['finite_horizon', 'local_opt', 'replanning'], num_weight_samples=[200,200,200]):
    true_returns, true_returns_yerr,random_returns, random_returns_yerr = [], [], [], []
    cmaes_returns, cmaes_returns_yerr =  [], []

    for scenario, nws in zip(scenarios, num_weight_samples):
        _, improvements, _ = get_bar_plot_data(load_random_search_results(scenario=scenario, num_weight_samples=400))
        baselines, cmaes_improvements, _ = get_bar_plot_data(load_single_init_cmaes_results(scenario=scenario))
        labels = ['Scenario 1', 'Scenario 2', 'Scenario 3']# ['Finite horizon\nplanning failure\nscenario', 'Locally optimal\nplanning failure\nscenario', 'Planning for\nreplanning failure']
        true_returns.append(np.mean(baselines)) 
        true_returns_yerr.append(standard_error(baselines))
        random_returns.append(np.mean(improvements)) 
        random_returns_yerr.append(standard_error(improvements))
        cmaes_returns.append(np.mean(cmaes_improvements)) 
        cmaes_returns_yerr.append(standard_error(cmaes_improvements)) 


    print("True returns:", true_returns)
    print("Cmaes returns:", cmaes_returns)


    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, r_to_c(true_returns), width, yerr=true_returns_yerr, label=r'MPC with $C$', color='xkcd:gray')
    rects2 = ax.bar(x, r_to_c(random_returns), width, yerr=random_returns_yerr, label=r"MPC with $C_{\small{\textrm{Random Search}}}'$", color='xkcd:dark orange')
    rects3 = ax.bar(x + width, r_to_c(cmaes_returns), width, yerr=cmaes_returns_yerr, label=r"MPC with $C_{\small{\textrm{CMA-ES}}}'$", color='xkcd:orange')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('True cost $C$ of rollout')

    # ax.set_title('True reward when MPC optimizes true reward function $R^*$ vs. best\ntrue reward when MPC optimizes another reward function $R\'$')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='center left')


    def autolabel(rects, num, move_label=False):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = np.around(rect.get_height(), decimals=3)
            xytext = (40, -7) if move_label and rect == rects[0] else (25  , 0) if rect==rects[1] else (5  , 10)
            if num == 2 and rect == rects[0]:
                xytext = (5,0)
            print(xytext)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=xytext,  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    #utolabel(rects1, 1)
    #autolabel(rects2, 2)
    #autolabel(rects3, 3, move_label=True) 

    fig.tight_layout()

    plt.show()
    plt.savefig(f'{scenarios}_bar_plot.png')
    plt.clf()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('scenario', choices=['finite_horizon', 'local_opt', 'replanning', 'bar_plot'])
    args = parser.parse_args()

    matplotlib.rc('font',family='serif', serif=['Palatino'])
    sns.set_style('white')
    def set_style():
        sns.set(font='serif', font_scale=2.1, rc={'text.usetex':True})
       # Make the background white, and specify the
        # specific font family
        sns.set_style("white", {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            'axes.facecolor': 'white',
            'lines.markeredgewidth': 1})

    set_style()
    # plot_generalization()
    if args.scenario == 'bar_plot':
        make_bar_plots()
    else:
        plot_generalization(scenario=args.scenario, num_weight_samples=200)
        print(f"PLOTTED GENERALIZATION for {args.scenario}\n\n")
