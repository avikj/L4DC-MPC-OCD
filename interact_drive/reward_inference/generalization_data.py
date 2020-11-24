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
import multiprocessing
from multiprocessing import Pool
import argparse
from .generalization_plot import file_dicts

parser = argparse.ArgumentParser()
parser.add_argument('scenario', type=str, choices=['finite_horizon', 'replanning', 'local_opt'])
parser.add_argument('--baseline', action='store_true')
parser.add_argument('--training_results', action='store_true')
args = parser.parse_args()

NUM_CMAES_EVALS = 85

datas = {
    k: pickle.load(open(fn, 'rb'))[:NUM_CMAES_EVALS] for k, fn in file_dicts[args.scenario].items()
}
for k, v in datas.items():
    print("truncated data file",file_dicts[args.scenario][k], "has", len(v), 'entries')
    assert len(v) == NUM_CMAES_EVALS


env_config = optimize_weights.envs[args.scenario]

car, world, test_inits = env_config['make_env'](env_seeds=[2**32-i-1 for i in list(range(40))])

"""for seed in [2,3,4,5,6,7]:
    for test_init in test_inits[1:3]:
        print("EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
        for n_inits in [1,3,5,7]:
            print("n_inits used to select weight =",n_inits)
            chosen_weights = max(datas[(n_inits,seed)], key=lambda a:a[1])[0]
            results[tuple(test_init)][(n_inits, seed)] = bord.eval_weights_for_init(test_init, chosen_weights, render=False)
        print("FINISHED EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
    with open('replanning_gen_test_results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile)

for seed in [2,3,4,5,6,7]:
    for test_init in test_inits[3:6]:
        print("EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
        for n_inits in [1,3,5,7]:
            print("n_inits used to select weight =",n_inits)
            chosen_weights = max(datas[(n_inits,seed)], key=lambda a:a[1])[0]
            results[tuple(test_init)][(n_inits, seed)] = bord.eval_weights_for_init(test_init, chosen_weights, render=False)
        print("FINISHED EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
    with open('replanning_gen_test_results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile)


for seed in [2,3,4,5,6,7]:
    for test_init in test_inits[6:9]:
        print("EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
        for n_inits in [1,3,5,7]:
            print("n_inits used to select weight =",n_inits)
            chosen_weights = max(datas[(n_inits,seed)], key=lambda a:a[1])[0]
            results[tuple(test_init)][(n_inits, seed)] = bord.eval_weights_for_init(test_init, chosen_weights, render=False)
        print("FINISHED EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
    with open('replanning_gen_test_results.pkl', 'wb') as outfile:
        pickle.dump(results, outfile)"""


def evaluate_all_sizes_and_seeds_for_init(test_init__test_init_idx):
    print('scenario', args.scenario)
    car, world, _ = env_config['make_env'](env_seeds=[2**32-i-1 for i in list(range(40))])

    bord = BayesOptRewardDesign(world, car, 
                [], env_config['eval_horizon'], num_samples=env_config['num_eval_samples'], save_path=f'{args.scenario}_gen_test_log.pkl')

    test_init, test_init_idx = test_init__test_init_idx
    print("Evaluatig test init number", test_init_idx, ';  use_baseline =', args.baseline)
    
    if not args.baseline:
        test_result = {}
        for seed in [1,2,3,4,5,6,7] if args.scenario != 'local_opt' else [2,3,4,5,6,7]:
            print("EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
            for n_inits in [1,3,5,7]:
                print("n_inits used to select weight =",n_inits)
                chosen_weights = max(datas[(n_inits,seed)], key=lambda a:a[1])[0]
                test_result[(n_inits, seed)] = bord.eval_weights_for_init(test_init, chosen_weights, render=False)
            print("FINISHED EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
            with open(f'{args.scenario}_gen_test_results_test_init_{test_init_idx}.pkl', 'wb') as outfile:
                pickle.dump(test_result, outfile)
    else:
        test_result = {}
        chosen_weights = datas[list(datas.keys())[0]][0][0]
        print('EVALUATING WEIGHTS:',chosen_weights)
        print('RUNNING BASELINE EVEAL WITH WEIGHTS', chosen_weights)
        test_result = bord.eval_weights_for_init(test_init, chosen_weights, render=False)
        # print("FINISHED EVALUATING WEIGHTS ON test_init", test_init, "for seed", seed)
        with open(f'{args.scenario}_gen_test_results_test_init_{test_init_idx}_baseline.pkl', 'wb') as outfile:
            pickle.dump(test_result, outfile)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn') 
    with Pool(8) as p:
        print(p.map(evaluate_all_sizes_and_seeds_for_init, zip(test_inits[:8], range(8))))

    with Pool(8) as p:
        print(p.map(evaluate_all_sizes_and_seeds_for_init, zip(test_inits[8:16], range(8, 16))))

    with Pool(8) as p:
        print(p.map(evaluate_all_sizes_and_seeds_for_init, zip(test_inits[16:24], range(16, 24))))

    with Pool(8) as p:
        print(p.map(evaluate_all_sizes_and_seeds_for_init, zip(test_inits[24:32], range(24, 32))))
