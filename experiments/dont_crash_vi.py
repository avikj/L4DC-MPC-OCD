from merging import ThreeLaneTestCar
from dont_crash import setup_world, is_in_right_lane, reaches_target_speed, doesnt_collide
import numpy as np
from bayes_opt import BayesianOptimization
import pickle
from interact_drive.reward_inference.coarse_value_iteration import ValueIteration

def main():
    our_car, other_car, world = setup_world()
    ValueIteration(world, our_car, filename='dont_crash_values_-5_center.pkl')._run_value_iteration()

if __name__ == '__main__':
    main()







