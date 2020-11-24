"""Proof of concept - a planning car in a simple merging scenario.

In this scenario, a planning car wants to achieve a target speed but 
is behind a slower car. Its optimal behavior is to switch lanes.

We also save the result into merging.gif.
"""
import contextlib
from typing import Iterable, Union 

import numpy as np
import tensorflow as tf

from interact_drive.world import ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar, LinearRewardCar, PlannerCar
from interact_drive.math_utils import gaussian
from merging import ThreeLaneTestCar

def setup_world(): 
    world = ThreeLaneCarWorld(visualizer_args=dict(name="don't crash", follow_main_car=True))

    our_car = ThreeLaneTestCar(world, np.array([0.01, -0.5, 0.8, np.pi / 2]),
                               horizon=5,
                               weights=np.array([-1, 0, -5, 0, 0, -50, -5], dtype=np.float32),
                               target_speed=1.2, planner_args={'n_iter':300})# , -10]))
    other_car = FixedVelocityCar(world, np.array([0, -0.1, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    our_car.debug = True
    other_car.debug=True
    world.add_cars([our_car, other_car])

    world.reset()
    return our_car, other_car, world
##
def main():
    """
    Runs a planning car in a merging scenario for 15 steps and visualizes it.

    The weights of our planning car should cause it to merge into the right lane
    between the two other cars.
    """
    # from sys import platform as sys_pf
    # if sys_pf == 'darwin':
    #     import matplotlib
    #     matplotlib.use("MacOSX")
    # from matplotlib import pyplot as plt
    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip

    our_car, other_car, world = setup_world()
    # our_car.weights = np.array([-8.0823083e+10 ,-5.3606113e+10, -3.8944246e+09,  1.9811942e+10,
    # -4.6828495e+10 ,-9.9395653e+11 ,-6.6240067e+09], dtype=np.float32)
    our_car.weights /= np.linalg.norm(our_car.weights)
    our_car.weights *= 50
    print('NORMED WEIGHTS', our_car.weights)
    our_car.initialize_planner(our_car.planner_args)
    from interact_drive.reward_inference.utils import evaluate_weights
    print("Reward from optimizing true weights:", evaluate_weights(our_car, our_car.weights, world))
    print("Reward from optimizing weights without center:", evaluate_weights(our_car, np.array([-1, 0, 0, 0, 0, -50, -5], dtype=np.float32), world))

    frames = []
    score = 0
    """for init_x in [0.0, 0.2/3]:
        our_car.init_state = np.array([init_x, our_car.init_state[1], our_car.init_state[2], our_car.init_state[3]], dtype=np.float32)
        world.reset()
        world.render()
        reward = 0
        frames.append(world.render("rgb_array"))
        for i in range(15):
            print("velocity", our_car.state[2])
            print("world.cars", world.cars)
            past_state, ctrl, state = world.step()
            reward += our_car.reward_fn(past_state, ctrl[0])
            world.render()
            frames.append(world.render("rgb_array"))
        print("Reward from traj:",reward)


    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.8).write_gif("dont_crash_lbfgs_reward_center_5.gif", program="ffmpeg")"""


def is_in_right_lane(car):
    return (abs(car.state[0]-0.1) < 0.02).numpy()

def is_in_left_lane(car):
    return (abs(car.state[0]+0.1) < 0.02).numpy()

def reaches_target_speed(car):
    return (abs(car.state[2]-car.target_speed) < 0.01).numpy()

def doesnt_collide(car, other_car):
    print("in doesnt_collide")
    print("past-traj:",[s for s, u in car.past_traj])
    print("other-traj:",[s for s, u in other_car.past_traj])
    for (state, ctrl), (other_state, other_ctrl) in zip(car.past_traj, other_car.past_traj):
        print(state.numpy(), other_state.numpy());
        if (abs(state[0]-other_state[0]) < 0.07 and abs(state[1]-other_state[1]) < 0.15).numpy():
            return False
    return True

##
if __name__ == "__main__":
    main()
