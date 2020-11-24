"""Proof of concept - a planning car in a simple merging scenario.

In this scenario, a planning car wants to merge to the right lane. However,
there are two cars in that lane, and the planning car will need to merge between
the two cars.

We also save the result into merging.gif.
"""
import contextlib
from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.world import CarWorld, ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar, LinearRewardCar, PlannerCar
from interact_drive.math_utils import gaussian, smooth_bump, smooth_threshold


class ThreeLaneTestCar(LinearRewardCar, PlannerCar):
    def __init__(self, env: CarWorld, init_state, horizon: int,
                 weights: Union[tf.Tensor, tf.Variable, np.array], target_speed=1.,
                 color='orange', friction=0.2, opacity=1.0, planner_args=None, debug=False,
                 num_lanes=3,
                 **kwargs):
        super().__init__(env, init_state, horizon=horizon, weights=weights,
                         color=color, friction=friction, opacity=opacity,
                         planner_args=planner_args, debug=debug, **kwargs)
        self.target_speed = np.float32(target_speed)
        self.num_lanes = num_lanes

    @tf.function
    def features(self, state: Iterable[Union[tf.Tensor, tf.Variable]],
                 control: Union[tf.Tensor, tf.Variable]) -> tf.Tensor:
        """
        Features this car cares about are:
            - its forward velocity
            - its squared distance to each of three lanes
            - the minimum squared distance to any of the three lanes
            - a Gaussian collision detector
            - smooth threshold feature for fencdes

        Args:
            state: the state of the world.
            control: the controls of this car in that state.

        Returns:
            tf.Tensor: the four features this car cares about.
        """

        def bounded_squared_dist(target, bound, x):
            return tf.minimum((x - target) ** 2, bound)

        feats = []
        lane_dists = []

        car_state = state[self.index]
        velocity = car_state[2] * tf.sin(car_state[3])
        feats.append(bounded_squared_dist(self.target_speed, 4 * self.target_speed ** 2, velocity))

        for i, lane in enumerate(self.env.lanes):
            lane_i_dist = lane.dist2median(car_state) * 10  # AVIK: CHANGED SCALE OF LANE DIST FEATURES TO MATCH OTHERS
            lane_dists.append(lane_i_dist)
            feats.append(lane_i_dist)
        feats.append(tf.reduce_min(lane_dists, axis=0))

        collision_feats = []
        for i, other_car in enumerate(self.env.cars):
            other_state = state[i]
            # g = gaussian(other_state[0:2], 0.03)
            # collision_feat = g(car_state[0:2])*np.sqrt(2*np.pi*0.03**2) # normalize so max val is 1
            x_bump = smooth_bump(other_state[0] - 0.08, other_state[0] + 0.08)
            y_bump = smooth_bump(other_state[1] - 0.15, other_state[1] + 0.15)
            collision_feat = x_bump(car_state[0]) * y_bump(car_state[1])
            if i != self.index:
                collision_feats.append(collision_feat)

        feats.append(tf.reduce_max(collision_feats, axis=0))

        fences = (smooth_threshold(0.05*self.num_lanes, width=0.05)(car_state[0]) + smooth_threshold(0.05*self.num_lanes, width=0.05)(
            -car_state[0])) * abs(car_state[0])
        feats.append(fences)
        return tf.stack(feats, axis=-1)


def setup_world():
    world = ThreeLaneCarWorld(visualizer_args=dict(name="Merging", heatmap_show=True))

    our_car = ThreeLaneTestCar(world, np.array([0, -1.8, 0.8, np.pi / 2]),
                               horizon=5,
                               weights=np.array([-1, 0., 0., -10., -10., -10, -5]))
    other_car_1 = FixedVelocityCar(world, np.array([0.1, -1.8, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    other_car_2 = FixedVelocityCar(world, np.array([0.1, -1.3, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    world.add_cars([our_car, other_car_1, other_car_2])

    world.reset()
    return our_car, other_car_1, other_car_2, world


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

    our_car, other_car_1, other_car_2, world = setup_world()

    frames = []
    # ßworld.render()
    frames.append(world.render("rgb_array"))

    for i in range(15):
        print("velocity", our_car.state[2])
        world.step()
        world.render()
        frames.append(world.render("rgb_array"))

    from interact_drive.reward_inference import LocalCIOC
    cioc = LocalCIOC(our_car, initial_weights=np.ones((6,), dtype=np.float32))
    # END DELETE THIS;

    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.5).write_gif("merging2.gif", program="ffmpeg")


##
if __name__ == "__main__":
    main()
