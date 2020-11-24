"""Contains utility classes used to test the planner classes."""

from typing import Iterable, Union

import tensorflow as tf
import numpy as np

from interact_drive.car import PlannerCar
from interact_drive.world import CarWorld


class TargetSpeedPlannerCar(PlannerCar):
    """
    Simple reward maximizer car where the reward is simply the negative squared
    difference between a target speed and the car's current speed.

    >>> init_state = np.array([0., 0., 1., np.pi / 2], dtype=np.float32)
    >>> car = TargetSpeedPlannerCar(None, init_state, 5,
    ...                             target_speed=0., friction=0.)
    >>> car.reward_fn([car.state], tf.constant([0., 0.])).numpy()
    -1.0
    >>> car.reward_fn([np.array([0., 0., 0., np.pi/2], dtype=np.float32)],
    ...               tf.constant([0., 0.]),).numpy()
    0.0
    >>> car.reward_fn([np.array([0., 0., 2., np.pi/2], dtype=np.float32)],
    ...               tf.constant([0., 0.])).numpy()
    -4.0

    """

    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable],
                 horizon: int, target_speed: float, friction: float = 0.2):
        super().__init__(env, init_state, horizon, friction=friction)
        self.target_speed_tf = tf.Variable(target_speed)

    @property
    def target_speed(self):
        return self.target_speed_tf.numpy()

    @target_speed.setter
    def target_speed(self, speed):
        self.target_speed_tf.assign(speed)

    @tf.function
    def reward_fn(self, world_state, self_control):
        r = 0

        x = world_state[self.index]
        velocity = x[2]
        r -= (velocity - self.target_speed_tf) ** 2

        return r
