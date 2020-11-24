"""Car classes used to unittest reward inference procedures."""
from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car import PlannerCar, LinearRewardCar
from interact_drive.world import CarWorld


class LinearTargetSpeedPlannerCar(LinearRewardCar, PlannerCar):
    """
    A car that has the following two features:
        - Velocity
        - Squared difference between its velocity and a "target speed"
    """
    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable],
                 weights: Union[tf.Tensor, tf.Variable, np.array, Iterable],
                 horizon: int, target_speed: float, friction: float = 0.2,
                 **kwargs):
        super().__init__(env, init_state, horizon=horizon, weights=weights,
                         friction=friction, **kwargs)
        self.target_speed_tf = tf.Variable(target_speed)

    @property
    def target_speed(self):
        return self.target_speed_tf.numpy()

    @target_speed.setter
    def target_speed(self, speed):
        self.target_speed_tf.assign(speed)

    @tf.function
    def features(self, world_state: Iterable[Union[tf.Tensor, tf.Variable]],
                 control: Union[tf.Tensor, tf.Variable]) -> tf.Tensor:
        features = []
        x = world_state[self.index]
        velocity = x[2]
        features.append(velocity)
        features.append((velocity - self.target_speed_tf) ** 2)
        return tf.stack(features, axis=-1)
