""""Module containing a fixed control car."""

from typing import Union, Iterable

import tensorflow as tf
import numpy as np

from interact_drive.car.car import Car
from interact_drive.world import CarWorld


class FixedControlCar(Car):
    """
    A car where the controls are fixed.
    """

    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable],
                 control: Union[np.array, tf.Tensor, Iterable],
                 color: str = 'gray',
                 opacity: float = 1.0, **kwargs):
        super().__init__(env, init_state, color, opacity, **kwargs)
        self.control = tf.constant(control, dtype=tf.float32)
        self.control_already_determined_for_current_step = True

    def step(self, dt):
        if self.debug:
            self.past_traj.append((self.state, self.control))
        self.state = self.dynamics_fn(self.state, self.control, dt)

    @tf.function
    def reward_fn(self, world_state, self_control):
        return 0

    def _get_next_control(self):
        return self.control
