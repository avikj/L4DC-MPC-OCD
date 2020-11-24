"""Utility functions for the car simulation."""

from typing import Union, Iterable

import numpy as np
import tensorflow as tf


@tf.function
def car_dynamics_step(x, y, v, angle, acc, ang_vel, dt, friction):
    acc = tf.maximum(tf.minimum(acc, 4.), -2*4.)
    ang_vel = tf.maximum(tf.minimum(ang_vel, 4), -4.)

    total_acc = acc - friction * v ** 2
    distance_travelled = v * dt + 0.5 * total_acc * (dt ** 2)
    new_x = x + tf.cos(angle) * distance_travelled
    new_y = y + tf.sin(angle) * distance_travelled
    new_v = v + total_acc * dt
    new_angle = angle + ang_vel * dt

    return new_x, new_y, new_v, new_angle


@tf.function
def batched_next_car_state(state: Union[np.array, tf.Tensor],
                           control: Union[np.array, tf.Tensor],
                           dt: float,
                           friction: float = 0.) -> tf.Tensor:
    """
    Computes the next car state  after applying the given controls.

    We assume that the direction the car travels is fixed until the end of the
        interval, but the speed can change during the interval.

    Args:
        state: An np.array or tf.Tensor of shape (batch_size, 4), where each row
                is of the form [x, y, v, angle],
                where:
                    - x: the x coordinate
                    - y: the y coordinate
                    - v: the instantaneous velocity (in units/second)
                    - angle: the angle of the car in radians (0 is due west)
        control: An np.array or tf.Tensor of shape (batch_size, 2), where
                each row is of the form [acc, angle_vel]
                where:
                    - acc: instantaneous change in velocity (in units/second^2)
                    - angle_vel: the angular velocity of the car

        dt: the unit of time to step forward for.
        friction: per timestep decrease in velocity due to friction (in 1/units)

    Returns:
        next_state: a tf.Tensor of shape (batch_size, 4), with rows of the form
                    [x', y', v', angle']
                representing the state after one tick of the simulation.
    """
    if len(state.shape) != 2 or state.shape[1] != 4:
        raise ValueError(
            "Input state has incorrect length {}".format(control.shape))
    if len(control.shape) != 2 or control.shape[1] != 2:
        raise ValueError(
            "Input control has incorrect shape".format(control.shape))

    x, y, v, angle = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    acc, ang_vel = control[:, 0], control[:, 1]

    new_x, new_y, new_v, new_angle = car_dynamics_step(x, y, v, angle, acc,
                                                       ang_vel, dt, friction)

    return tf.stack([new_x, new_y, new_v, new_angle], axis=-1)


@tf.function
def next_car_state(state: Union[np.array, tf.Tensor],
                   control: Union[np.array, tf.Tensor, Iterable],
                   dt: float,
                   friction: float = 0.):
    """
    Computes the next car state  after applying the given controls.

    We assume that the direction the car travels is fixed until the end of the
        interval, but the speed can change during the interval.

    Args:
        state: An np.array or tf.Tensor of shape (4,), of the form
                    [x, y, v, angle],
                where:
                    - x: the x coordinate
                    - y: the y coordinate
                    - v: the instantaneous velocity (in units/second)
                    - angle: the angle of the car in radians
                            (where 0 is due west)
        control: An np.array or tf.Tensor of shape (2,), of the form
                    [acc, angle_vel],
                where:
                    - acc: the instantaneous change in velocity
                            (in units/second^2)
                    - angle_vel: the angular velocity of the car

        dt: the unit of time to step forward for.
        friction: the per timestep decrease in velocity due to friction
                    (in 1/units)

    Returns:
        next_state: a tf.Tensor of shape (4,) of the form
                    [x', y', v', angle']
                representing the state after one tick of the simulation.
    """

    if state.shape[0] != 4:
        raise ValueError(
            "Input state has incorrect length {}".format(len(state)))
    if control.shape[0] != 2:
        raise ValueError(
            "Input control has incorrect length {}".format(len(control)))

    x, y, v, angle = state[0], state[1], state[2], state[3]
    acc, ang_vel = control[0], control[1]

    new_x, new_y, new_v, new_angle = car_dynamics_step(x, y, v, angle, acc,
                                                       ang_vel, dt, friction)

    return tf.stack([new_x, new_y, new_v, new_angle], axis=-1)


class CarState(object):
    def __init__(self, numpy_or_tf_state):
        """
        Args:
            numpy_or_tf_state: array-like object (e.g. a tf.Tensor or np.array)
                    representing the state of the car.

                    This should be a rank 1 tensor of the form [x, y, v, angle]:
                        - x: the x coordinate
                        - y: the y coordinate
                        - v: the instantaneous velocity (in units/second)
                        - angle: the angle of the car in radians
                                (where 0 is due west)
        """
        self.state = tf.Variable(numpy_or_tf_state, name='state',
                                 dtype=tf.float32)
        self._check_state_is_valid()

    def _check_state_is_valid(self):
        if self.state.shape[0] != 4:
            raise ValueError(
                "Input state has incorrect shape {}".format(self.state.shape))

    @property
    def x(self):
        return self.state[0]

    @property
    def y(self):
        return self.state[1]

    @property
    def v(self):
        return self.state[2]

    @property
    def angle(self):
        return self.state[3]

    def assign(self, value):
        return self.state.assign(value)

    def assign_add(self, delta):
        return self.state.assign_add(delta)

    def assign_sub(self, delta):
        return self.state.assign_sub(delta)

    def numpy(self):
        return self.state.numpy()

    def value(self):
        return self.state.value()


# The following classes and methods are in development and not currently used.

class BatchedCarState(CarState):
    def __init__(self, numpy_or_tf_state):
        """
        Args:
            numpy_or_tf_state: array-like object (e.g. a tf.Tensor or np.array)
                    representing the state of the car.

                    This should be a rank 2 tensor, where rows are of the form
                                [x, y, v, angle]:
                        - x: the x coordinate
                        - y: the y coordinate
                        - v: the instantaneous velocity (in units/second)
                        - angle: the angle of the car in radians
                                (where 0 is due west)
        """
        super().__init__(numpy_or_tf_state)

    def _check_state_is_valid(self):
        if len(self.state.shape) != 2 or self.state.shape[1] != 4:
            raise ValueError(
                "Input state has inccorrect shape {}".format(self.state.shape))

    @property
    def x(self):
        return self.state[:, 0]

    @property
    def y(self):
        return self.state[:, 1]

    @property
    def v(self):
        return self.state[:, 2]

    @property
    def angle(self):
        return self.state[:, 3]


class CarControl(object):
    def __init__(self, numpy_or_tf_controls):
        """
        Args:
            numpy_or_tf_controls: array-like object (e.g. tf.Tensor or np.array)
                    representing the controls to be applied to a car.

                    This should be a rank 1 tensor of the form [acc, ang_vel]:
                        - acc: the instantaneous change in velocity
                        - ang_vel: the angular velocity of the car
        """
        self.control = tf.Variable(numpy_or_tf_controls, name="control",
                                   dtype=tf.float32)
        self._check_control_is_valid()

    def _check_control_is_valid(self):
        if self.control.shape[0] != 2:
            raise ValueError(
                "Input control has incorrect shape {}".format(
                    self.control.shape))

    @property
    def acc(self):
        return self.control[0]

    @property
    def ang_vel(self):
        return self.control[1]

    def assign(self, value):
        return self.control.assign(value)

    def assign_add(self, delta):
        return self.control.assign_add(delta)

    def assign_sub(self, delta):
        return self.control.assign_sub(delta)

    def numpy(self):
        return self.control.numpy()

    def value(self):
        return self.control.value()


class BatchedCarControl(CarControl):
    def __init__(self, numpy_or_tf_controls):
        """
        Args:
            numpy_or_tf_controls: array-like object (e.g. tf.Tensor or np.array)
                    representing the controls to be applied to a car.

                    This should be a rank 2 tensor with rows of the form
                        [acc, ang_vel], where:
                        - acc: the instantaneous change in velocity
                        - ang_vel: the angular velocity of the car
        Args:
            numpy_or_tf_controls:
        """
        super().__init__(numpy_or_tf_controls)

    def _check_control_is_valid(self):
        if len(self.control.shape) != 2 or self.control.shape[1] != 2:
            raise ValueError(
                "Input state has incorrect shape {}".format(self.control.shape))

    @property
    def acc(self):
        return self.control[:, 0]

    @property
    def ang_vel(self):
        return self.control[:, 1]


def new_next_car_state(state: CarState, control: CarControl, dt: float,
                       friction: float = 0., inplace: bool = False) -> CarState:
    """
    Computes the next CarState from the given state.

    We follow the dynamics function used by Sadigh et al 2016.

    Note: control is batched, then state must also be batched.
    """

    new_state_value = tf.stack(
        car_dynamics_step(state.x, state.y, state.v, state.angle, control.acc,
                          control.ang_vel, dt, friction), axis=-1)

    if inplace:
        state.assign(new_state_value)
        return state

    if isinstance(state, BatchedCarState):
        return BatchedCarState(new_state_value)

    return CarState(new_state_value)


def get_dynamics_fn(friction):
    @tf.function
    def next_state(state, control, dt):
        return next_car_state(state, control, dt, friction)

    return next_state
