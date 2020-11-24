"""OpenAI gym wrappers for CarWorlds."""
from typing import Dict, Iterable, Tuple, Union

import gym
from gym import spaces
import numpy as np
import tensorflow as tf

from interact_drive.car import Car
from interact_drive.world import CarWorld


class CarEnv(gym.Env):
    """
    Serves as a OpenAI gym wrapper for a CarWorld, where the state is the
    raw state of the environment.
    """

    def __init__(self, car_world: CarWorld, main_car_index: int = 0):
        self.car_world = car_world
        # not sure what the observation space should be
        self.observation_space = spaces.Box(low=-100, high=-100,
                                            shape=(len(self.car_world.cars), 4))
        self.action_space = spaces.Box(low=np.array([-5, np.pi]),
                                       high=np.array([5, np.pi]))
        self.main_car_index = main_car_index
        super().__init__()

    def get_state_from_car_state(self, state: Iterable[tf.Tensor]):
        """Converts a CarWorld.state (a list of tf.Tensors) into a np array."""
        state_np = np.stack([car_state.numpy() for car_state in state], axis=0)
        return state_np

    def reset(self):
        self.car_world.reset()
        state = self.car_world.state
        return self.get_state_from_car_state(state)

    def render(self, mode='human') -> Union[None, np.array, str]:
        """
        Renders the environment by calling the render() method of the
        wrapped CarWorld.
        """
        return self.car_world.render(mode=mode)

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict]:
        """
        Steps the environment forward by one timestep. We

        Args:
            action: an np.array of shape (2,), of the form
                        [acc, angle_vel],
                    where:
                        - acc: the instantaneous change in velocity
                                (in units/second^2)
                        - angle_vel: the angular velocity of the car
        Returns:
            state: the state of all the cars in the wrapped environment.
            reward: the reward accrued by the main car in this timestep
            done: whether or not the environment has terminated.
            info: a dictionary of helpful information. In this case, the
                    dictionary contains:
                    past_state (List[tf.Tensor]): the last state of the CarWorld
                    controls (List[tf.Tensor]): the current controls of all cars

        """
        main_car: Car = self.car_world.cars[self.main_car_index]
        main_car.set_next_control(action)

        past_state, controls, state = self.car_world.step()

        state_np = self.get_state_from_car_state(state)
        reward = main_car.reward_fn(state, controls[self.main_car_index])
        info = dict(past_state=past_state, controls=controls)
        done = False
        return state_np, reward.numpy(), done, info


# class FeaturizedCarEnv(CarEnv):
#     """
#     Implements a feature-based gym wrapper for a CarWorld, where the state is
#     the features of a LinearRewardCar.
#     """
#
#     def __init__(self, car_world: CarWorld, *args, **kwargs):
#         super.__init__(car_world=car_world, *args, **kwargs)
#         main_car: Car = self.car_world.cars[self.main_car_index]
#         if not isinstance(main_car, LinearRewardCar):
#             raise ValueError("The main car must be a LinearRewardCar.")
