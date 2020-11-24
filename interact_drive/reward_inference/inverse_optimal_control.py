"""Base class for reward inference algorithms."""
from typing import Collection, List, Tuple

import tensorflow as tf

from interact_drive.car import PlannerCar


class InverseOptimalControl(object):
    """
    Abstract class for inverse optimal control algorithms.
    """

    def __init__(self, car: PlannerCar, **kwargs):
        self.car = car
        self.world = self.car.env

    def rationalize(self, trajectory: List[Tuple]) -> tf.Tensor:
        """
        Args:
            trajectory: a list of (state, controls) tuples, where state is
                a list of car states and control is a list of controls.

                We assume that the states and controls are in the same order as
                    self.world.state.

        Returns:
            weights: a tensor of weights that rationalizes the behavior
        """
        raise NotImplementedError

    def rationalize_trajectories(
            self,
            trajectories: Collection[List[Tuple]],
    ) -> tf.Tensor:
        """
        Args:
            trajectories: a collection of trajectories, where each trajectory is
                a list of (state, controls) tuples, where state is a list of
                car states and control is a list of controls.

                We assume that the states and controls are in the same order as
                    self.world.state.

        Returns:
            weights: a tensor of weights that rationalizes the behavior
        """
        raise NotImplementedError
