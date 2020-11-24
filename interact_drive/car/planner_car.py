"""Module containing the base class for planner cars."""

from typing import Iterable, Union

import numpy as np
import tensorflow as tf

from interact_drive.car.car import Car
from interact_drive.world import CarWorld


class PlannerCar(Car):
    """
    A car that performs some variant of model predictive control, maximizing the
     sum of rewards.

    TODO(chanlaw): this should probably also be an abstract class using ABCMeta.
    """

    def __init__(self, env: CarWorld,
                 init_state: Union[np.array, tf.Tensor, Iterable],
                 horizon: int, color: str = 'orange', opacity: float = 1.0,
                 friction: float = 0.2,
                 planner_args: dict = None,
                 check_plans: bool = False,
                 **kwargs):
        """
        Args:
            env: the carWorld associated with this car.
            init_state: a vector of the form (x, y, vel, angle), representing
                        the x-y coordinates, velocity, and heading of the car.
            horizon: the planning horizon for this car.
            color: the color of this car. Used for visualization.
            opacity: the opacity of the car. Used in visualization.
            friction: the friction of this car, used in the dynamics function.
            planner_args: the arguments to the planner (if any).
            check_plans: checks if the other cars are fixed_control_cars, and if so feeds in their controls to optimization
        """
        super().__init__(env, init_state, color=color, opacity=opacity,
                         friction=friction, **kwargs)
        self.horizon = horizon
        self.planner = None
        self.plan = []
        if planner_args is None:
            planner_args = {}
        self.planner_args = planner_args
        self.check_plans = check_plans

    def initialize_planner(self, planner_args):
        from interact_drive.planner.naive_planner import NaivePlanner
        self.planner = NaivePlanner(self.env, self, self.horizon,
                                    **planner_args)

    def _get_next_control(self):
        if self.planner is None:
            self.initialize_planner(self.planner_args)

        if self.check_plans:
            other_plans = []
            for i, other_car in enumerate(self.env.cars):
                if i == self.index:
                    other_plan = [tf.constant([0.0], dtype=tf.float32)] * self.horizon
                    other_plan = tf.stack(other_plan, axis=0)
                else:
                    other_plan = []
                    for j in range(self.horizon):
                        if hasattr(other_car, 'plan') and other_car.plan is not None:
                            if j < len(other_car.plan):
                                other_plan.append(other_car.plan[j])
                            else:
                                if hasattr(other_car, 'default_control') and other_car.default_control is not None:
                                    other_plan.append(other_car.default_control)
                                else:
                                    other_plan.append(tf.constant([0., 0.], dtype=tf.float32))
                        else:
                            other_plan.append(tf.constant([0., 0.], dtype=tf.float32))
                    other_plan = tf.stack(other_plan, axis=0)

                other_plans.append(other_plan)
            self.plan = self.planner.generate_plan(other_controls=other_plans)
        else:
            self.plan = self.planner.generate_plan()


        return tf.identity(self.plan[0])

    # Note: don't uncommment this or you will mess with multiple inheritance!
    # @tf.function
    # def reward_fn(self, state, control):
    #     raise NotImplementedError
