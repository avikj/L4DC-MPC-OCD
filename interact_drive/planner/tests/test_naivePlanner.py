"""Tests for the naivePlanner class.
"""

import unittest

import numpy as np
import tensorflow as tf

from interact_drive.car import FixedPlanCar
from interact_drive.planner.naive_planner import NaivePlanner
from interact_drive.world import ThreeLaneCarWorld
from interact_drive.planner.tests.targetSpeedRewardMaximizerCar import \
    TargetSpeedPlannerCar


class TargetSpeedTestCases(unittest.TestCase):
    def setUp(self):
        self.world = ThreeLaneCarWorld()
        self.init_state = np.array([0., 0., 1., np.pi / 2], dtype=np.float32)

    def test_zero_friction_correct_speed(self):
        car = TargetSpeedPlannerCar(self.world, self.init_state, 4,
                                    target_speed=1., friction=0.)
        self.world.add_car(car)

        planner = NaivePlanner(self.world, car, horizon=5, learning_rate=5.0,
                               n_iter=100)
        plan = planner.generate_plan([car.state])
        self.assertEqual(len(plan), 5)
        for i in range(len(plan)):
            np.testing.assert_allclose(plan[i].numpy(), np.array([0., 0.]),
                                       atol=1e-5)

    # def test_zero_friction_incorrect_speed(self):
    #     car = TargetSpeedPlannerCar(self.world, self.init_state, 4,
    #                                 target_speed=2., friction=0.)
    #     self.world.add_car(car)
    #
    #     planner = NaivePlanner(self.world, car, horizon=5, learning_rate=5.0,
    #                            n_iter=500)
    #     plan = planner.generate_plan([car.state])
    #     self.assertEqual(len(plan), 5)
    #     print(plan)
    #     np.testing.assert_allclose(plan[0].numpy(), np.array([10., 0.]),
    #                                atol=1e-5)
    #     for i in range(1, len(plan)):
    #         np.testing.assert_allclose(plan[i].numpy(), np.array([0., 0.]),
    #                                    atol=1e-5)

    def test_friction_correct_speed(self):
        friction = 0.5
        car = TargetSpeedPlannerCar(self.world, self.init_state, 4,
                                    target_speed=1., friction=friction)
        self.world.add_car(car)

        planner = NaivePlanner(self.world, car, horizon=3, learning_rate=5.0,
                               n_iter=500)
        plan = planner.generate_plan([car.state])
        self.assertEqual(len(plan), 3)
        for i in range(len(plan)):
            np.testing.assert_allclose(plan[i].numpy(),
                                       np.array([friction * 1.0 ** 2, 0.]),
                                       atol=1e-5)


class TestPlanVsFixedPlanCar(unittest.TestCase):
    """
    Conduct tests of the NaivePlanner vs a FixedPlanCar
    """

    def setUp(self):
        self.world = ThreeLaneCarWorld()
        self.init_state = np.array([0., 0., 1., np.pi / 2], dtype=np.float32)
        self.other_car_init_state = np.array([0.1, 0., 1., np.pi / 2], dtype=np.float32)

    def test_no_interaction(self):
        car = TargetSpeedPlannerCar(self.world, self.init_state, 4,
                                    target_speed=1., friction=0.)
        other_car = FixedPlanCar(self.world, self.other_car_init_state,
                                 plan=[np.array([0., 0.], dtype=np.float32)],
                                 default_control=np.array([0., 0.], dtype=np.float32))
        self.world.add_cars([car, other_car])

        planner = NaivePlanner(self.world, car, horizon=3, learning_rate=5.0,
                               n_iter=500)
        other_plan = []
        for j in range(planner.horizon):
            if j < len(other_car.plan):
                other_plan.append(other_car.plan[j])
            else:
                other_plan.append(other_car.default_control)

        other_plan = tf.stack(other_plan, axis=0)

        plan = planner.generate_plan([car.state, other_car.state],
                                     other_controls=[tf.constant(0.0), other_plan])


if __name__ == '__main__':
    unittest.main()
