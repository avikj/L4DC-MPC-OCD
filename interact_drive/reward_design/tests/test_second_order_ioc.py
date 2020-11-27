"""Tests for second order IOC methods (such as CIOC)"""

import unittest

import numpy as np
import tensorflow as tf

from interact_drive.world import CarWorld
from interact_drive.reward_design.second_order_ioc import LocalCIOC
from interact_drive.reward_design.tests.linearTargetSpeedPlannerCar import \
    LinearTargetSpeedPlannerCar


##

class CIOCTests(unittest.TestCase):
    def setUp(self):
        self.world = CarWorld()
        self.npt = np.testing
        logger = tf.get_logger()
        logger.setLevel(20)

    def test_cioc_correct_init_speed(self):
        initial_theta = tf.constant([1., 1.])
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        world = self.world
        car = LinearTargetSpeedPlannerCar(world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 1.0, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=10,
                                                            learning_rate=5.0))
        world.add_car(car)
        trajectory = []
        traj_len = 5

        for t in range(traj_len):
            past_state, controls, next_state = world.step(dt=0.1)
            trajectory.append((past_state, controls))

        cioc = LocalCIOC(car, weight_norm=10000.,
                         initial_weights=initial_theta)
        weights = cioc.rationalize(trajectory)
        weights_np = tf.nn.l2_normalize(weights).numpy() * np.sqrt(5)
        self.npt.assert_allclose(weights_tf.numpy(), weights_np, rtol=1e-3)

    def test_cioc_bad_theta_init(self):
        """
        The initial_theta is set to -theta. This means that the first-order
        ioc methods fail.
        """
        initial_theta = tf.constant([-2., 1.])
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        world = self.world
        car = LinearTargetSpeedPlannerCar(world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 1.0, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=10,
                                                            learning_rate=5.0))
        world.add_car(car)
        trajectory = []
        traj_len = 5

        for t in range(traj_len):
            past_state, controls, next_state = world.step(dt=0.1)
            trajectory.append((past_state, controls))

        cioc = LocalCIOC(car, weight_norm=10000.,
                         initial_weights=initial_theta)
        weights = cioc.rationalize(trajectory)
        weights_np = tf.nn.l2_normalize(weights).numpy() * np.sqrt(5)
        self.npt.assert_allclose(weights_tf.numpy(), weights_np, rtol=1e-3)


if __name__ == '__main__':
    unittest.main()
