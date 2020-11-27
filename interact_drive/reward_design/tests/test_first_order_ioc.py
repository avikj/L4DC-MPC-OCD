"""Tests for inverse locally optimal control algorithms."""

import unittest

import numpy as np
import tensorflow as tf

from interact_drive.world import CarWorld
from interact_drive.reward_design.first_order_ioc import \
    InverseLocallyOptimalControl, LinearInverseLocallyOptimalControl
from interact_drive.reward_design.tests.linearTargetSpeedPlannerCar import \
    LinearTargetSpeedPlannerCar


class ILOCTests(unittest.TestCase):
    """
    Tests for the gradient-based ILOC method.

    Note that we set up the reward functions so that the task of inferring
    reward parameters is (strongly) convex from the initialization,
    so we should recover the true weights.
    """

    def setUp(self):
        self.world = CarWorld()
        self.npt = np.testing
        self.initial_theta = tf.constant([1., 1.])

    def test_iloc_correct_init_speed(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array([0., 0., 1., np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=10,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 6

        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = InverseLocallyOptimalControl(car, weight_norm=1.,
                                           initial_weights=self.initial_theta)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        self.npt.assert_allclose(weights_hat.numpy(),
                                 tf.nn.l2_normalize(weights_tf).numpy(),
                                 rtol=1e-4)

    def test_iloc_correct_init_speed_friction(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array([0., 0., 1., np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.2,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5

        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = InverseLocallyOptimalControl(car, weight_norm=1.,
                                           initial_weights=self.initial_theta)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        self.npt.assert_allclose(weights_hat.numpy(),
                                 tf.nn.l2_normalize(weights_tf).numpy(),
                                 rtol=1e-4)

    def test_iloc_incorrect_init_speed(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 0.5, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5
        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = InverseLocallyOptimalControl(car, weight_norm=1.,
                                           initial_weights=self.initial_theta)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        self.npt.assert_allclose(weights_hat.numpy(),
                                 tf.nn.l2_normalize(weights_tf).numpy(),
                                 rtol=1e-4)

    def test_iloc_incorrect_init_speed_friction(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 0.5, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.2,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5
        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = InverseLocallyOptimalControl(car, weight_norm=1.,
                                           initial_weights=self.initial_theta)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        self.npt.assert_allclose(weights_hat.numpy(),
                                 tf.nn.l2_normalize(weights_tf).numpy(),
                                 rtol=1e-4)


# class CIOCTests(unittest.TestCase):
#     pass

class LILOCTests(unittest.TestCase):
    """
    Tests for the SVD/Convex Optimization-based Linear ILOC method.

    Because there are multiple solutions to the optimization, even with norm 1
    (notably, if w is a solution, so is -w), we need to consider all these cases
    in our tests
    """

    def setUp(self):
        self.world = CarWorld()
        self.npt = np.testing

    def test_liloc_correct_init_speed(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array([0., 0., 1., np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=10,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 6

        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = LinearInverseLocallyOptimalControl(car, weight_norm=1.)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        test_val = (np.isclose(weights_hat.numpy(),
                               tf.nn.l2_normalize(weights_tf).numpy(),
                               rtol=1e-4).all()
                    or np.isclose(weights_hat.numpy(),
                                  -tf.nn.l2_normalize(weights_tf).numpy(),
                                  rtol=1e-4).all())

        self.assertTrue(test_val)

    def test_liloc_correct_init_speed_friction(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array([0., 0., 1., np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.2,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5

        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = LinearInverseLocallyOptimalControl(car, weight_norm=1.)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        test_val = (np.isclose(weights_hat.numpy(),
                               tf.nn.l2_normalize(weights_tf).numpy(),
                               rtol=1e-4).all()
                    or np.isclose(weights_hat.numpy(),
                                  -tf.nn.l2_normalize(weights_tf).numpy(),
                                  rtol=1e-4).all())

        self.assertTrue(test_val)

    def test_liloc_incorrect_init_speed(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 0.5, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5
        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = LinearInverseLocallyOptimalControl(car, weight_norm=1.)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        test_val = (np.isclose(weights_hat.numpy(),
                               tf.nn.l2_normalize(weights_tf).numpy(),
                               rtol=1e-4).all()
                    or np.isclose(weights_hat.numpy(),
                                  -tf.nn.l2_normalize(weights_tf).numpy(),
                                  rtol=1e-4).all())

        self.assertTrue(test_val)

    def test_liloc_incorrect_init_speed_friction(self):
        weights_tf = tf.constant([2., -1.], dtype=tf.float32)
        car = LinearTargetSpeedPlannerCar(self.world,
                                          tf.constant(
                                              np.array(
                                                  [0., 0., 0.5, np.pi / 2]),
                                              dtype=tf.float32),
                                          horizon=5, target_speed=0.,
                                          weights=weights_tf,
                                          friction=0.2,
                                          planner_args=dict(n_iter=200,
                                                            learning_rate=5.0))
        # this leads to zero controls
        self.world.add_car(car)
        trajectory = []
        traj_len = 5
        for t in range(traj_len):
            past_state, controls, next_state = self.world.step(dt=0.1)
            trajectory.append((past_state, controls))

        ioc = LinearInverseLocallyOptimalControl(car, weight_norm=1.)
        weights_hat = ioc.rationalize(trajectory, n_iter=300)
        test_val = (np.isclose(weights_hat.numpy(),
                               tf.nn.l2_normalize(weights_tf).numpy(),
                               rtol=1e-4).all()
                    or np.isclose(weights_hat.numpy(),
                                  -tf.nn.l2_normalize(weights_tf).numpy(),
                                  rtol=1e-4).all())

        self.assertTrue(test_val)


if __name__ == '__main__':
    # Note: these unittests take about a few minutes to run
    unittest.main()
