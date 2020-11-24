"""Tests for the simulation utility functions."""

import unittest

import numpy as np
import tensorflow as tf

from interact_drive.simulation_utils import CarControl, CarState, \
    BatchedCarControl, BatchedCarState, next_car_state, new_next_car_state


class TestCarState(unittest.TestCase):
    def test_single_state_0(self):
        state = [1., 0.5, 2., np.pi / 2]
        state = CarState(state)
        self.assertAlmostEqual(state.x.numpy(), 1.)
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v.numpy(), 2.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi / 2)

    def test_single_state_1(self):
        state = np.array([1., 0.5, 2., np.pi / 2])
        state = CarState(state)
        self.assertAlmostEqual(state.x.numpy(), 1.)
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v.numpy(), 2.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi / 2)

    def test_single_state_2(self):
        state = tf.constant([1., 0.5, 2., np.pi / 2])
        state = CarState(state)
        self.assertAlmostEqual(state.x.numpy(), 1.)
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v.numpy(), 2.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi / 2)

    def test_single_state_invalid_input(self):
        state = [1., 0.5, 2.]
        with self.assertRaises(ValueError):
            state = CarState(state)

    def test_state_assign(self):
        state = [1., 0.5, 2., np.pi / 2]
        state = CarState(state)
        state.assign([0., 0.5, 1., np.pi])
        self.assertAlmostEqual(state.x.numpy(), 0., )
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v.numpy(), 1.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi, places=6)

    def test_state_assign_add(self):
        state = [1., 0.5, 2., np.pi / 2]
        state = CarState(state)
        state.assign_add([-1., 0., -1., np.pi / 2])
        self.assertAlmostEqual(state.x.numpy(), 0.)
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v, 1.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi, places=6)

    def test_state_assign_sub(self):
        state = [1., 0.5, 2., np.pi / 2]
        state = CarState(state)
        state.assign_sub([1., 0., 1., -np.pi / 2])
        self.assertAlmostEqual(state.x.numpy(), 0.)
        self.assertAlmostEqual(state.y.numpy(), 0.5)
        self.assertAlmostEqual(state.v.numpy(), 1.)
        self.assertAlmostEqual(state.angle.numpy(), np.pi, places=6)

    def test_batch_state(self):
        state = [[1., 0.5, 2., np.pi / 2], [0., 0., 1., np.pi / 2]]
        state = BatchedCarState(state)
        np.testing.assert_almost_equal(state.x.numpy(), [1., 0.])
        np.testing.assert_almost_equal(state.y.numpy(), [0.5, 0.])
        np.testing.assert_almost_equal(state.v.numpy(), [2., 1.])
        np.testing.assert_almost_equal(state.angle.numpy(),
                                       [np.pi / 2, np.pi / 2])


class TestCarControl(unittest.TestCase):
    def test_single_control(self):
        control = CarControl([1., 0.])
        self.assertAlmostEqual(control.acc.numpy(), 1.)
        self.assertAlmostEqual(control.ang_vel.numpy(), 0.)

    def test_batched_control(self):
        control = BatchedCarControl([[1., 0.], [0.5, np.pi / 4]])
        np.testing.assert_almost_equal(control.acc.numpy(), [1., 0.5])
        np.testing.assert_almost_equal(control.ang_vel.numpy(), [0., np.pi / 4])

    def test_control_invalid_input_0(self):
        with self.assertRaises(ValueError):
            CarControl([1., 0., 10.])

    def test_control_invalid_input_1(self):
        with self.assertRaises(ValueError):
            BatchedCarControl([1., 0.])


class TestDynamicsFn(unittest.TestCase):
    def test_invalid_state(self):
        with self.assertRaises(ValueError):
            next_car_state(state=[0., 0.],
                           control=[0., 0.],
                           dt=0.1)

    def test_invalid_control(self):
        with self.assertRaises(ValueError):
            next_car_state(state=[0., 0., 1., np.pi / 2],
                           control=[0., 0., 0.],
                           dt=0.1)


class TestNewDynamicsFn(unittest.TestCase):
    def test_next_car_state_0(self):
        state = CarState([0., 0., 1., np.pi / 2])
        control = CarControl([0., 0.])
        next_state = new_next_car_state(state, control, friction=0., dt=1.)
        self.assertAlmostEqual(next_state.x.numpy(), 0.)
        self.assertAlmostEqual(next_state.y.numpy(), 1.)
        self.assertAlmostEqual(next_state.v.numpy(), 1.)
        self.assertAlmostEqual(next_state.angle.numpy(), np.pi / 2)

    def test_next_car_state_1(self):
        state = CarState([0., 0., 1., np.pi / 2])
        control = CarControl([0., 0.])
        next_state = new_next_car_state(state, control, friction=1.0, dt=1.)
        self.assertAlmostEqual(next_state.x.numpy(), 0.)
        self.assertAlmostEqual(next_state.y.numpy(), 0.5)
        self.assertAlmostEqual(next_state.v.numpy(), 0.)
        self.assertAlmostEqual(next_state.angle.numpy(), np.pi / 2)

    def test_next_car_state_2(self):
        state = CarState([0., 0., 1., np.pi / 2])
        control = CarControl([0., 0.])
        next_state = new_next_car_state(state, control, friction=0.5, dt=1.)
        self.assertAlmostEqual(next_state.x.numpy(), 0.)
        self.assertAlmostEqual(next_state.y.numpy(), 0.75)
        self.assertAlmostEqual(next_state.v.numpy(), 0.5)
        self.assertAlmostEqual(next_state.angle.numpy(), np.pi / 2)

    def test_next_car_state_3(self):
        state = CarState([0., 0., 1., 0.])
        control = CarControl([0., 0.])
        next_state = new_next_car_state(state, control, friction=0.5, dt=1.)
        self.assertAlmostEqual(next_state.x.numpy(), 0.75)
        self.assertAlmostEqual(next_state.y.numpy(), 0.)
        self.assertAlmostEqual(next_state.v.numpy(), 0.5)
        self.assertAlmostEqual(next_state.angle.numpy(), 0.)

    def test_next_car_state_batched(self):
        state = BatchedCarState([[0., 0., 1., np.pi / 2],
                                 [0., 0., 1., 0.]])
        control = CarControl([0., 0.])
        next_state = new_next_car_state(state, control, friction=0.5, dt=1.)
        np.testing.assert_almost_equal(next_state.x.numpy(), [0., 0.75])
        np.testing.assert_almost_equal(next_state.y, [0.75, 0.])
        np.testing.assert_almost_equal(next_state.v, [0.5, 0.5])
        np.testing.assert_almost_equal(next_state.angle, [np.pi / 2, 0.])


if __name__ == '__main__':
    unittest.main()
