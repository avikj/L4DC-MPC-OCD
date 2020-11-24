"""Planner that assumes all other cars travel at constant velocity."""

from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from interact_drive.planner.car_planner import CarPlanner
from interact_drive.simulation_utils import next_car_state
from interact_drive.world import CarWorld
from interact_drive.car.car import Car


class NaivePlanner(CarPlanner):
    """
    MPC-based CarPlanner that assumes all the other cars are FixedVelocityCars.
    """

    def __init__(self, world: CarWorld, car: Car, horizon: int,
                 learning_rate: float = 0.1, n_iter: int = 100, leaf_evaluation=None, extra_inits=False):
        super().__init__(world, car)
        self.leaf_evaluation = leaf_evaluation
        self.reward_func = self.initialize_mpc_reward()
        self.learning_rate = learning_rate
        self.horizon = horizon
        self.planned_controls = [tf.Variable([0., 0.]) for _ in range(horizon)]
        self.n_iter = n_iter
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        print('extra_inits:', extra_inits)
        self.extra_inits = extra_inits

    def initialize_mpc_reward(self):
        @tf.function
        def mpc_reward(init_state: Union[tf.Tensor, tf.Variable],
                       controls: Union[tf.Tensor, tf.Variable],
                       other_controls: Optional[List[Union[tf.Tensor, tf.Variable]]] = None,
                       weights: Union[
                           None, tf.Tensor, tf.Variable, np.array] = None):
            world_state = init_state
            dt = self.world.dt
            r = 0
            traj = []

            for ctrl_idx in range(self.horizon):
                control = controls[ctrl_idx]
                new_state = []
                for i in range(len(self.world.cars)):
                    x, car = world_state[i], self.world.cars[i]
                    if i == self.car.index:
                        new_x = car.dynamics_fn(x, control, dt)
                    else:
                        # we assume 0 friction
                        if other_controls is not None:
                            v, angle = x[2], x[3]
                            acc, ang_vel = other_controls[i][ctrl_idx][0], other_controls[i][ctrl_idx][1]
                            update = tf.stack([tf.cos(angle) * (v * dt + 0.5 * acc * dt ** 2),
                                               tf.sin(angle) * (v * dt + 0.5 * acc * dt ** 2),
                                               acc * dt,
                                               ang_vel * dt])
                        else:
                            v, angle = x[2], x[3]
                            update = tf.stack([tf.cos(angle) * v * dt,
                                               tf.sin(angle) * v * dt,
                                               0.0,
                                               0.0])
                        new_x = x + update
                    new_state.append(new_x)
                world_state = tf.stack(new_state, axis=0)
                if ctrl_idx == self.horizon - 1 and self.leaf_evaluation is not None:
                    r += self.leaf_evaluation(world_state, control)
                else:
                    if weights is not None:
                        r += self.car.reward_fn(world_state, control, weights)
                    else:
                        r += self.car.reward_fn(world_state, control)
                traj.append(world_state)
            return r

        return mpc_reward

    def generate_plan(self,
                      init_state: Union[
                          None, tf.Tensor, tf.Variable, np.array] = None,
                      weights: Union[
                          None, tf.Tensor, tf.Variable, np.array] = None,
                      other_controls: Optional[List] = None,
                      use_lbfgs=False):

        """
        Generates a sequence of controls of length self.horizon by performing
        gradient ascent on the predicted reward of the resulting trajectory.

        Args:
            init_state: The initial state to plan from. If none, we use the
                        current state of the world associated with the car.
            weights: The weights of the reward function (if any).
                (Note: weights should only be not None if the reward function
                of the car associated with this planner takes as input a weight
                vector.)
            other_controls: List of sequences of controls belonging to the other cars.
            use_lbfgs: if true, use L-BFGS for optimization; otherwise, SGD is used.

        Returns:

        """

        init_controls = []
        init_controls.append([[0.0, 0.0] for _ in range(len(self.planned_controls))])
        init_controls.append([[0, -5 * 0.13] for _ in range(len(self.planned_controls))])
        init_controls.append([[0, 5 * 0.13] for _ in range(len(self.planned_controls))])

        if self.extra_inits:
            #init_controls.append([[2*self.car.friction*self.car.state[2] ** 2, 0.0] for _ in range(len(self.planned_controls))])
            init_controls.append([[self.car.friction*self.car.state[2] ** 2, 0.0] for _ in range(len(self.planned_controls))])
            init_controls.append([[self.car.friction*self.car.state[2]**2, -5 * 0.13] for _ in range(len(self.planned_controls))])
            init_controls.append([[self.car.friction*self.car.state[2]**2, 5 * 0.13] for _ in range(len(self.planned_controls))]) # use control bounds from old code base for left/right initialization
            #init_controls.append([[3*self.car.friction * self.car.state[2] ** 2, -5 * 0.13] for _ in range(len(self.planned_controls))])
            #init_controls.append([[3*self.car.friction * self.car.state[2] ** 2, 5 * 0.13] for _ in range(len(self.planned_controls))]) # use control bounds from old code base for left/right initialization


        if init_state is None:
            init_state = self.world.state

        def loss():
            return -self.reward_func(init_state, self.planned_controls, other_controls=other_controls, weights=weights)

        if use_lbfgs:
            def flat_controls_to_loss(flat_controls):
                return -self.reward_func(init_state, tf.reshape(flat_controls, (self.horizon, 2)), weights)

            import tensorflow_probability as tfp
            @tf.function
            def loss_and_grad(flat_controls):
                v, g = tfp.math.value_and_gradient(flat_controls_to_loss, flat_controls[0])
                return tf.convert_to_tensor([v]), tf.convert_to_tensor([g])

        losses = []
        opts = []
        for init_control in init_controls:
            for control, val in zip(self.planned_controls, init_control):
                control.assign(val)

            if use_lbfgs:
                opt = tfp.optimizer.lbfgs_minimize(loss_and_grad, initial_position=[
                    tf.reshape(self.planned_controls, self.horizon * 2)], max_iterations=200, tolerance=1e-12)
                # print("RESULT OF LBFGS_MINIMIZE",opt)
                losses.append(opt.objective_value.numpy()[0])

                opts.append(list(tf.reshape(opt.position, (self.horizon, 2)).numpy()))
            else:
                for _ in range(self.n_iter):
                    # tf.print("Optimizating controls:",self.planned_controls)
                    self.optimizer.minimize(loss, self.planned_controls)
                losses.append(loss().numpy())
                opts.append([c.numpy() for c in self.planned_controls])

        ##  show effects of multiple initialization
        # print("LRS initializations gave: ",opts)
        # print("selected:", losses.index(min(losses)))

        for control, val in zip(self.planned_controls, opts[losses.index(min(losses))]):
            control.assign(val)

        return self.planned_controls

##
