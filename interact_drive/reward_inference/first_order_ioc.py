"""
Inverse optimal control methods based on first order (ie, gradient) optimality
conditions.
"""
from typing import Collection, List, Optional, Union, Tuple

import tensorflow as tf

from interact_drive.car import PlannerCar, LinearRewardCar
from interact_drive.simulation_utils import next_car_state
from interact_drive.planner import NaivePlanner
from interact_drive.reward_inference.inverse_optimal_control import \
    InverseOptimalControl


class InverseLocallyOptimalControl(InverseOptimalControl):
    """
    A necessary condition for a trajectory to be locally optimal is that the
    gradients dr/du of the reward wrt the controls are zero.

    This optimizer directly optimizes the weights such that dr/du = 0 for all
    controls u. In particular, we minimize the l2 norm of the gradient dr/du.
    """

    def __init__(self, car: PlannerCar,
                 weight_norm: Union[tf.Tensor, float] = 1.,
                 initial_weights: Union[None, tf.Tensor] = None, **kwargs):
        """
        Initializes this IOC algorithm.

        Args:
            car: a car that performs planning. Must have a weights attribute
                that parameterizes the car's reward function.
            weight_norm: the l2 norm of the inferred weights (to prevent e.g.
                the degenerate solution of all 0s.)
            **kwargs: extra keyword arguments, for multiple inheritance's sake.
        """
        super().__init__(car, **kwargs)
        if not isinstance(car.planner, NaivePlanner):
            raise NotImplementedError("Only NaivePlanners are supported.")
        if initial_weights is None:
            self.initial_weights = tf.ones_like(self.car.weights)
        else:
            self.initial_weights = tf.identity(initial_weights)
        self.unnorm_weights = tf.Variable(self.initial_weights)
        self.weight_norm = weight_norm
        self.control_variables = [tf.Variable([0., 0.], dtype=tf.float32) for _
                                  in range(self.car.horizon)]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @property
    def weights(self):
        norm_weights = self.weight_norm * tf.nn.l2_normalize(
            self.unnorm_weights,
            axis=-1)
        return norm_weights

    @weights.setter
    def weights(self, new_value):
        self.unnorm_weights.assign(new_value)

    @tf.function
    def segment_loss(self, weights: tf.Variable,
                     initial_state: Union[tf.Variable, tf.Tensor],
                     controls: Collection[tf.Variable],
                     index: Optional[int] = None) -> tf.Tensor:
        """
        Computes the loss of the weights on the sequence of controls starting
        from the initial_state.

        In this case, the loss is simply squared l2 norm of the gradient of the
        reward with respect to the controls.

        Args:s
            weights: an estimate of the reward weights
            initial_state: the initial state that this trajectory begins at
            controls: the controls associated with self.car
            index: the index of the controls we want to take the gradient
                    with respect to. If index=None, we take it with respect
                    to all the controls.

        Returns:
            loss: a scalar representing the loss
        """
        with tf.GradientTape() as t:
            r = self.car.planner.reward_func(initial_state, controls, weights)
        if index is None:
            dr_dus = tf.stack(t.gradient(r, controls), axis=-1)
        else:
            dr_dus = t.gradient(r, controls[index])

        return tf.reduce_sum(dr_dus ** 2)

    @tf.function
    def compute_total_loss(self, weights: tf.Variable,
                           trajectory: List[Tuple]) -> tf.Tensor:
        horizon = self.car.planner.horizon
        total_loss = 0

        for i in range(len(trajectory) - horizon + 1):
            initial_state = trajectory[i][0]
            for j in range(horizon):
                self.control_variables[j].assign(
                    trajectory[i + j][1][self.car.index])
            if i < len(trajectory) - horizon:
                # to prevent double counting, only add the gradient of
                # the first control
                total_loss += self.segment_loss(weights, initial_state,
                                                self.control_variables,
                                                index=0)
            else:
                # at the end, sum up the gradients of all the controls
                total_loss += self.segment_loss(weights, initial_state,
                                                self.control_variables)
        return total_loss

    def rationalize(self, trajectory: List[Tuple],
                    n_iter: int = 300) -> tf.Tensor:
        """
        Args:
            trajectory: a list of (state, control) tuples, where state is a list
             of the car states of all the cars and control is a list of controls
             of all the cars.
            n_iter: the number of iterations to minimize the loss for.

        Returns:
            weights: a tensor of weights that rationalizes the trajectory as
                        thfe behavior of self.car
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.unnorm_weights.assign(self.initial_weights)

        def loss_fn():
            return self.compute_total_loss(self.weights, trajectory)

        for _ in range(n_iter):
            self.optimizer.minimize(loss_fn, self.unnorm_weights)
        return self.weights

    def rationalize_trajectories(
            self,
            trajectories: Collection[List[Tuple]],
            n_iter: int = 200,
    ) -> tf.Tensor:
        """
        Args:
            trajectories: a collection of trajectories, where each trajectory is
                a list of (state, controls) tuples, where state is a list of
                car states and control is a list of controls for all the cars.

                We assume that the states and controls are in the same order as
                    self.world.state.
            n_iter: the number of iterations to minimize the loss for.

        Returns:
            weights: a tensor of weights that rationalizes the behavior
        """
        self.unnorm_weights.assign(self.initial_weights)

        def loss_fn():
            loss = 0
            for trajectory in trajectories:
                loss += self.compute_total_loss(self.weights, trajectory)
            return loss

        for _ in range(n_iter):
            self.optimizer.minimize(loss_fn, self.unnorm_weights)
        return self.weights


class LinearInverseLocallyOptimalControl(InverseLocallyOptimalControl):
    """
    A necessary condition for a trajectory to be locally optimal is that the
    gradients dr/du of the reward wrt the controls are zero.

    When the reward function is a linear function of some set of features
        r(x, u) = w * phi(x, u),
    we can directly minimize dr/du = w * dphi/du.

    In particular, we simply perform SVD to either
        a) find a unit vector w that lies inside the nullspace of the Jacobian
            dphi/du, or
        b) solve the least squares problem.
    """

    def __init__(self, car: LinearRewardCar, **kwargs):
        if not isinstance(car, PlannerCar):
            raise ValueError("Car must also be a planner car.")
        super().__init__(car=car, **kwargs)

    @tf.function
    def segment_jacobian(
            self, initial_state: Union[tf.Variable, tf.Tensor],
            controls: Collection[tf.Variable],
            index: Union[None, int] = None) -> List[tf.Tensor]:
        """
        Computes the Jacobian of the features with respect to the sequence of
        controls starting from the initial_state.

        Args:
            weights: an estimate of the reward weights.
            initial_state: the initial state that this trajectory begins at.
            controls: the controls associated with self.car.
            index: the index of the controls we want to take the Jacobian
                    with respect to. If none, we take it wrt all the controls.

        Returns:
            Jacobian: a list of tensors representing the Jacobian of the sum of
                    features with respect to the controls.
        """
        world_state = initial_state
        with tf.GradientTape() as t:
            features = 0
            for control in controls:
                new_state = []
                dt = self.car.env.dt
                for i in range(len(self.world.cars)):
                    x, car = world_state[i], self.world.cars[i]
                    if i == self.car.index:
                        new_x = car.dynamics_fn(x, control, dt)
                    else:
                        zero_control = tf.zeros_like(control)
                        new_x = next_car_state(x, zero_control, dt, friction=0.)
                    new_state.append(new_x)
                world_state = tf.stack(new_state, axis=0)
                features += self.car.features(world_state, control)

        if index is None:
            return t.jacobian(features, controls)
        else:
            return [t.jacobian(features, controls[index])]

    @tf.function
    def total_jacobian(self, trajectory: List[Tuple]) -> tf.Tensor:
        """

        Args:
            trajectory: a list of (state, control) tuples, where state is a list
             of the car states of all the cars and control is a list of controls
             of all the cars.

        Returns:
            Jacobian: the concatenated Jacobian of the sum of features
                    with respect to all the controls.
        """
        horizon = self.car.planner.horizon
        total_jacobian = []

        for i in range(len(trajectory) - horizon + 1):
            initial_state = trajectory[i][0]
            for j in range(horizon):
                self.control_variables[j].assign(
                    trajectory[i + j][1][self.car.index])
            if i < len(trajectory) - horizon:
                # to prevent double counting, only take the Jacobian of
                # the first control
                total_jacobian += self.segment_jacobian(initial_state,
                                                        self.control_variables,
                                                        index=0)

            else:
                # at the end, sum up the gradients of all the controls
                total_jacobian += self.segment_jacobian(initial_state,
                                                        self.control_variables)

        total_jacobian = tf.concat(total_jacobian, axis=-1)

        return total_jacobian

    @tf.function
    def rationalize(self, trajectory: List[Tuple],
                    **kwargs) -> tf.Tensor:
        """
        Args:
            trajectory: a list of (state, control) tuples, where state is a list
             of the car states of all the cars and control is a list of controls
             of all the cars.

        Returns:
            weights: a tensor of weights that rationalizes the trajectory as
                        the behavior of self.car.
        """
        jacobian = self.total_jacobian(trajectory)
        _, u, _ = tf.linalg.svd(jacobian)
        self.unnorm_weights.assign(u[:, -1])

        return self.weights

    def rationalize_trajectories(
            self,
            trajectories: Collection[List[Tuple]],
            **kwargs
    ) -> tf.Tensor:
        """
        Args:
            trajectories: a collection of trajectories, where each trajectory is
                a list of (state, controls) tuples, where state is a list of
                car states and control is a list of controls for all the cars.

                We assume that the states and controls are in the same order as
                    self.world.state.

        Returns:
            weights: a tensor of weights that rationalizes the behavior
        """
        jacobian = []
        for trajectory in trajectories:
            jacobian.append(self.total_jacobian(trajectory))
        jacobian = tf.concat(jacobian, axis=-1)
        _, u, _ = tf.linalg.svd(jacobian)
        self.unnorm_weights.assign(u[:, -1])

        return self.weights
##
