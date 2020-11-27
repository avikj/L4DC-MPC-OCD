"""
Inverse optimal control methods based on first (ie, gradient) and second order
(ie, hessian) optimality conditions.
"""
from typing import Collection, List, Tuple, Union

import tensorflow as tf

from interact_drive.reward_design.first_order_ioc import (
    InverseLocallyOptimalControl,
)

logger = tf.get_logger()


class LocalCIOC(InverseLocallyOptimalControl):
    """
    Implements Levine and Koltun (2012)'s
    "Continuous Inverse Optimal Control with Locally Optimal Examples".
    Computes the MLE weights for the trajectory, assuming the demonstrator is
    Boltzmann-rational.
    To prevent singular or ill-conditioned Hessians, we follow the procedure
    described in section 6.1 of the paper. In particular, we add a negative
    identity matrix, multiplied by a sufficiently large value theta_r, to the
    Hessian. We then use the augmented Lagrangian method to bring the value of
    theta_r close to 0. We use standard first-order optimization to solve each
    of the sub-problems that occur in the augmented Lagrangian method.
    """

    def __init__(self, *args, split_traj=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_r = tf.Variable(0.01)
        self.split_traj = split_traj

    @tf.function()
    def segment_gradient(
        self,
        weights: tf.Variable,
        initial_state: Union[tf.Variable, tf.Tensor],
        controls: List[Union[tf.Tensor, tf.Variable]],
        index: Union[None, int] = None,
    ) -> tf.Tensor:
        """
        Returns the gradient of the reward of this segment with respect to
        the given controls.
        Args:
            weights: an estimate of the reward weights.
            initial_state: the initial state that this trajectory begins at.
            controls: a list of controls associated with self.car during this
                    trajectory.
            index: the index of the controls we want to take the gradient
                    with respect to. If none, we take it wrt all the controls.
        Returns:
            gradient: a tensor representing the gradient of the reward with
                    respect to the controls.
        """
        with tf.GradientTape() as t:
            t.watch(controls)
            r = self.car.planner.reward_func(initial_state, controls, weights)

        if index is None:
            return tf.concat(t.gradient(r, controls), axis=-1)
        else:
            return t.gradient(r, controls[index])

    @tf.function
    def compute_total_augmented_loss(
        self,
        weights: tf.Variable,
        trajectory: List[Tuple],
        theta_r: tf.Variable,
        mu: tf.Tensor,
        lm: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the CIOC loss with augmented Lagrangian terms. We also return
        the sign of the determinant of the negative Hessian, to help with
        initializing theta_r.
        Args:
            weights: an estimate of the reward weights.
            trajectory: a list of (state, control) tuples, where state is a list
                of the car states of all the cars and control is a list of
                controls of all the cars.
            theta_r: the weight of the dummy feature, used to ensure the
                loss is well-defined.
            mu: the penalty weight of the constraint in the augmented
                Lagrangian.
            lm: the Lagrange multiplier for the constraint in the augmented
                Lagrangian.
        Returns:
            loss: the CIOC loss with appropriate penalty terms added.
            sign: the sign of the determinant of the negative Hessian.
        """
        horizon = self.car.planner.horizon
        gradients = []
        controls = [control[self.car.index] for state, control in trajectory]
        with tf.GradientTape() as t:
            t.watch(controls)
            # compute gradients
            if self.split_traj:
                for i in range(0, max(len(trajectory) - horizon + 1, 1), horizon):
                    logger.debug(
                        "tracing subgraph starting from control {}".format(i)
                    )
                    initial_state = trajectory[i][0]
                    controls_i = controls[i: i + horizon]
                    gradients.append(
                        self.segment_gradient(
                            weights,
                            initial_state=initial_state,
                            controls=controls_i,
                        )
                    )
            else:
                for i in range(max(len(trajectory) - horizon + 1, 1)):
                    logger.debug(
                        "tracing subgraph starting from control {}".format(i)
                    )
                    initial_state = trajectory[i][0]
                    controls_i = controls[i: i + horizon]
                    if i < len(trajectory) - horizon:
                        # to prevent double counting, only add the gradient of
                        # the first control
                        gradients.append(
                            self.segment_gradient(
                                weights,
                                initial_state=initial_state,
                                controls=controls_i,
                                index=0,
                            )
                        )
                    else:
                        gradients.append(
                            self.segment_gradient(
                                weights,
                                initial_state=initial_state,
                                controls=controls_i,
                            )
                        )

            gradients = tf.concat(gradients, axis=-1)

        # compute Hessian
        logger.debug("tracing hessian")
        hessian = tf.concat(
            t.jacobian(gradients, controls, experimental_use_pfor=True),
            axis=-1,
        )
        logger.debug("hessian is done")
        gradients = tf.expand_dims(gradients, axis=-1)  # vector -> matrix
        hessian = hessian - theta_r * tf.eye(len(trajectory) * len(controls[0]))
        log_hess_det = tf.linalg.slogdet(-hessian)

        logger.debug("tracing loss")
        log_ll = (
            0.5
            * tf.matmul(
            tf.matmul(tf.transpose(gradients), tf.linalg.inv(hessian)),
            gradients,
        )
            + 0.5 * log_hess_det.sign * log_hess_det.log_abs_determinant
            # Augmented Lagrangian terms:
            - 0.5 * mu * theta_r ** 2
            + lm * theta_r
        )

        return -log_ll, log_hess_det.sign

    def rationalize(
        self,
        trajectory: List[Tuple],
        initial_theta_r: Union[float, tf.Tensor] = 0.01,
        n_iter: int = 200,
        initial_mu: float = 10.0,
        tol: float = 0.01,
    ) -> tf.Tensor:
        """
        Rationalizes the trajectory using the augmented Lagrangian method
        described in section 6.1 of the Levine and Kolton (2012).
        We use the standard first-order (gradient-based) method to solve each
        optimization problem.
        Note: this function takes a long time to run, so it logs a lot of info
        along the way. Set logger level to info or lower to see the progress.
        Args:
            trajectory: a list of (state, controls) tuples, where state is
                a list of car states and control is a list of controls.
            initial_theta_r: the initial value of theta_r. Defaults to 0.01.
            n_iter: the number of gradient steps used to solve each optimization
                problem. Defaults to 200.
            initial_mu: The initial penalty weight in the augmented Lagrangian.
                Defaults to 10.
            tol: The tolerance for violations of the theta_r = 0 constraint.
                When |theta_r| < tol, we terminate the optimization.
                Unfortunately, if the tolerance is too small, numerical
                instability can occur. Defaults to 0.01.
        Returns:
            weights: a tensor of weights that rationalizes the behavior
        TODO(chanlaw): better understand the numerical instability that occurs
                when the tolerance is too small.
        TODO(chanlaw): function is too long (80+ lines) and should be split up
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.unnorm_weights.assign(self.initial_weights)
        self.theta_r.assign(initial_theta_r)
        theta_r = self.theta_r

        mu = tf.constant(initial_mu)
        lm = tf.constant(0.0)

        # first, increase initial theta_r until Hessian has negative determinant
        logger.info(
            "building loss function and "
            "initializing theta_r = {:.3f}".format(initial_theta_r)
        )
        _, sign = self.compute_total_augmented_loss(
            self.weights,
            trajectory=trajectory,
            theta_r=self.theta_r,
            mu=mu,
            lm=lm,
        )
        while tf.less(sign, 0):
            theta_r.assign(theta_r * 2)
            logger.info("\tdoubling theta_r to {:.3f}".format(theta_r.numpy()))
            _, sign = self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=self.theta_r,
                mu=mu,
                lm=lm,
            )

        logger.info("finished initializing theta_r")
        theta_r_val = theta_r.numpy()

        # next, optimize the CIOC loss using the augmented lagrangian method
        logger.info("initializing optimizer and optimizing weights and theta_r")

        def loss_fn():
            return self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=theta_r,
                mu=mu,
                lm=lm,
            )[0]

        for i in range(n_iter):
            if i % 10 == 0:
                tf.print("iteration", i)
            self.optimizer.minimize(loss_fn, [self.unnorm_weights, theta_r])

        while abs(theta_r.numpy()) > tol:
            logger.info(
                "|theta_r| = {:.3f} > {:.3f}".format(abs(theta_r.numpy()), tol)
                + ", so augmenting the Lagrange multiplier and trying again"
            )
            lm = lm - mu * theta_r

            if abs(theta_r.numpy()) - abs(theta_r_val) >= -5e-4:
                mu *= 10
                logger.info(
                    "\ttheta_r didn't decrease, so penalty "
                    "increased to to {:.2f}".format(mu.numpy())
                )
            theta_r_val = theta_r.numpy()

            for _ in range(n_iter):
                self.optimizer.minimize(loss_fn, [self.unnorm_weights, theta_r])

        logger.info(
            "|theta_r| = {:.3f} <= {:.3f}, ".format(abs(theta_r.numpy()), tol)
            + "so returning weights"
        )

        _, sign = self.compute_total_augmented_loss(
            self.weights, trajectory=trajectory, theta_r=theta_r, mu=mu, lm=lm
        )
        if sign < 0:
            logger.warning(
                "Negative Hessian is ill-conditioned (ie, |-H| < 0)"
                " - results may be nonsensical - "
                "try increasing the tolerance and rerunning."
            )
        return self.weights

    def rationalize_half_tf(
        self,
        trajectory: List[Tuple],
        initial_theta_r: Union[float, tf.Tensor] = 0.01,
        n_iter: int = 200,
        initial_mu: float = 10.0,
        tol: float = 0.01,
    ):
        """
        Rationalizes the trajectory using the augmented Lagrangian method
        described in section 6.1 of the Levine and Kolton (2012).
        We use the standard first-order (gradient-based) method to solve each
        optimization problem.
        Note: this function takes a long time to run, so it logs a lot of info
        along the way. Set logger level to info or lower to see the progress.
        Args:
            trajectory: a list of (state, controls) tuples, where state is
                a list of car states and control is a list of controls.
            initial_theta_r: the initial value of theta_r. Defaults to 0.01.
            n_iter: the number of gradient steps used to solve each optimization
                problem. Defaults to 200.
            initial_mu: The initial penalty weight in the augmented Lagrangian.
                Defaults to 10.
            tol: The tolerance for violations of the theta_r = 0 constraint.
                When |theta_r| < tol, we terminate the optimization.
                Unfortunately, if the tolerance is too small, numerical
                instability can occur. Defaults to 0.01.
        Returns:
            weights: a tensor of weights that rationalizes the behavior
        TODO(chanlaw): better understand the numerical instability that occurs
                when the tolerance is too small.
        TODO(chanlaw): function is too long (80+ lines) and should be split up
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.unnorm_weights.assign(self.initial_weights)
        self.theta_r.assign(initial_theta_r)
        theta_r = self.theta_r

        mu = tf.constant(initial_mu)
        lm = tf.constant(0.0)

        # first, increase initial theta_r until Hessian has negative determinant
        logger.info(
            "building loss function and "
            "initializing theta_r = {:.3f}".format(initial_theta_r)
        )
        _, sign = self.compute_total_augmented_loss(
            self.weights,
            trajectory=trajectory,
            theta_r=self.theta_r,
            mu=mu,
            lm=lm,
        )
        while tf.less(sign, 0):
            theta_r.assign(theta_r * 2)
            logger.info("\tdoubling theta_r to {:.3f}".format(theta_r.numpy()))
            _, sign = self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=self.theta_r,
                mu=mu,
                lm=lm,
            )

        logger.info("finished initializing theta_r")
        theta_r_val = theta_r.numpy()

        # next, optimize the CIOC loss using the augmented lagrangian method
        logger.info("initializing optimizer and optimizing weights and theta_r")

        def loss_fn():
            return self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=theta_r,
                mu=mu,
                lm=lm,
            )[0]

        @tf.function
        def train_op():
            for _ in range(n_iter):
                self.optimizer.minimize(loss_fn,
                                           [self.unnorm_weights, theta_r])

        train_op()

        while abs(theta_r.numpy()) > tol:
            logger.info(
                "|theta_r| = {:.3f} > {:.3f}".format(abs(theta_r.numpy()), tol)
                + ", so augmenting the Lagrange multiplier and trying again"
            )
            lm = lm - mu * theta_r

            if abs(theta_r.numpy()) - abs(theta_r_val) >= -5e-4:
                mu *= 10
                logger.info(
                    "\ttheta_r didn't decrease, so penalty "
                    "increased to to {:.2f}".format(mu.numpy())
                )
            theta_r_val = theta_r.numpy()

            train_op()

        logger.info(
            "|theta_r| = {:.3f} <= {:.3f}, ".format(abs(theta_r.numpy()), tol)
            + "so returning weights"
        )

        _, sign = self.compute_total_augmented_loss(
            self.weights, trajectory=trajectory, theta_r=theta_r, mu=mu, lm=lm
        )
        if sign < 0:
            logger.warning(
                "Negative Hessian is ill-conditioned (ie, |-H| < 0)"
                " - results may be nonsensical - "
                "try increasing the tolerance and rerunning."
            )
        return self.weights

    @tf.function
    def rationalize_tf(
        self,
        trajectory: List[Tuple],
        initial_theta_r: Union[float, tf.Tensor] = 0.01,
        n_iter: int = 200,
        initial_mu: float = 10.0,
        tol: float = 0.01,
    ) -> tf.Tensor:
        """
        Rationalizes the trajectory using the augmented Lagrangian method
        described in section 6.1 of the Levine and Kolton (2012).
        Unlike the rationalize method, this has been decorated with a
        `@tf.function` decorator.
        We use the standard first-order (gradient-based) method to solve each
        optimization problem.
        Note: this function takes a long time to run, so it logs a lot of info
        along the way. Set logger level to info or lower to see the progress.
        Args:
            trajectory: a list of (state, controls) tuples, where state is
                a list of car states and control is a list of controls.
            initial_theta_r: the initial value of theta_r. Defaults to 0.01.
            n_iter: the number of gradient steps used to solve each optimization
                problem. Defaults to 200.
            initial_mu: The initial penalty weight in the augmented Lagrangian.
                Defaults to 10.
            tol: The tolerance for violations of the theta_r = 0 constraint.
                When |theta_r| < tol, we terminate the optimization.
                Unfortunately, if the tolerance is too small, numerical
                instability can occur. Defaults to 0.01.
        Returns:
            weights: a tensor of weights that rationalizes the behavior
        TODO(chanlaw): better understand the numerical instability that occurs
                when the tolerance is too small.
        TODO(chanlaw): function is too long (80+ lines) and should be split up
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.unnorm_weights.assign(self.initial_weights)
        self.theta_r.assign(initial_theta_r)
        theta_r = self.theta_r

        mu = tf.constant(initial_mu)
        lm = tf.constant(0.0)

        # first, increase initial theta_r until Hessian has negative determinant
        logger.info(
            "Building loss function."
        )
        _, sign = self.compute_total_augmented_loss(
            self.weights,
            trajectory=trajectory,
            theta_r=self.theta_r,
            mu=mu,
            lm=lm,
        )

        # tf.print(ll)
        while tf.less(sign, 0):
            theta_r.assign(theta_r * 2)
            tf.print("\tDoubling theta_r to {}".format(theta_r))

            _, sign = self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=self.theta_r,
                mu=mu,
                lm=lm,
            )

        tf.print("Theta_r initialized.")

        theta_r_val = tf.identity(theta_r)  # store the current value of theta_r

        # next, optimize the CIOC loss using the augmented lagrangian method
        logger.info("initializing optimizer and optimizing weights and theta_r")

        def loss_fn():
            return self.compute_total_augmented_loss(
                self.weights,
                trajectory=trajectory,
                theta_r=theta_r,
                mu=mu,
                lm=lm,
            )[0]

        print('right before loop')
        _, sign = self.compute_total_augmented_loss(
            self.weights,
            trajectory=trajectory,
            theta_r=self.theta_r,
            mu=mu,
            lm=lm,
        )
        print('done right before loop')
        for i in range(n_iter):
            self.optimizer.minimize(loss_fn, [self.unnorm_weights, theta_r])

        # while theta_r > tol:
        #     tf.print(
        #         "|theta_r| = {} > {}".format(theta_r, tol)
        #         + ", so augmenting the Lagrange multiplier and trying again"
        #     )
        #     lm = lm - mu * theta_r
        #
        #     if tf.abs(theta_r) - tf.abs(theta_r_val) >= -5e-4:
        #         # TODO(chanlaw): this is not the correct calculation
        #         mu *= 10
        #         logger.info(
        #             "\ttheta_r didn't decrease, so penalty "
        #             "increased to to {}".format(mu)
        #         )
        #     theta_r_val = tf.identity(theta_r)
        #
        #     for _ in range(n_iter):
        #         self.optimizer.minimize(loss_fn, [self.unnorm_weights, theta_r])
        #
        # logger.info(
        #     "|theta_r| = {} <= {}, ".format(abs(theta_r), tol)
        #     + "so returning weights"
        # )
        #
        # _, sign = self.compute_total_augmented_loss(
        #     self.weights, trajectory=trajectory, theta_r=theta_r, mu=mu, lm=lm
        # )
        # if sign < 0:
        #     logger.warning(
        #         "Negative Hessian is ill-conditioned (ie, |-H| < 0)"
        #         " - results may be nonsensical - "
        #         "try increasing the tolerance and rerunning."
        #     )
        return tf.identity(self.weights)

    def rationalize_trajectories(
        self, trajectories: Collection[List[Tuple]], n_iter: int = 200,
    ) -> tf.Tensor:
        # TODO(chanlaw): implement this
        raise NotImplementedError

# class LocalLinearCIOC(InverseOptimalControl):s
#     """
#     Implements Levine et al (2012)'s
#     "Continuous IOC with Locally Optimal Examples".
#
#     When the reward function is a linear function of some set of features
#         r(x, u) = w * phi(x, u),
#     we can use something less trivial
#
#     Computes the MLE weights for the trajectory, assuming the demonstrator is
#     Boltzmann-rational.
#     """
#
#     @tf.function
#     def segment_gradient(self, weights: tf.Variable,
#                          initial_state: Union[tf.Variable, tf.Tensor],
#                          controls: Collection[Union[tf.Tensor, tf.Variable]],
#                          index: Union[None, int] = None) -> tf.Tensor:
#         with tf.GradientTape() as t:
#             t.watch(controls)
#             r = self.car.planner.reward_func(initial_state, controls, weights)
#
#         if index is None:
#             return tf.stack(t.gradient(r, controls), axis=-1)
#         else:
#             return t.gradient(r, controls[index])
#
#     @tf.function
#     def compute_total_loss(self, weights: tf.Variable,
#                            trajectory: List[Tuple],
#                            eta: tf.Tensor) -> tf.Tensor:
#
#         with tf.GradientTape() as t:
#             # compute gradients
#             gradients = 0
#         # compute Hessian
#         hessian = t.jacobian(gradients, controls)
#
#         pass
#
#     def rationalize(self, trajectory: Collection[Tuple]):
#         pass