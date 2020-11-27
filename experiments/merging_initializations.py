import numpy as np
import tensorflow as tf
from typing import List, Union
from sacred import Experiment

from merging import ThreeLaneTestCar
from interact_drive.world import ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar
from interact_drive.reward_design import InverseLocallyOptimalControl, LocalCIOC


ex = Experiment("CIOC-ILOC-merging-initializations")
ex.logger = tf.get_logger()

ex.logger.setLevel(20)

@ex.config
def merging_initializations_config():
    debug_level: int = 20
    initial_thetas: List[np.ndarray] = [np.random.normal(size=(6,)).astype(np.float32) for i in range(15)]
    weight_norm = 100
    horizon = 5
    traj_len = 15

@ex.main
def merging_cioc_beta_experiment(
    debug_level: int,
    initial_thetas: List[np.ndarray],
    weight_norm: float,
    traj_len: int,
    horizon: int,
    _log
):
    _log.info('Initializing World ...')

    world = ThreeLaneCarWorld(visualizer_args=dict(name="Merging"))
    # features are [velocity, lane_0_dist, lane_1_dist, lane_2_dist, min_lane_dist, collision]
    true_weights = np.array([0.1, 0., 0., -100., -1., -10])
    our_car = ThreeLaneTestCar(world, np.array([0, -0.5, 0.8, np.pi / 2]),
                               horizon=5,
                               weights=true_weights)
    other_car_1 = FixedVelocityCar(world, np.array([0.1, -0.7, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    other_car_2 = FixedVelocityCar(world, np.array([0.1, -0.2, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    world.add_cars([our_car, other_car_1, other_car_2])


    world.reset()
    trajectory = []

    for i in range(15):
        past_state, controls, state = world.step()
        trajectory.append((past_state, controls))

    cioc_thetas = []
    for initial_theta in initial_thetas:
        cioc = LocalCIOC(our_car, initial_weights=initial_theta)
        # iloc = InverseLocallyOptimalControl(our_car, initial_weights=initial_theta)

        _log.info("rationalizing with initial_theta={}".format(initial_theta))
        theta_cioc = cioc.rationalize(trajectory=trajectory).numpy()
        _log.info(
            "initial_theta={}, theta_cioc={}".format(
                initial_theta, theta_cioc
            )
        )
        cioc_thetas.append(theta_cioc)

    norm_cioc_thetas = [theta/np.linalg.norm(theta) for theta in cioc_thetas]
    norm_true_weights = true_weights/np.linalg.norm(true_weights)
    _log.info(norm_true_weights)
    _log.info(norm_cioc_thetas)




if __name__ == '__main__':
	ex.run_commandline()
