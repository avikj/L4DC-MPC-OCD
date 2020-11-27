import numpy as np
import tensorflow as tf
from typing import List, Union
from sacred import Experiment

from merging import ThreeLaneTestCar
from interact_drive.world import ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar
from interact_drive.reward_design import InverseLocallyOptimalControl, LocalCIOC


ex = Experiment("CIOC-merging-beta")
ex.logger = tf.get_logger()

ex.logger.setLevel(20)

@ex.config
def merging_cioc_beta_config():
    debug_level: int = 20
    weight_norms: List[Union[float, int]] = [10.0 ** (i / 2) for i in range(10)]
    initial_theta: np.ndarray = np.ones((5,), dtype=np.float32)
    horizon = 5
    traj_len = 15

@ex.main
def merging_cioc_beta_experiment(
    debug_level: int,
    weight_norms: List[Union[float, int]],
    initial_theta: np.ndarray,
    traj_len: int,
    horizon: int,
    _log
):
    _log.info('Initializing World ...')

    world = ThreeLaneCarWorld(visualizer_args=dict(name="Merging"))
    # features are [velocity, lane_0_dist, lane_1_dist, lane_2_dist, min_lane_dist, collision]
    true_weights = np.array([-1, 0., 0., -100., -1.])#, -10])
    our_car = ThreeLaneTestCar(world, np.array([0, -0.5, 0.8, np.pi / 2]),
                               horizon=5,
                               weights=true_weights)
    other_car_1 = FixedVelocityCar(world, np.array([0.1, -0.7, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    other_car_2 = FixedVelocityCar(world, np.array([0.1, -0.2, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    world.add_cars([our_car])

    print("Simulating sample trajectory 1")
    world.reset()
    trajectory = [] 
    for i in range(15):
        past_state, controls, state = world.step()
        trajectory.append((past_state, controls))

    other_car_1.init_state =  tf.constant([0.1, -0.6, 0.8, np.pi / 2], dtype=tf.float32)
    other_car_2.init_state = tf.constant([0.1, -0.1, 0.8, np.pi / 2], dtype=tf.float32)
    print("Simulating sample trajectory 2")
    """world.reset()
    trajectory2 = []
    for i in range(15):
        past_state, controls, state = world.step()
        trajectory2.append((past_state, controls))"""
    """
    other_car_1.init_state = [0.1, -0.8, 0.8, np.pi / 2]
    other_car_2.init_state = [0.1, -0.3, 0.8, np.pi / 2]
    world.reset()
    trajectory3 = []
    for i in range(15):
        past_state, controls, state = world.step()
        trajectory3.append((past_state, controls))

    other_car_1.init_state = [0.1, -1.8, 0.8, np.pi / 2]
    other_car_2.init_state = [0.1, -1.3, 0.8, np.pi / 2]
    world.reset()
    trajectory4 = []
    for i in range(15):
        past_state, controls, state = world.step()
        trajectory4.append((past_state, controls))
    """
    cioc = LocalCIOC(our_car, initial_weights=initial_theta)
    # iloc = InverseLocallyOptimalControl(our_car, initial_weights=initial_theta)

    cioc_thetas = []
    for weight_norm in weight_norms:
        _log.info("rationalizing with weight_norm={}".format(weight_norm))
        weight_norm_tf = tf.constant(weight_norm, dtype=tf.float32)
        cioc.weight_norm = weight_norm_tf
        theta_cioc = cioc.rationalize_trajectories([trajectory]).numpy()
        _log.info(
            "weight_norm={}, theta_cioc={}".format(
                weight_norm, theta_cioc
            )
        )
        cioc_thetas.append(theta_cioc)

    norm_cioc_thetas = [theta/np.linalg.norm(theta) for theta in cioc_thetas]
    norm_true_weights = true_weights/np.linalg.norm(true_weights)
    _log.info(norm_true_weights)
    _log.info(norm_cioc_thetas)



if __name__ == '__main__':
	ex.run_commandline()
