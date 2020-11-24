from typing import Optional, Tuple, List

import contextlib
import numpy as np
import tensorflow as tf

from interact_drive.world import TwoLaneCarWorld
from interact_drive.car import FixedVelocityCar, FixedPlanCar
from experiments.merging import ThreeLaneTestCar
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign
import scipy 
class ReplanningCarWorld(TwoLaneCarWorld):
    """
    One of the other cars disappears at a fixed timestep.

    "Disappearing" is implemented via the hack of teleporting the car to really, really far away.
    """

    def __init__(self, dt=0.1, critical_t=4, **kwargs):
        super().__init__(dt=dt, **kwargs)
        self.critical_t = critical_t
        self.unlucky_car_idx = 1
        self.t = 0

    def reset(self):
        super().reset()
        self.unlucky_car_idx = 2 if self.unlucky_car_idx == 1 else 1
        self.t = 0

    def step(self, dt: Optional[float] = None) -> Tuple[List[tf.Tensor],
                                                        List[tf.Tensor],
                                                        List[tf.Tensor]]:
        self.t += 1
        if self.t == self.critical_t:
            self.cars[self.unlucky_car_idx].state = tf.constant([10., 0., 0., 0.], dtype=tf.float32)

        return super().step()

og_weights = np.array([-3, 0, 0, -2, -10, -10], dtype=np.float32)
og_weights /= np.linalg.norm(og_weights)
# these tuned lane weights are designed such that we always want to be closer to the further lane; 
# the +0.25 any lane weight cancels with the cost from the closer of the two lanes, leaving only the further lane cost to minimize
tuned_weights = np.array( [-0.55899817, -0.4436692 , -0.37245109, -0.19964276, -0.5438697 ,
        0.12770044], dtype=np.float32) #np.array([[-0.71931692, -0.18519479, -0.30940993, -0.08956627, -0.54680025 , 0.21339851]])# 
tuned_weights /= np.linalg.norm(tuned_weights)
def setup_world(env_seeds=[1], debug=True):

    ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE = -0.0, 0.02, (-0.005, 0.005) # clip gaussian at bounds
    ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE = -0.9, 0.04, (-1., -0.8) # clip gaussian at bounds
    ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE = 1.0, 0.05, (0.8, 1.2)


    def get_init_state(env_seed):
        rng = np.random.RandomState(env_seed)
        np.random.seed(seed=env_seed)
        def sample(mean, std, rang):
            a, b = (rang[0] - mean) / std, (rang[1] - mean) / std
            return np.squeeze(scipy.stats.truncnorm.rvs(a,b)*std+mean)

        robot_x = sample(ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE)
        robot_y = sample(ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE)
        robot_init_speed = sample(ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE)
        return np.array([robot_x, robot_y, robot_init_speed, np.pi / 2])

    init_states = [get_init_state(s) for s in env_seeds]


    world = ReplanningCarWorld()
    weights = og_weights
    our_car = ThreeLaneTestCar(world, init_states[0],
                               horizon=5,
                               weights=weights,
                               target_speed=1.2, planner_args={'n_iter': 100},
                               check_plans=True,
                               num_lanes=2,
                               debug=debug
                               )
    # our_car = FixedVelocityCar(world, tf.constant([0.0, -0.9, 1.5, np.pi / 2], dtype=tf.float32))
    other_car_1 = FixedPlanCar(world, np.array([0., -0.7, 0.8, np.pi / 2]),
                               plan=[np.array([0., 0.], dtype=np.float32)] + [
                                   np.array([0.7, 2.7], dtype=np.float32)] + [
                                        np.array([0., 0.], dtype=np.float32)] + [
                                        np.array([0.0, -2.7], dtype=np.float32)],
                               default_control=np.array([0.0, 0.0], dtype=np.float32),
                               horizon=5, color='gray', opacity=0.8, debug=debug)
    other_car_2 = FixedPlanCar(world, np.array([0., -0.7, 0.8, np.pi / 2]),
                               plan=[np.array([0., 0.], dtype=np.float32)] + [
                                   np.array([0.7, -2.7], dtype=np.float32)] + [
                                        np.array([0., 0.], dtype=np.float32)] + [
                                        np.array([0.0, 2.7], dtype=np.float32)],
                               default_control=np.array([0.0, 0.0], dtype=np.float32),
                               horizon=5, color='gray', opacity=0.8, debug=debug)
    world.add_cars([our_car, other_car_1, other_car_2])
    world.reset()

    return our_car, world, init_states

def fmt(arr):
    s = str(arr).replace("\n", ' ').replace('\t', " ")
    while '  ' in s:
        s = s.replace('  ', ' ')
    return s

def main():
    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip

    for use_tuned in [True, False]:
        if use_tuned:
            print("evaluating using MPC + tuned weights")
        else:
            print("evaluating using MPC + true weights")

        our_car, world = setup_world()


        bord = BayesOptRewardDesign(world, our_car, [our_car.init_state], 20, save_path=None, num_samples=2)
        reward = bord.eval_weights(tuned_weights if use_tuned else our_car.weights, gif=f'replanning_designer_{fmt(our_car.weights)}__agent_{fmt(tuned_weights if use_tuned else our_car.weights)}.gif')
        print("reward", reward)

        """for unlucky_car_idx in [1,2]:
            
            reward = 0.0
            for i in range(20):
                reward += our_car.reward_fn(
                    world.state, control=our_car.control, weights=og_weights
                )
                print("[{}] velocity {} cum reward {}".format(i, our_car.state[2].numpy(), reward))
                world.step()
                world.render()
                frames.append(world.render("rgb_array"))

            clip = ImageSequenceClip(frames, fps=int(1 / world.dt))

            if use_tuned:
                clip.speedx(0.5).write_gif("replanning_carpool{}_tuned_{}.gif".format(world.unlucky_car_idx, fmt(tuned_weights)), program="ffmpeg")
            else:
                clip.speedx(0.5).write_gif("replanning_carpool{}_{}.gif".format(world.unlucky_car_idx, fmt(og_weights)), program="ffmpeg")
            """
##
if __name__ == "__main__":
    main()
