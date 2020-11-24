from merging import ThreeLaneTestCar
from dont_crash import setup_world, is_in_right_lane, reaches_target_speed, doesnt_collide
import numpy as np
import contextlib
from bayes_opt import BayesianOptimization
import pickle
import tensorflow as tf
from interact_drive.reward_inference.coarse_value_iteration import ValueIteration
from interact_drive.reward_inference.value_interpolation import ValueFeature
from interact_drive.car.planner_car import PlannerCar
def main():
    SAVE_FILENAME = 'bayesopt_results_traj_collisions_save_traj.pkl'
    DESIGNER_WEIGHTS = np.array([-1, 0, 0, 0, -1, -50, -5], dtype=np.float32)
    DESIGNER_WEIGHTS /= np.linalg.norm(DESIGNER_WEIGHTS)

    our_car, other_car, world = setup_world()
    vi = ValueIteration(world, our_car, filename='dont_crash_values.pkl')
    data = pickle.load(open('dont_crash_values.pkl', 'rb'))
    data['disc_grid'] = vi.state_disc_grid
    # use a slightly different initialization because original init is too close to VI boundaries
    our_car.init_state = np.array([0.01, -0.4, 0.9, np.pi / 2], dtype=np.float32)
    our_car.horizon = 2

    other_init_y = other_car.init_state[1]
    world.main_car = our_car
    @tf.function
    def proj(states):

        return tf.identity([states[0][0], states[0][1]-states[1][1]+other_init_y, states[0][2]])#+(-states[1][:3])*np.array([0., 1., 0.])+other_car.init_state[1]*np.array([0., 1., 0.])
        # return tf.identity([states[0][0], states[0][1]-states[1][1]+other_car.init_state[1], states[0][2]])
    val = ValueFeature(proj=proj,
                 value_data=data)

    interpolate_val = val.interpolate_value(t=0)
    @tf.function
    def leaf_value(world_state, controls):
        return interpolate_val(world_state)

    # our_car.reward_fn = leaf_value
    our_car.planner_args['leaf_evaluation'] = leaf_value
    our_car.initialize_planner(our_car.planner_args)

    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip
    frames = []
    world.reset()
    print("INIT world state", world.state)
    # world.visualizer.set_main_car(index=0)

    # Render trajectory using interpolated value function as leaf evaluation
    for i in range(15):
        _, controls, state = world.step()
        print(controls[0])
        world.render()
        frames.append(world.render("rgb_array", heatmap_show=True))

    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.8).write_gif("dont_crash_leaf_evaluation.gif", program="ffmpeg")

    """traj_inds = [vi.main_car_init_index]
    actions = []
    for i in range(15):
        actions.append(data['policy_grids'][i][traj_inds[-1][0]][traj_inds[-1][1]][traj_inds[-1][2]])
        traj_inds.append(data['next_state_inds'][i][traj_inds[-1][0]][traj_inds[-1][1]][traj_inds[-1][2]])
    coarse_traj = [vi.coarse_state_dict[tuple(ind)] for ind in traj_inds]
    traj = [np.array([cs[0], cs[1], cs[2], np.pi/2], dtype=np.float32) for cs in coarse_traj]
    """
    """for ca in vi.coarse_action_values:
        ns = vi._next_coarse_state(vi.main_car_init_state[:-1], ca)
        print("action", ca, "results in state", ns, "with value", data['v_grids'][1][vi._round_to_grid(ns)])"""
    """
    
    ## Render coarse optimal trajectory
    frames = []
    world.reset()
    for i in range(15):
        world.step()
        our_car.state = tf.constant(traj[i])
        print('ind', traj_inds[i], 'state', traj[i], 'action', actions[i], 'next state', vi._next_coarse_state(coarse_traj[i], actions[i]))
        frames.append(world.render("rgb_array"))

    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.8).write_gif("dont_crash_coarse_optimal.gif", program="ffmpeg")
    print(traj)"""



if __name__ == '__main__':
    main()

