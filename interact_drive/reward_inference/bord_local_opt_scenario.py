from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign

from experiments.merging import ThreeLaneCarWorld, ThreeLaneTestCar
from interact_drive.car import FixedVelocityCar
import numpy as np
import tensorflow as tf
import time
import scipy

def local_opt_env(env_seeds=[1], extra_inits=False, debug=True):

	"""
	Runs a planning car in a merging scenario for 15 steps and visualizes it.
	The weights of our planning car should cause it to merge into the right lane
	between the two other cars.
	"""


	ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE = -0.1, 0.005, (-0.12, -0.08) # clip gaussian at bounds
	ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE = -0.9, 0.04, (-1., -0.8) # clip gaussian at bounds
	ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE = 1.0, 0.03, (0.9, 1.1)
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

	world = ThreeLaneCarWorld(visualizer_args=dict(name="Switch Lanes"))

	weights = np.array([-5, 0., 0., -10, 0, -50, -50])

	our_car = ThreeLaneTestCar(
		world,
		init_states[0],
		horizon=5,
		weights=weights/np.linalg.norm(weights),
		planner_args=dict(extra_inits=extra_inits),
		debug=debug
	)
	other_car= FixedVelocityCar(
		world,
		np.array([0, -0.9, 1., np.pi / 2]),
		horizon=5,
		color="gray",
		opacity=0.8,debug=debug
	)
	world.add_cars([our_car, other_car])

	world.reset()

	return our_car, world, init_states

if __name__ == '__main__':
	car, world = local_opt_env(extra_inits=False)
	designer_weights = car.weights

	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	SHOW_RESULTS = True
	bord = BayesOptRewardDesign(world, car, [car.init_state], 
		15, save_path=f'local_opt_bopt_results_validated_{fmt(car.weights)}__{time.time()}_single_init.pkl' if not SHOW_RESULTS else None)

	# bord.eval_weights(designer_weights, gif=f'local_opt_designer_weights_{fmt(car.weights)}_no_lsr_single_init_{"single" if car.debug else "anim"}.gif')
	bord.optimize(n_iter=1000)
	#car, world = local_opt_env(extra_inits=True)
	#bord = BayesOptRewardDesign(world, car, [car.init_state+np.array([0.,0.1,0,0])], 
	#	15, save_path=f'local_opt_bopt_results_-20_right_{fmt(car.weights)}__{time.time()}.pkl' if not SHOW_RESULTS else None)

	#bord.eval_weights(designer_weights, gif=f'local_opt_designer_weights_{fmt(car.weights)}_lsr_single_init_{"single" if car.debug else "anim"}.gif')
	

	# bord.eval_weights(designer_weights, gif='local_opt_heatmap_-40_right_no_lsr.gif')
	#if not SHOW_RESULTS:
	#	bord.optimize(n_iter=1000)
	#else:
	#	import pickle
	#	history = pickle.load(open('local_opt_bopt_results_-500_right_[-0.00680398 0. 0. -0.27215922 0. -0.68039805 -0.68039805]__1603757524.733558.pkl', 'rb'))
	#	better_entries = [e for e in history if e[1] >= history[0][1]]
	#
	#	for weight, reward in sorted(better_entries, key=lambda tup:-tup[1]):
	#		print(weight, reward)
	#		bord.eval_weights(np.array(weight), gif=f'local_opt_bopt_designer_{fmt(designer_weights)}__agent_{fmt(weight)}__reward_{reward}.gif')
