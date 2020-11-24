from experiments.merging import ThreeLaneCarWorld, ThreeLaneTestCar
from interact_drive.car import FixedVelocityCar
import numpy as np
import tensorflow as tf
import robo.fmin
import time
import pickle
from multiprocessing import Pool
import cma
import scipy

class list2(list):
	def __init__(self, *args, **kwargs):
		super().__init__(self, *args, **kwargs)


class BayesOptRewardDesign:
	def __init__(self, world, car, init_car_states, designer_horizon, save_path=None, num_samples=1):
		self.world = world
		self.car = car

		self.designer_horizon = designer_horizon
		self.init_car_states = init_car_states
		self.save_path = save_path

		self.designer_weights = car.weights/np.linalg.norm(car.weights)
		self.weight_dim = len(car.weights)

		self.history = list2()
		self.iter = 0
		self.should_save_history = False
		self.done = False
		self.num_samples = num_samples

	def optimize(self, n_iter=1000, seed=1):
		assert not self.done
		self.history.seed = seed

		self.should_save_history = True
		print("\n\nSTARTING ENTROPY SEARCH")
		self.iter = 0


		# prime bayesopt by evaluating designer weights first
		X_init=[self.designer_weights]
		Y_init = [self.eval_weights(xi) for xi in X_init]

		res = robo.fmin.entropy_search(self.eval_weights, -np.ones(self.weight_dim), np.ones(self.weight_dim), 
			num_iterations=n_iter, X_init=np.array(X_init), Y_init=np.array(Y_init), rng=np.random.RandomState(seed))['x_opt']
		self.should_save_history = False
		self.done=True
		return res

	def optimize_cmaes(self, seed=1, sigma0=0.1):
		self.history.seed = seed
		assert seed != 0
		assert not self.done
		self.should_save_history = True

		self.eval_weights(self.designer_weights)

		x, es = cma.evolution_strategy.fmin2(self.eval_weights, list(self.designer_weights), sigma0=sigma0, options={'seed': seed})

		self.should_save_history = False
		self.done = True

	def optimize_random_search(self, n_iter=1000, seed=1):
		self.history.seed = seed
		assert not self.done
		self.should_save_history = True
		print("\n\nSTARTING RANDOM SEARCH")
		self.iter = 0


		# print designer results for reference
		self.eval_weights(self.designer_weights)

		np.random.seed(seed)

		for _ in range(n_iter):
			weights = np.random.rand(*self.designer_weights.shape)*2-1
			self.eval_weights(weights)
		self.should_save_history = False
		self.done = True
		return max(self.history, key=lambda a: a[1])

	def eval_weights_for_init(self, init, weights, render, heatmap_show=False):
		# evaluates weights for a single initialization
		if weights.ndim ==  2:
			weights = weights[0] # reshape 2d array with one row into 1d array
		weights = weights/np.linalg.norm(weights)

		if self.car.debug:
			for car in self.world.cars:
				assert car.debug

		frames = []
		def maybe_render():
			if render:
				frames.append(self.world.render("rgb_array", heatmap_show=heatmap_show))
		

		self.car.weights = weights
		self.car.init_state = tf.constant(init, dtype=tf.float32)

		designer_reward = 0
		for _ in range(self.num_samples):

			self.world.reset()
			#if not self.car.debug:
			maybe_render()

			sample_reward = 0

			for i in range(self.designer_horizon):
				past_state, controls, state = self.world.step()
				#if not self.car.debug:
				maybe_render()
				sample_reward += self.car.reward_fn(past_state, controls[self.car.index], weights=self.designer_weights)
			#if self.car.debug:
			#	maybe_render()
			designer_reward += sample_reward
			print('sample_reward',sample_reward)
		designer_reward = designer_reward.numpy()
		print('init', init, 'weights', weights ,'\tgave return:',designer_reward, 'time',time.time())
		return (designer_reward, frames) if render else designer_reward


	def eval_weights(self, weights, gif=None, heatmap_show=False):
		if gif:
			import contextlib
			with contextlib.redirect_stdout(None):  # disable the pygame import print
				from moviepy.editor import ImageSequenceClip

		if isinstance(weights, list):
			weights = np.array(weights)

		if weights.ndim == 2:
			weights = weights[0] # reshape 2d array with one row into 1d array
		weights = weights/np.linalg.norm(weights)
		
		print('ITERATION', self.iter)
		print('eval',weights)


		total_designer_reward = 0
		frames = []
		for init_car_state in self.init_car_states:

			r = self.eval_weights_for_init(init_car_state, weights, gif is not None, heatmap_show=heatmap_show)
			if gif:
				r, frames_from_init = r
				frames.extend(frames_from_init)

			if len(self.init_car_states) > 1:
				print('\tinit_reward',r)
			total_designer_reward += r

		total_designer_reward /= self.num_samples
		print('eval reward for weights:',total_designer_reward,'\n\n')

		if gif:
			clip = ImageSequenceClip(frames, fps=int(1 / self.world.dt))
			clip.speedx(0.5).write_gif(gif, program="ffmpeg")

		self.history.append((weights, total_designer_reward))
		if self.should_save_history and self.save_path is not None:
			self.save_history()
		self.iter += 1

		return -total_designer_reward  # minus since we're minimizing the cost

	def save_history(self):
		assert self.save_path is not None
		with open(self.save_path, 'wb') as file:
			pickle.dump(self.history, file)

		print('Wrote results so far to',self.save_path)



def finite_horizon_env(horizon=5, env_seeds=[1], debug=True, extra_inits=False):

	"""
	Runs a planning car in a merging scenario for 15 steps and visualizes it.
	The weights of our planning car should cause it to merge into the right lane
	between the two other cars.
	"""


	ROBOT_X_MEAN, ROBOT_X_STD, ROBOT_X_RANGE = 0, 0.04, (-0.1, 0.1) # clip gaussian at bounds
	ROBOT_Y_MEAN, ROBOT_Y_STD, ROBOT_Y_RANGE = -0.9, 0.02, (-0.95, -0.85) # clip gaussian at bounds
	ROBOT_SPEED_MEAN, ROBOT_SPEED_STD, ROBOT_SPEED_RANGE = 0.8, 0.03, (0.7, 0.9)
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

	our_car = ThreeLaneTestCar(
		world,
		init_state=init_states[0], #,np.array([0, -0.5, 0.8, np.pi / 2]),
		horizon=horizon,
		weights=np.array([-5, 0., 0., 0., -6., -50, -50]),
		debug=debug,
		planner_args=dict(n_iter=200 if horizon==6 else 100, extra_inits=extra_inits)
	)
	other_car= FixedVelocityCar(
		world,
		np.array([0, -0.6, 0.5, np.pi / 2]),
		horizon=horizon,
		color="gray",
		opacity=0.8,
		debug=debug,
		planner_args=dict(n_iter=200 if horizon==6 else 100, extra_inits=extra_inits)
	)
	world.add_cars([our_car, other_car])

	world.reset()

	return our_car, world, init_states

DEBUG=True

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	car, world, init_states = finite_horizon_env(horizon=5, env_seeds=[0])
	import sys;sys.exit(1)
	bord = BayesOptRewardDesign(world, car, 
		init_states,15, save_path=f'sl_bopt_results__{fmt(car.weights)}__{time.time()}.pkl',)

	bord.optimize(n_iter=1000)# bord.eval_weights(car.weights, gif=f'finite_horizon_designer_weights_{fmt(car.weights)}_horizon_5_100_iter_single_init_{"single" if car.debug else "anim"}.gif')
	"""
	car, world = finite_horizon_env(horizon=6)
	bord = BayesOptRewardDesign(world, car, 
		[car.init_state, car.init_state+np.array([0.05,0,0,0]), car.init_state+np.array([-0.05,0,0,0])],15, save_path=f'sl_bopt_results__{fmt(car.weights)}__{time.time()}.pkl',)

	bord.eval_weights(car.weights, gif=f'finite_horizon_designer_weights_{fmt(car.weights)}_horizon_6.gif')"""
	SHOW_RESULTS = False

	#if not SHOW_RESULTS:
	#	bord.optimize()
	#else:
	#
	#	history = pickle.load(open('sl_bopt_results__[-1.0.0.0.-10.-50.-50.]__1603191100.287489.pkl', 'rb'))
	#	history = history[1:] # history[0] is designer weights entry, history[1] is normed designer weights entry
	#	better_entries = [e for e in history if e[1] >= history[0][1]]

	#	for weight, reward in sorted(better_entries, key=lambda tup:-tup[1]):
	#		print(weight, reward)

	#		bord.eval_weights(np.array(weight), gif=f'finite_horizon_bopt_designer_{fmt(designer_weights)}__agent_{fmt(weight)}__reward_{reward}.gif')
