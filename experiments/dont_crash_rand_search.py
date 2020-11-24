from merging import ThreeLaneTestCar
from dont_crash import setup_world, is_in_right_lane, reaches_target_speed, doesnt_collide
import numpy as np
from bayes_opt import BayesianOptimization
import pickle

def main():
	SAVE_FILENAME = 'rand_search_results_traj_collisions_save_traj.pkl'
	our_car, other_car, world = setup_world()

	results = []
	pbounds = {'w0': (-1, 1), 'w1': (-1, 1), 'w2': (-1, 1), 'w3': (-1, 1), 'w4': (-1, 1), 'w5': (-1, 1)}
	def eval_weights(w0, w1, w2, w3, w4, w5):
		"""Simulates planning car with these weights in the world and returns reward"""
		our_car.weights = np.array([w0, w1, w2, w3, w4, w5], dtype=np.float32)
		print("Evaluating weights:", str(our_car.weights))


		result = {}
		score = 0
		for init_x in [0.0, 0.1/3, 0.2/3, 0.1]:
			our_car.init_state = np.array([init_x, our_car.init_state[1], our_car.init_state[2], our_car.init_state[3]], dtype=np.float32)
			world.reset()
			for i in range(15):
				world.step()
				# world.render()
			init_result = {'is_in_right_lane': is_in_right_lane(our_car), 'reaches_target_speed': reaches_target_speed(our_car), 'doesnt_collide': doesnt_collide(our_car, other_car)}
			if init_result['is_in_right_lane']:
				score += 1
			if init_result['reaches_target_speed']:
				score += 1
			if init_result['doesnt_collide']:
				score += 1
			result[init_x] = init_result


		print('weights:', our_car.weights, 'result:', result, 'score:',score)
		results.append({'weights': our_car.weights, 'result': result, 'score': score, 'traj': our_car.past_traj})

		with open(SAVE_FILENAME, 'wb') as outfile:
			pickle.dump(results, outfile)

		return score

	"""
	optimizer = BayesianOptimization(f=eval_weights, pbounds=pbounds, random_state=0)
	optimizer.maximize(init_points=200, n_iter=1000)
	print(optimizer.max)
	"""

	for i in range(1200):
		weights = np.random.uniform(size=6)
		eval_weights(*weights)



main()







