from merging import ThreeLaneTestCar
from dont_crash import setup_world, is_in_right_lane, reaches_target_speed, doesnt_collide
import numpy as np
from bayes_opt import BayesianOptimization
import pickle

def main():
	SAVE_FILENAME = 'bayesopt_results_traj_collisions_save_traj.pkl'
	DESIGNER_WEIGHTS = np.array([-1, 0,0,0,-1, -50, -5], dtype=np.float32)
	DESIGNER_WEIGHTS /= np.linalg.norm(DESIGNER_WEIGHTS)
	
	our_car, other_car, world = setup_world()

	results = []
	pbounds = {'w0': (-1, 1), 'w1': (-1, 1), 'w2': (-1, 1), 'w3': (-1, 1), 'w4': (-1, 1), 'w5': (-1, 1), 'w6': (-1, 1)}
	def eval_weights(w0, w1, w2, w3, w4, w5, w6):
		"""Simulates planning car with these weights in the world and returns reward accumulated against designer weights"""
		our_car.weights = np.array([w0, w1, w2, w3, w4, w5, w6], dtype=np.float32)
		print("Evaluating weights:", str(our_car.weights))


		result = {}
		score = 0
		for init_x in [0.0,0.2/3]:
			our_car.init_state = np.array([init_x, our_car.init_state[1], our_car.init_state[2], our_car.init_state[3]], dtype=np.float32)
			world.reset()
			init_score = 0
			for i in range(15):
				_, controls, state = world.step()
				step_reward = our_car.reward_fn(state, controls[our_car.index], weights=DESIGNER_WEIGHTS)
				init_score += step_reward
				# world.render()
			result[init_x] = init_score
			score += init_score


		print('weights:', our_car.weights, 'result:', result, 'score:',score)
		results.append({'weights': our_car.weights, 'result': result, 'score': score, 'traj': our_car.past_traj})

		with open(SAVE_FILENAME, 'wb') as outfile:
			pickle.dump(results, outfile)

		return score

	optimizer = BayesianOptimization(f=eval_weights, pbounds=pbounds, random_state=0)
	optimizer.probe(params={'w0': DESIGNER_WEIGHTS[0], 'w1': DESIGNER_WEIGHTS[1], 'w2': DESIGNER_WEIGHTS[2], 'w3': DESIGNER_WEIGHTS[3], 'w4': DESIGNER_WEIGHTS[4], 'w5': DESIGNER_WEIGHTS[5], 'w6': DESIGNER_WEIGHTS[6]})
	optimizer.maximize(init_points=100, n_iter=1000)
	print(optimizer.max)



main()







