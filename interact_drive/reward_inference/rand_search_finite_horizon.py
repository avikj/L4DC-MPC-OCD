from interact_drive.reward_inference.bayesopt_rd import finite_horizon_env, BayesOptRewardDesign
import numpy as np
import time

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	car, world = finite_horizon_env(horizon=5)
	bord = BayesOptRewardDesign(world, car, 
		[car.init_state],15, save_path=f'rand_search_finite_horizon_designer_weights_{fmt(car.weights)}.pkl',)

	bord.optimize_random_search(n_iter=500, seed=0)
