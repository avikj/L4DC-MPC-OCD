from interact_drive.reward_inference.bayesopt_rd import finite_horizon_env, BayesOptRewardDesign
import numpy as np
import time

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	car, world, init_states = finite_horizon_env(horizon=5, env_seeds=[0,1])
	bord = BayesOptRewardDesign(world, car, 
		init_states, 15, save_path=f'cmaes_finite_horizon_designer_weights_{fmt(car.weights)}.pkl',)

	bord.optimize_cmaes(seed=1, sigma0=0.2)
