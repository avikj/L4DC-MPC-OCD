from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign
from interact_drive.reward_inference.bord_local_opt_scenario import local_opt_env
import numpy as np
import time

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	car, world = local_opt_env(debug=True, extra_inits=False)
	bord = BayesOptRewardDesign(world, car, 
		[car.init_state],15, save_path=f'cmaes_finite_horizon_designer_weights_{fmt(car.weights)}.pkl',)

	bord.optimize_cmaes(seed=1, sigma0=0.2)
