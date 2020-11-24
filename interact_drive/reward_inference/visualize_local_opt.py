from interact_drive.reward_inference.bord_local_opt_scenario import local_opt_env
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign
import numpy as np
import pickle

def fmt(arr):
	s = str(arr).replace("\n", ' ').replace('\t', " ")
	while '  ' in s:
		s = s.replace('  ', ' ')
	return s
if __name__ == '__main__':
	results = pickle.load(open('local_opt_rand_search_results_validated_[-0.06984303 0. 0. -0.13968606 0. -0.6984303 -0.6984303 ]__1603954796.8140538_single_init.pkl', 'rb'))

	agent_weights = max(results, key=lambda a: a[1])[0]

	car, world = local_opt_env(extra_inits=False, debug=True)
	bord = BayesOptRewardDesign(world, car, [car.init_state], 
			15, save_path=None)
	bord.eval_weights(agent_weights, gif=f'local_opt_designer_weights_{fmt(car.weights)}_agent_weights_{fmt(agent_weights)}_single_init_{"single" if car.debug else "anim"}.gif')
