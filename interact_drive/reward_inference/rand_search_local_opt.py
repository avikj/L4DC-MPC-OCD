from interact_drive.reward_inference.bord_local_opt_scenario import local_opt_env
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign
import time

car, world = local_opt_env(init_lsr=False)
designer_weights = car.weights

NUM_INITS = 3

def fmt(arr):
	s = str(arr).replace("\n", ' ').replace('\t', " ")
	while '  ' in s:
		s = s.replace('  ', ' ')
	return s

init_states = list(np.linspace(car.init_state-[0., 0.1, 0., 0.], car.init_state+[0., 0.1, 0., 0.], NUM_INITS))

bord = BayesOptRewardDesign(world, car, init_states, 
	15, save_path=f'local_opt_rand_search_results_validated_{fmt(car.weights)}__{time.time()}_single_init.pkl')

bord.optimize_random_search(n_iter=1000)