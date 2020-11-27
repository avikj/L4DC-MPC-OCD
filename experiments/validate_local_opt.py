# Demonstrates the effect of local optimization on Scenario 2; running gradient descent from more control initializations,
# triggered by the extra_inits flag, results in a trajectory which ends up in the right line, as desired.

from experiments.local_opt_scenario import local_opt_env
from interact_drive.reward_design.mpc_ord import MPC_ORD
import numpy as np

NUM_INITS = 3
def fmt(arr):
	s = str(arr).replace("\n", ' ').replace('\t', " ")
	while '  ' in s:
		s = s.replace('  ', ' ')
	return s
if __name__ == '__main__':
	car, world = local_opt_env(extra_inits=False, debug=True)
	init_states = list(np.linspace(car.init_state-[0., 0.1, 0., 0.], car.init_state+[0., 0.1, 0., 0.], NUM_INITS))
	bord = MPC_ORD(world, car, init_states,
				   15, save_path=None)

	bord.eval_weights(car.weights, gif=f'local_opt_designer_weights_{fmt(car.weights)}_no_lsr_single_init_{"single" if car.debug else "anim"}.gif')


	car, world = local_opt_env(extra_inits=True, debug=True)
	bord = MPC_ORD(world, car, init_states,
				   15, save_path=None)

	bord.eval_weights(car.weights, gif=f'local_opt_designer_weights_{fmt(car.weights)}_lsr_single_init_{"single" if car.debug else "anim"}.gif')
