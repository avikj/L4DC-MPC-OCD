# Demonstrates the effect of short planning horizon on Scenario 1; increasing the horizon from 5 to 6 results in preferred
# lane switching behavior.

from interact_drive.reward_design.mpc_ord import finite_horizon_env, MPC_ORD
import numpy as np

NUM_INITS = 3

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s

	car, world = finite_horizon_env(horizon=6, extra_inits=False)
	init_states = np.linspace(car.state-np.array([0.1,0,0,0]), car.state+np.array([0.1,0,0,0]), NUM_INITS)

	bord = MPC_ORD(world, car,
				   init_states, 15, save_path=None, )

	bord.eval_weights(car.weights, gif=f'finite_horizon_designer_weights_{fmt(car.weights)}_horizon_6_200_iter_{NUM_INITS}_inits_{"single" if car.debug else "anim"}.gif')
	
	car, world = finite_horizon_env(horizon=5, extra_inits=False)
	bord = MPC_ORD(world, car,
				   init_states, 15, save_path=None, )
	bord.eval_weights(car.weights, gif=f'finite_horizon_designer_weights_{fmt(car.weights)}_horizon_5_100_iter_{NUM_INITS}_inits_{"single" if car.debug else "anim"}.gif')