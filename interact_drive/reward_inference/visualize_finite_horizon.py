from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign, finite_horizon_env
import numpy as np

def fmt(arr):
	s = str(arr).replace("\n", ' ').replace('\t', " ")
	while '  ' in s:
		s = s.replace('  ', ' ')
	return s
if __name__ == '__main__':
	car, world = finite_horizon_env(horizon=5, debug=True)
	bord = BayesOptRewardDesign(world, car, [car.init_state], 
			15, save_path=None)
[-5, 0., 0., 0., -6., -50, -50]
	agent_weights = np.array([-0.36774956 ,-0.4542278 ,  0.13472266,  0.03827454  ,0.0750249,  -0.65690708,
 -0.44907304])
	bord.eval_weights(agent_weights, gif=f'finite_horizon_designer_weights_{fmt(car.weights)}_agent_weights_{fmt(agent_weights)}_single_init_{"single" if car.debug else "anim"}.gif')
