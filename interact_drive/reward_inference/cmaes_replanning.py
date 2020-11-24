from experiments.planner_world import setup_world as setup_replanning_world
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign

if __name__ == '__main__':
	def fmt(arr):
		s = str(arr).replace("\n", ' ').replace('\t', " ")
		while '  ' in s:
			s = s.replace('  ', ' ')
		return s
	car, world = setup_replanning_world()

	bord = BayesOptRewardDesign(world, car, [car.init_state], 20, num_samples=2, save_path=f'cmaes_finite_horizon_designer_weights_{fmt(car.weights)}.pkl')
	bord.optimize_cmaes(seed=1, sigma0=0.2)