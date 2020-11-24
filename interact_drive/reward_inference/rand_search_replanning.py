from experiments.planner_world import setup_world as setup_replanning_world
from interact_drive.reward_inference.bayesopt_rd import BayesOptRewardDesign

if __name__ == '__main__':
    our_car, world = setup_replanning_world()

    bord = BayesOptRewardDesign(world, our_car, [our_car.init_state], 20, save_path=None)
    bord.optimize_random_search(n_iter=1000, num_samples=2, save_path='rand_search_replanning_1_init.pkl')