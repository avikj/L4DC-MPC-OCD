
def evaluate_weights(car, agent_weights, world, horizon=15, init_states=None):
    designer_weights = car.weights
    designer_reward_fn = car.reward_fn
    car.weights = agent_weights
    car.initialize_planner(car.planner_args)
    reward = 0
    world.reset()
    for i in range(horizon):
        past_state, ctrl, state = world.step()
        reward += designer_reward_fn(past_state, ctrl[0])
    car.weights = designer_weights
    car.initialize_planner(car.planner_args)
    return reward
