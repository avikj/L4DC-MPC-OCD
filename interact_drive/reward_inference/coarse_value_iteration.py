import numpy as np
import bisect
from interact_drive.car import LinearRewardCar
import pickle
class ValueIteration:
    """
        Performs value iteration optimizing main_car's reward
        in a coarse approximation of a  given CarWorld. We
        use a coarser state and action representation.
        The main_car assumes all other cars maintain their
        current speed. In the simplified environment, state
        is 3-dimensional, representing the car's x coordinate,
        y coordinate, and vertical speed. The action space is
        2-dimensional, representing acceleration (unchanged) and
        velocity in the x dimension, per the assumption that in
        one timestep, the car can achieve any horizontal velocity
        in a given range.

        Assumes optimal trajectory will not involve the car going
        backwards, or more than double its initial speed. Assumes
        other cars in the world only move vertically.
    """
    def __init__(self, world, main_car, filename, reward_fn=None, n_timesteps=15, friction=0.2):
        self.world = world
        self.main_car = main_car
        self.n_timesteps = n_timesteps
        self.friction = friction
        self.filename = filename

        if reward_fn is None:
            assert isinstance(main_car, LinearRewardCar)
            self.reward_fn = main_car.reward_fn
        else:
            self.reward_fn = reward_fn

        self.other_cars = [car for car in world.cars if car is not main_car]
        self.other_cars_init_states = [car.state.numpy() for car in self.other_cars]

        self.coarse_state_dim = 3
        self.coarse_action_dim = 2

        self.main_car_init_state = main_car.state.numpy()
        self.coarse_state_ranges = [
            (-0.22, 0.22), # x-coordinate within lane bounds + a little bit out of bounds
            (self.main_car_init_state[1], # y-coordinate bounds
                self.main_car_init_state[1]+world.dt*n_timesteps*self.main_car_init_state[2]*2),
            (0., self.main_car_init_state[2]*2) # vertical speed bounds
        ]
        self.coarse_action_ranges = [
            (-3, 3), # acceleration bounds
            (-0.1, 0.1) # horizontal speed bounds
        ]
        self.num_action_values = [15, 15]  # might include one more each, if 0 is not already included

        self.num_state_values = [20, 60, 30]
        self.state_disc_grid = [np.linspace(rang[0], rang[1], n) for rang, n in
                                zip(self.coarse_state_ranges, self.num_state_values)]
        self.action_disc_grid = [list(np.linspace(rang[0], rang[1], n)) for rang, n in
                                 zip(self.coarse_action_ranges, self.num_action_values)]

        self.coarse_state_dict = self._coords_from_disc_grid(self.state_disc_grid)
        for action_dim in range(len(self.coarse_action_ranges)): # ensure we include 0-control in each dimension
            if 0. not in self.action_disc_grid[action_dim]:
                self.action_disc_grid[action_dim].append(0.)

        self.coarse_action_values = list(self._coords_from_disc_grid(self.action_disc_grid).values())
        self.coarse_action_dict = self._coords_from_disc_grid(self.action_disc_grid)

        # Value grids; first dimension is time steps, remaining dimensions are coarse state
        self.v_grids = np.zeros(shape=[self.n_timesteps + 1] +
                                      [len(self.state_disc_grid[i]) for i in range(self.coarse_state_dim)], dtype=np.float32)
        # Q-value grids; first dimension is time steps, remaining dimensions are coarse state and action
        self.q_grids = np.zeros(shape=[self.n_timesteps + 1] +
                                      [len(self.state_disc_grid[i]) for i in range(self.coarse_state_dim)] +
                                      [len(self.action_disc_grid[i]) for i in range(self.coarse_action_dim)],
                                    dtype=np.float32)
        # Policy grid; stores optimal action for each state for each timestep
        self.policy_grids = np.zeros(shape=[self.n_timesteps + 1] +
                                      [len(self.state_disc_grid[i]) for i in range(self.coarse_state_dim)]
                                      +[self.coarse_action_dim], dtype=np.float32)
        # best next state grids; stores the index of the state we end up in by taking optimal action for each state for each timestep
        self.best_next_state_indices = np.zeros(shape=[self.n_timesteps + 1]
                                                      + [len(self.state_disc_grid[i]) for i in range(self.coarse_state_dim)]
                                                      + [self.coarse_state_dim], dtype=np.int32)


        self.main_car_init_index = self._round_to_grid(self.main_car_init_state)
        print("Init index", self.main_car_init_index)

    def _coords_from_disc_grid(self, disc_grid):
        """Given a list of values for each dimension, returns all possible combinations of values,
            in a dictionary mapping index to coordinate value.
            i.e. disc_grid = [[1,2], [3,4,5]] -> [(0,0): (1,3), (0,1): (1,4), (0,2): (1,5),
                                    (1,0): (2,3), (1,1): (2,4), (1,2): (2,5)]
        """
        inds = [[]]
        coords = [[]]
        for dim in range(len(disc_grid)): # iteratively add all values along a given dimension
            expanded_coords = []
            expanded_inds = []
            for partial_ind, partial_coord in zip(inds, coords):
                for i, dim_val in enumerate(disc_grid[dim]):
                    expanded_coords.append(partial_coord+[dim_val])
                    expanded_inds.append(partial_ind+[i])
            inds = expanded_inds
            coords = expanded_coords

        return {tuple(i): tuple(c) for i, c in zip(inds, coords)}

    def _next_coarse_state(self, coarse_state, coarse_action):
        prev_x, prev_y, prev_vertical_speed = coarse_state
        acc, horiz_speed = coarse_action

        x = prev_x + horiz_speed*self.world.dt
        y = prev_y + prev_vertical_speed*self.world.dt
        vertical_speed = prev_vertical_speed + self.world.dt*(acc - self.friction * prev_vertical_speed**2)
        return (x, y, vertical_speed)

    def _round_to_grid(self, coarse_state):
        """
            Returns index of closest grid state to coarse_state, unless coarse_state is out of grid bounds, in which
            case it returns None
        """
        for dim, (low, high) in enumerate(self.coarse_state_ranges):
            if coarse_state[dim] < low or coarse_state[dim] > high:
                return None

        def closer_adjacent_index(i, x, a):
            """Returns i if a[i] is closer to x than a[i-1], otherwise returns i-1"""
            return i if abs(a[i] - x) < abs(a[i - 1] - x) else (i - 1)

        rounded_state = []
        for dim, grid_values_for_dim in enumerate(self.state_disc_grid):
            i = bisect.bisect_left(grid_values_for_dim, coarse_state[dim])
            rounded_state.append(closer_adjacent_index(i, coarse_state[dim], grid_values_for_dim))

        return tuple(rounded_state)

    def _index_to_coarse_state(self, ind):
        return self.coarse_state_dict[ind]

    def _set_other_car_states(self, t):
        """
            Updates other car states to the predicted state at time t.
            Assumes heading is vertical and speed doesn't change
        """
        for car, init_state in zip(self.other_cars, self.other_cars_init_states):
            state_at_t = init_state + np.array([0., init_state[2]*t, 0., 0.])
            car.state = state_at_t

    def _coarse_reward(self, coarse_state, coarse_action):
        x, y, vertical_speed = coarse_state
        acc, horiz_speed = coarse_action
        mapped_state = np.array([x, y, vertical_speed, np.pi / 2], dtype=np.float32)
        states = [(mapped_state if car == self.main_car else car.state.astype(np.float32)) \
                  for car in self.world.cars]
        return self.reward_fn(states, np.array([acc, 0], dtype=np.float32))

    def _run_value_iteration(self):
        self.v_grids.fill(np.nan)
        self.q_grids.fill(np.nan)
        self.policy_grids.fill(np.nan)
        self.best_next_state_indices.fill(np.nan)

        self.v_grids[-1].fill(0.)

        for timestep in range(self.n_timesteps-1, -1, -1):
            t = timestep*self.world.dt
            print("Beginning value computation for timestep=%d, t=%f"%(timestep, t))
            self._set_other_car_states(t)
            for cs_index, coarse_state in self.coarse_state_dict.items():
                as_qs_nss = [] # list of tuples (action, q value, next_state)
                for ac_index, coarse_action in self.coarse_action_dict.items():
                    ns_index = self._round_to_grid(self._next_coarse_state(coarse_state, coarse_action))

                    if ns_index is None: # don't set q-value if action take s state out of bounds
                        continue

                    if not np.isnan(self.v_grids[timestep+1][ns_index]):
                        q_value = self._coarse_reward(coarse_state, coarse_action).numpy() \
                                  + self.v_grids[timestep+1][ns_index]
                        self.q_grids[timestep][cs_index][ac_index] = q_value
                        as_qs_nss.append([coarse_action, q_value, ns_index])
                if as_qs_nss:
                    best_a, best_q, best_next_state_index = max(as_qs_nss, key=lambda a_q_ns: a_q_ns[1])
                    self.v_grids[timestep][cs_index] = best_q
                    self.policy_grids[timestep][cs_index] = best_a
                    self.best_next_state_indices[timestep][cs_index] = np.array(best_next_state_index, dtype=int)
                    print("\ntimestep=%d, cs_index=%s, best_a=%s, value=%f"%(timestep, str(cs_index), str(best_a), best_q))
                    # print("as_qs_nss",as_qs_nss)
        if self.filename:
            with open(self.filename, 'wb') as file:
                pickle.dump({'v_grids': self.v_grids, 'q_grids': self.q_grids, 'policy_grids': self.policy_grids,
                             'next_state_inds': self.best_next_state_indices}, file)
