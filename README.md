# Optimal Cost Design for Model Predictive Control (L4DC 2020 submission code)

_Note: while in the paper we discuss cost functions to be minimized, we use the convention of maximizing reward functions in this codebase; these are identical modulo a negative sign_

## Requirements
```
python>=3.6
```

## Installation

```
pip install -e .
```

### Visualizing Scenarios 

We study three scenarios, which we refer to as `finite_horizon`, `local_opt`, and `replanning`. To visualize the `finite_horizon` scenario sampled at 3 different initial conditions, run the following script from the root directory. 

```
python experiments/run_mpc_ord.py finite_horizon vis --n_inits 3
```

This will generate a reward heatmap PNG and a GIF displaying 3 trajectories in the scenario from sampled initial conditions; this is done both for the true hand-designed reward weights for the scenario and for tuned reward weights which are hard coded in `run_mpc_ord.py`. You can play around with these weight values to see how the reward landscape and resulting trajectories are affected.
<p float="left">
<img src="https://raw.githubusercontent.com/avikj/L4DC-MPC-OCD/master/finite_horizon_true_weights.gif" width="45%" padding="left:30px;bottom:30px;"/>
<img src="https://raw.githubusercontent.com/avikj/L4DC-MPC-OCD/master/finite_horizon_true_weights_heatmap.png" width="45%" />
</p>

## Running Cost (Reward) Design 

To actually run zeroth order optimization to find weights for a surrogate reward function for one of the environments, the same script can be run with the second argument set to `cmaes` or `random` rather than the `vis` (`cmaes` is strongly recommended). Here, the `n_inits` flag determines how many sampled initial conditions in the environment are used to by the optimizer to evaluate each set of weights; using more samples improves generalization to unseen initial conditions, but requires more computation.

```
python experiments/run_mpc_ord.py finite_horizon cmaes --n_inits 5
```

The reward printed in `Iteration 0` is the total return of the true weights on the sampled initial conditions, and all other rewards printed correspond to surrogate weights proposed by the optimization algorithm.
