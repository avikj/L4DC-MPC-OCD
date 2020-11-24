"""
Sacred experiment where we verify that the inference of CIOC as beta->infinity
converges to the inference of Inverse Locally Optimal Control, which assumes
a fully (locally) optimal demonstrator.
(We use a simple example so CIOC's Laplace approximation is exact.)

Since CIOC doesn't exactly have a Beta term, this is achieved by scaling the
rewards. And since we use a linear reward function, we simply scale the norm
of the weight vector.

If you want to try out Sacred logging without setting up a MongoDB instance,
try running this file with the -F flag:
```
python cioc_beta_experiment.py -F /path/to/log/dir
```
"""

from os.path import dirname, join
from pathlib import Path
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
import seaborn as sns
import tensorflow as tf

from interact_drive.world import CarWorld
from interact_drive.reward_inference.first_order_ioc import (
    InverseLocallyOptimalControl,
)
from interact_drive.reward_inference.second_order_ioc import LocalCIOC
from interact_drive.reward_inference.tests.linearTargetSpeedPlannerCar import (
    LinearTargetSpeedPlannerCar,
)

default_output_dir = join(dirname(__file__), "data")

ex = Experiment("CIOC-vs-ILOC")
ex.logger = tf.get_logger()

@ex.config
def cioc_beta_config():
    """
    Config file for this Sacred experiment.

    These are passed to any method with the appropriate header and Sacred
    decorator.

    Sacred allows you to change these when running this file from the command
    line using the `with` flag, for example:
        `python cioc_beta_experiment.py with debug_level=30`
    will set debug_level to 30, thereby suppressing INFO and DEBUG messages
    from the logger.
    """
    debug_level: int = 20  # debug_level = INFO
    weight_norms: List[Union[float, int]] = [10.0 ** (i / 2) for i in range(10)]
    output_dir: str = default_output_dir
    horizon = 5
    traj_len = 5
    fig_name = "l2_vs_beta.png"


def set_style():
    """Sets the matplotlib plotting style to the InterACT lab standard."""
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rc("font", family="serif", serif=["Palatino"])
    sns.set_style("white")

    sns.set(font="serif", font_scale=1.4)

    # Make the background white, and specify the
    # specific font family
    sns.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )


@ex.main
def cioc_beta_experiment(
    debug_level: int,
    weight_norms: List[Union[float, int]],
    output_dir: str,
    traj_len: int,
    horizon: int,
    fig_name: str,
    _log,
):
    """
    Main Sacred method for this experiment.
    """
    # setup
    ex.logger.setLevel(debug_level)
    world = CarWorld()
    initial_theta = tf.constant([1.0, 1.0])
    weights_tf = tf.constant([2.0, -1.0], dtype=tf.float32)
    car = LinearTargetSpeedPlannerCar(
        world,
        tf.constant(np.array([0.0, 0.0, 1.0, np.pi / 2]), dtype=tf.float32),
        horizon=horizon,
        target_speed=0.0,
        weights=weights_tf,
        friction=0.0,
        planner_args=dict(n_iter=10, learning_rate=5.0),
    )
    world.add_car(car)

    trajectory = []
    for t in range(traj_len):
        past_state, controls, next_state = world.step(dt=0.1)
        trajectory.append((past_state, controls))

    cioc = LocalCIOC(
        car, weight_norm=tf.constant(1.0), initial_weights=initial_theta
    )
    iloc = InverseLocallyOptimalControl(
        car, weight_norm=tf.constant(1.0), initial_weights=initial_theta
    )

    # compute theta_hat with the two algorithms
    cioc_thetas = []
    iloc_thetas = []

    for weight_norm in weight_norms:
        _log.info("rationalizing with weight_norm={}".format(weight_norm))
        weight_norm_tf = tf.constant(weight_norm, dtype=tf.float32)
        cioc.weight_norm, iloc.weight_norm = weight_norm_tf, weight_norm_tf
        theta_cioc = cioc.rationalize(trajectory=trajectory).numpy()
        theta_iloc = iloc.rationalize(trajectory=trajectory, n_iter=300).numpy()
        _log.info(
            "weight_norm={}, theta_cioc={}, theta_iloc={}".format(
                weight_norm, theta_cioc, theta_iloc
            )
        )
        cioc_thetas.append(theta_cioc)
        iloc_thetas.append(theta_iloc)

    true_norm_weights = tf.nn.l2_normalize(weights_tf).numpy()
    norm_cioc_thetas = [
        tf.nn.l2_normalize(theta).numpy() for theta in cioc_thetas
    ]
    norm_iloc_thetas = [
        tf.nn.l2_normalize(theta).numpy() for theta in iloc_thetas
    ]
    l2_dists = dict(
        cioc=[
            np.linalg.norm(true_norm_weights - weight)
            for weight in norm_cioc_thetas
        ],
        iloc=[
            np.linalg.norm(true_norm_weights - weight)
            for weight in norm_iloc_thetas
        ],
    )

    _log.info(true_norm_weights)
    _log.info(norm_cioc_thetas)
    _log.info(norm_iloc_thetas)
    _log.info(l2_dists)

    # plot
    set_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(weight_norms, l2_dists["cioc"], label="CIOC (Boltzmann)")
    ax.plot(weight_norms, l2_dists["iloc"], label="ILOC (Locally Optimal)")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$||\hat \theta-\theta^*||_2$")
    ax.set_title(r"$L^2$ distance between inferred and true reward weights")
    ax.legend()
    plt.tight_layout()

    # save the plot
    Path(output_dir).mkdir(
        parents=True, exist_ok=True
    )  # make dir if it doesn't exist
    fig_path = join(output_dir, fig_name)
    plt.savefig(fig_path, bbox_inches="tight")
    ex.add_artifact(filename=fig_path, name=fig_name)


if __name__ == "__main__":
    ex.run_commandline()
