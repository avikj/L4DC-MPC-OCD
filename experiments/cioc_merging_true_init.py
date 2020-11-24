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

from merging import ThreeLaneTestCar
from interact_drive.world import ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar
from interact_drive.reward_inference import InverseLocallyOptimalControl, LocalCIOC
from cioc_beta_experiment import set_style

default_output_dir = join(dirname(__file__), "data")

ex = Experiment("CIOC-merging-true-init")
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
    weight_norms: List[Union[float, int]] = [10.0 ** i for i in range(-6,4)]
    output_dir: str = default_output_dir
    horizon = 5
    traj_len = 5
    fig_name = "merging_l2_vs_beta.png"


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
    
    world = ThreeLaneCarWorld(visualizer_args=dict(name="Merging"))
    # features are [velocity, lane_0_dist, lane_1_dist, lane_2_dist, min_lane_dist, collision]
    true_weights = np.array([0.1, 0., 0., -100., -1., -10], dtype=np.float32)
    our_car = ThreeLaneTestCar(world, np.array([0, -0.5, 0.8, np.pi / 2]),
                               horizon=5,
                               weights=true_weights)
    other_car_1 = FixedVelocityCar(world, np.array([0.1, -0.7, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    other_car_2 = FixedVelocityCar(world, np.array([0.1, -0.2, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    world.add_cars([our_car, other_car_1, other_car_2])


    world.reset()
    trajectory = []
    for i in range(15):
        past_state, controls, state = world.step()
        trajectory.append((past_state, controls))

    cioc = LocalCIOC(our_car, initial_weights=true_weights)
    # iloc = InverseLocallyOptimalControl(our_car, initial_weights=initial_theta)

    cioc_thetas = []
    for weight_norm in weight_norms:
        _log.info("rationalizing with weight_norm={}".format(weight_norm))
        weight_norm_tf = tf.constant(weight_norm, dtype=tf.float32)
        cioc.weight_norm = weight_norm_tf
        theta_cioc = cioc.rationalize(trajectory=trajectory).numpy()
        _log.info(
            "weight_norm={}, theta_cioc={}".format(
                weight_norm, theta_cioc
            )
        )
        cioc_thetas.append(theta_cioc)

    true_norm_weights = true_weights/np.linalg.norm(true_weights)
    norm_cioc_thetas = [
        theta/np.linalg.norm(theta) for theta in cioc_thetas
    ]
    l2_dists = [
        np.linalg.norm(true_norm_weights - weight)
        for weight in norm_cioc_thetas
    ]

    _log.info(true_norm_weights)
    _log.info(norm_cioc_thetas)
    _log.info(l2_dists)

    # plot
    set_style()
    fig, ax = plt.subplots(nrows=1, ncols=1, dpi=150)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(weight_norms, l2_dists, label="CIOC (Boltzmann)")
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
