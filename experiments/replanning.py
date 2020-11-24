import contextlib
import numpy as np

from interact_drive.world import ThreeLaneCarWorld
from interact_drive.car import FixedVelocityCar, FixedPlanCar
from merging import ThreeLaneTestCar


def setup_world():
    world = ThreeLaneCarWorld(visualizer_args=dict(name="Merging"))

    our_car = ThreeLaneTestCar(world, np.array([0.0, -1., 1.5, np.pi / 2]),
                               horizon=5,
                               weights=np.array([-1, 0, -5, 0, 0, -50, -5], dtype=np.float32),
                               target_speed=1.2, planner_args={'n_iter': 300},
                               check_plans=True
                               )
    other_car_1 = FixedPlanCar(world, np.array([0.1, -0.8, 0.8, np.pi / 2]),
                               plan=[np.array([0.35, 1.3], dtype=np.float32)] * 3 + [
                                   np.array([0.0, -1.3], dtype=np.float32)] * 3,
                               default_control=np.array([0.0, 0.0], dtype=np.float32),
                               horizon=5, color='gray', opacity=0.8)
    other_car_2 = FixedVelocityCar(world, np.array([0.1, -0.8, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    other_car_3 = FixedVelocityCar(world, np.array([-0.1, -0.8, 0.8, np.pi / 2]),
                                   horizon=5, color='gray', opacity=0.8)
    world.add_cars([our_car, other_car_1, other_car_2, other_car_3])

    world.reset()
    return our_car, other_car_1, other_car_2, world


def main():
    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip

    our_car, other_car_1, other_car_2, world = setup_world()
    other_car_1.reset()

    frames = []
    frames.append(world.render("rgb_array"))

    for i in range(10):
        print("[{}] velocity".format(i), our_car.state[2])
        world.step()
        world.render()
        frames.append(world.render("rgb_array"))

    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.5).write_gif("replanning2.gif", program="ffmpeg")


##
if __name__ == "__main__":
    main()

##
