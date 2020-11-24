"""Base class for driving scenarios."""

from typing import Dict, Iterable, List, Optional, Tuple

import tensorflow as tf
import numpy as np


class CarWorld(object):
    """
    Contains the objects in a driving scenario - cars, lanes, obstacles, etc.

    In addition, contains a step() function that increments the state of the
    environment over time.

    Finally, this class provides visualizations for the environment.
    """

    def __init__(self, dt: float = 0.1, lanes: Optional[List] = None,
                 obstacles: Optional[List] = None,
                 visualizer_args: Optional[Dict] = None,
                 **kwargs):
        """
        Initializes this CarWorld. Note: the visualizer is *not* initialized
        until the first render() call.

        Args:
            dt: the time increment per tick of simulation.
            lanes: a list of lanes this world should contain.
            obstacles: a list of obstacles this world should contain.
            visualizer_args: a dict of arguments for the visualizer.
            **kwargs:
        """

        self.cars = []
        self.dt = dt

        if lanes is None:
            self.lanes = []
        else:
            self.lanes = lanes

        if obstacles is None:
            self.obstacles = []
        else:
            self.obstacles = obstacles

        if visualizer_args is None:
            self.visualizer_args = dict()
        else:
            self.visualizer_args = visualizer_args

        self.visualizer = None

    def add_car(self, car):
        car.index = len(self.cars)
        self.cars.append(car)

    def add_cars(self, cars: Iterable):
        for car in cars:
            self.add_car(car)

    @property
    def state(self):
        return [c.state for c in self.cars]

    @state.setter
    def state(self, new_state: Iterable):
        for c, x in zip(self.cars, new_state):
            c.state = x

    def reset(self):
        for car in self.cars:
            car.reset()

        if self.visualizer is not None:
            self.visualizer.reset()

    def step(self, dt: Optional[float] = None) -> Tuple[List[tf.Tensor],
                                                        List[tf.Tensor],
                                                        List[tf.Tensor]]:
        """
        Asks all cars to generate plans, and then updates the world state based
        on those plans

        We need to split the plan generation and car state updating because
        all the cars act at once (in the simulation)

        Args:
            dt: the amount of time to increment the simulation forward by.

        Returns:
            past_state: the previous state of the world, before this tick.
            controls: the controls applied to all the cars in this timestep.
            state: the current state of the world.
        """
        past_state = self.state

        if dt is None:
            dt = self.dt

        for car in self.cars:
            if not car.control_already_determined_for_current_step:
                car.set_next_control()

        for car in self.cars:
            car.step(dt)

        return past_state, [c.control for c in self.cars], self.state

    def render(self, mode: str = "human", heatmap_show=False) -> Optional[np.array]:
        """
        Renders the state of this car world. If mode="human", we display
        it using the visualizer. If mode="rgb_array", we return a np.array
        with shape (x, y, 3) representing RGB values, useful for making gifs
        and videos.

        Note: we currently assume that the main car is the first car in
            self.cars.

        Args:
            mode: One of ["human", "rgb_array"].

        Returns:
            rgb_representation: if str="rgb_array", we return an np.array of
                    shape (x, y, 3), representing the rendered image.

        TODO(chanlaw): add support for terminal visualization
        """
        if self.visualizer is None:
            from interact_drive.visualizer import CarVisualizer
            self.visualizer = CarVisualizer(world=self, **self.visualizer_args)
            self.visualizer.set_main_car(index=0)

        if mode == "human":
            self.visualizer.render(display=True, return_rgb=False, heatmap_show=heatmap_show)
        elif mode == "rgb_array":
            return self.visualizer.render(display=False, return_rgb=True, heatmap_show=heatmap_show)
        else:
            raise ValueError("Mode must be either `human` or `rgb_array`.")


class ThreeLaneCarWorld(CarWorld):
    """
    A car world initialized with three straight lanes that extend for
    a long while in either direction.
    """

    def __init__(self, dt=0.1, **kwargs):
        lane = StraightLane((0.0, -5.), (0.0, 10.), 0.1)
        lanes = [lane.shifted(1), lane, lane.shifted(-1)]
        super().__init__(dt=dt, lanes=lanes, **kwargs)


class TwoLaneCarWorld(CarWorld):
    def __init__(self, dt=0.1, **kwargs):
        lane = StraightLane((-0.05, -5.), (-0.05, 10.), 0.1)
        lanes = [lane, lane.shifted(-1)]
        super().__init__(dt=dt, lanes=lanes, **kwargs)


class StraightLane(object):
    """
    Defines a lane with median defined by the line segment between points
    p and q, and width w.

    TODO(chanlaw): need to implement roads that aren't line segments
    """

    def __init__(self, p: Tuple[float, float], q: Tuple[float, float],
                 w: float):
        """
        Initializes the straight lane.

        Args:
            p: the x,y coordinates of the start point for the center of the lane
            q: the x,y coordinates of the end point for the center of the lane
            w: the width of the lane
        """

        self.p = np.asarray(p)
        self.q = np.asarray(q)
        self.w = w

        self.m = (self.q - self.p) / np.linalg.norm(
            self.q - self.p)  # unit vector in direction of lane
        self.n = np.asarray(
            [-self.m[1], self.m[0]])  # normal vector to the lane

    def shifted(self, n_lanes: int):
        """
        Returns a lane that is shifted n_lanes in the direction of self.n.

        When n_lanes < 0, this is shifted in the other direction instead.

        Args:
            n_lanes: number of lanes to shift

        Returns:
            (StraightLane): a straight lane shifted in the appropriate way.

        """
        return StraightLane(self.p + self.n * self.w * n_lanes,
                            self.q + self.n * self.w * n_lanes, self.w)

    def dist2median(self, point: Tuple[float]):
        """
        Returns the squared distance of a point to the median of the lane.

        Args:
            point: the x,y coordinates of the point.

        Returns:
            (float): the distance to the median of this lane
        """
        r = ((point[0] - self.p[0]) * self.n[0]
             + (point[1] - self.p[1]) * self.n[1])
        return r ** 2

    def on_road(self, point):
        raise NotImplementedError
