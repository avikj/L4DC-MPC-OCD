"""Base class for planners."""

from interact_drive.world import CarWorld
from interact_drive.car.car import Car


class CarPlanner(object):
    """
    Parent class for all the trajectory finders for a car.
    """

    def __init__(self, world: CarWorld, car: Car):
        self.world = world
        self.car = car

    def generate_plan(self):
        raise NotImplementedError


class CoordinateAscentPlanner(CarPlanner):
    """
    CarPlanner that performs coordinate ascent to find an approximate Nash
    equilibrium trajectory.
    """


# class HierarchicalPlanner(CarPlanner):
#     def __init__(self, world, car):
#         pass
#
#     def initial
