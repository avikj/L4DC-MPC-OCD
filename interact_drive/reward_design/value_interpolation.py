import numpy as np
import tensorflow as tf
import pickle
import itertools
class ValueFeature:
    def __init__(
        self,
        proj,
        value_data,
        dim=3,
        scale=1
    ):
        assert dim == 3
        self.proj = proj
        self.dims = dim
        self.scale = scale
        self.disc_grid = [tf.constant(row, dtype=tf.float32) for row in value_data['disc_grid']]
        self.v_grids = tf.constant(value_data['v_grids'])
        self.cell_corner = tf.Variable(np.zeros((len(self.disc_grid)), dtype=np.int32))
    def get_config(self):
        """Return JSON object describing the parameters of this strategic value."""
        return {
            "mat_name": self.mat_name,
            "dim": self.dims,
            "scale": self.scale
        }

    def interpolate_value(self, t=0):
        """Return a feature that computes the interpolated value for the robot car
        given its state. (multilinear interpolation)
        """

        @tf.function
        def interpolated_func(world_state):
            # Project x to coarse state space
            x_coarse = self.proj(world_state)
            # compute "top-left" cell corner
            if tf.reduce_all([x_coarse[dim] >= self.disc_grid[dim][0] and x_coarse[dim] <= self.disc_grid[dim][-1] for dim in range(self.dims)]):

                self.cell_corner.assign([tf.where(self.disc_grid[dim]<=x_coarse[dim])[-1][0] for dim, grid_values in enumerate(self.disc_grid)])
                # multilinear interpolation
                step_grid = [self.disc_grid[dim][self.cell_corner[dim]+1]-self.disc_grid[dim][self.cell_corner[dim]] for dim in range(self.dims)]

                cell_volume = tf.reduce_prod(step_grid)
                sumterms = []
                for i in itertools.product(range(2), repeat=self.dims):
                    partial_volume = tf.reduce_prod([
                        (
                                (-1) ** (i[dim] + 1) * (x_coarse[dim] - self.disc_grid[dim][self.cell_corner[dim]])
                                + (1 - i[dim]) * step_grid[dim]
                        )
                        for dim in range(self.dims)
                    ])
                    i = np.array(i)
                    sumterm = tf.gather_nd(self.v_grids[t], [self.cell_corner+i])[0] * partial_volume / cell_volume
                    sumterms.append(sumterm)
                sum_val = sum(sumterms)
                return sum_val
            else:
                return float("nan")
        return interpolated_func
