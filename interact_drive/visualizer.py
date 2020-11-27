"""Visualizer class for CarWorlds."""

import math
from os.path import dirname, join
from typing import Dict, Optional

import numpy as np
import matplotlib
import matplotlib.cm

import pyglet
import pyglet.gl as gl
import pyglet.graphics as graphics

from interact_drive.car import Car, LinearRewardCar
from interact_drive.world import CarWorld, StraightLane

##
default_asset_dir = join(dirname(__file__), "assets")


def centered_image(filename):
    """
    Helper function that centers and returns the image located at `filename`.
    """
    # img = pyglet.resource.image(filename)
    img = pyglet.image.load(filename).get_texture()
    img.anchor_x = img.width / 2.0
    img.anchor_y = img.height / 2.0
    return img


def car_sprite(
    color, scale=0.85*0.15 / 600.0, asset_dir=default_asset_dir
) -> pyglet.sprite.Sprite:
    """
    Helper function that returns a sprite of an appropriately-colored car.
    """
    sprite = pyglet.sprite.Sprite(
        centered_image(join(asset_dir, "car-{}.png".format(color))),
        subpixel=True,
    )
    sprite.scale = scale
    return sprite


class CarVisualizer(object):
    """
    Class that visualizes a CarWorld.

    Attributes:
        world (CarWorld): the CarWorld this CarVisualizer is visualizing.

    TODO(chanlaw): add support for curved roads and obstacles.
    """
    def __init__(
        self,
        world: CarWorld,
        name: str = "car_sim",
        follow_main_car: bool = False,
        window_args: Optional[Dict] = None,
        asset_dir: Optional[str] = None
    ):
        """
        Initializes this car visualizer.

        Args:
            world: the `CarWorld` to visualize.
            name: the name of the scenario.
            follow_main_car: whether or not the camera should follow the main
                car (once one has been specified).
            window_args: a dict of arguments to pass to the window created by
                the visualizer.
            asset_dir: the directory containing image assets. If not specified,
                we use the `default_asset_dir` specified in this file.
        """
        self.world = world
        self.follow_main_car = follow_main_car

        if window_args is None:
            window_args = dict(
                caption=name, height=600, width=600, fullscreen=False
            )
        else:
            window_args["caption"] = name
        self.window_args = window_args

        self.window = pyglet.window.Window(**self.window_args)
        self.reset()

        if asset_dir is None:
            asset_dir = default_asset_dir
        self.asset_dir = asset_dir

        self.car_sprites = {
            c: car_sprite(c, asset_dir=self.asset_dir)
            for c in [
                "red",
                "yellow",
                "purple",
                "white",
                "orange",
                "gray",
                "blue",
            ]
        }
        self.label = pyglet.text.Label(
            "Speed: ",
            font_name="Times New Roman",
            font_size=24,
            x=30,
            y=self.window.height - 30,
            anchor_x="left",
            anchor_y="top",
        )
        self._grass = pyglet.image.load(
            join(asset_dir, "grass.png")
        ).get_texture()

        self.main_car = None

        # gist_rainbow works well with 0.75 multiplier
        self.colormap = matplotlib.cm.gist_rainbow  # jet, nipy_spectral, gist_rainbow

        self.heatmap_show = False

        self.magnify = 1 # magnification not implemented
        self.heatmap_size = (128, 128)# small: (32, 32), medium: (128, 128), large: (256, 256)
        self.fixed_heatmap_scale = False
        self.min_heatmap_val = None # must set these if fixed_heatmap_scale is True
        self.max_heatmap_val = None

    def set_main_car(self, index):
        """
        Sets the main car to follow with the camera and display the speed of.
        """
        self.main_car = self.world.cars[index]
        if hasattr(self.main_car, 'reward_fn'):
            self._set_heat(self.main_car.reward_fn, self.main_car)


    def reset(self):
        """
        Resets the visualized by closing the current window and opening
        a new window.
        """
        self._close_window()
        self._open_window()

    def _open_window(self):
        if self.window is None:
            self.window = pyglet.window.Window(**self.window_args)
            self.window.on_draw = self._draw_world
            self.window.dispatch_event("on_draw")

    def _close_window(self):
        self.window.close()
        self.window = None

    def render(
        self, display: bool = True, return_rgb: bool = False, heatmap_show=False, show_trajectory=False
    ) -> Optional[np.array]:
        """
        Renders the state of self.world. If display=True, then we display
        the result in self.window. If return_rgb=True, we return the result
        as an RGB array.

        Args:
            display: whether to display the result in self.window.
            return_rgb: whether to return the result as an rgb array.

        Returns:
            rgb_representation: If return_rgb=True, we return an np.array
                    of shape (x, y, 3), representing the rendered image.
                    Note: on MacOS systems, the rgb array is twice as large
                    as self.window's width/height parameters suggest.
        """
        self.heatmap_show = heatmap_show
        pyglet.clock.tick()
        window = self.window

        window.switch_to()
        window.dispatch_events()

        window.dispatch_event("on_draw")

        if return_rgb:
            # copy the buffer into an np.array
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            img_data = buffer.get_image_data()
            arr = np.fromstring(img_data.get_data(), dtype=np.uint8, sep="")

            # default format is RGBA, and we need to turn this into RGB.
            from sys import platform as sys_pf

            if sys_pf == "darwin":
                # if system is mac, the image is twice as large as the window
                width, height = self.window.width * 2, self.window.height * 2
            else:
                width, height = self.window.width, self.window.height

            arr = arr.reshape([width, height, 4])
            arr = arr[::-1, :, 0:3]

        if display:
            window.flip()  # display the buffered image in the window

        if return_rgb:
            return arr

    def _set_heat(self, reward, car=None):
        import tensorflow as tf
        # reward: the reward function
        # car: the car corresponding to the reward function. Used to get its
        # current heading and velocity. If car is None, visualize the reward for
        # 0 heading and 0 velocity.
        assert(self.main_car is not None)
        x = tf.Variable(np.zeros(shape=(4,), dtype=np.float32), name='set_heat_x');
        u = tf.Variable(np.zeros(shape=(2,), dtype=np.float32));
        if car is not None:
            def val(pos):
                # maintain same heading and velocity in a different x, y location
                states = [(x if c == self.main_car else c.state) for c in self.world.cars]
                x.assign(np.array([pos[0], pos[1], car.state[2], car.state[3]]))

                # maintain same control
                if car.control is not None:
                    u.assign(np.array(car.control))
                else:
                    u.assign(np.array([0., 0.]))
                return reward(states, u).numpy()
        else:
            def val(pos):
                states = [(x if c == self.main_car else c.state) for c in self.world.cars]
                x.assign(np.array([pos[0], pos[1], 0.0, 0.0]))
                u.assign(np.array([0., 0.]))
                return reward(states, u).numpy()
        self.heat = val

    def _draw_heatmap(self):
        """Draw reward defined by self.heat as a heatmap."""
        if not self.heatmap_show or not self.heat:
            return
        center_x = 0. # TODO make this more general?
        center_vis = self._get_center()  # center of visualization

        # heatmap center is (center of road, y coordinate of main car)
        center_heatmap = [center_x, center_vis[1]]
        # proportion of width and height to draw heatmap around the center
        if False: # if config.FULL_HEATMAP: # TODO initialization option for this
            w_h = [1.0, 1.0]
        else:
            w_h = [0.15, 1.0]

        # Min and max coordinates of the heatmap that define the largest area
        # that could be visible.
        visible_heatmap_min_coord = center_heatmap - np.asarray(w_h) / self.magnify
        visible_heatmap_max_coord = center_heatmap + np.asarray(w_h) / self.magnify

        # Set the min and max coordinates of the heatmap
        self.heatmap_min_coord = visible_heatmap_min_coord
        self.heatmap_max_coord = visible_heatmap_max_coord

        size = self.heatmap_size
        min_coord = self.heatmap_min_coord
        max_coord = self.heatmap_max_coord

        vals = np.zeros(size)
        for i, x in enumerate(np.linspace(min_coord[0] + 1e-6, max_coord[0] - 1e-6, size[0])):
            for j, y in enumerate(np.linspace(min_coord[1] + 1e-6, max_coord[1] - 1e-6, size[1])):
                vals[j, i] = self.heat(np.asarray([x, y]))

        # Set min and max values if showing the strategic value heatmap
        # using either fixed values or dynamic values based on the visible heatmap
        if self.min_heatmap_val is None or not self.fixed_heatmap_scale:
            min_val = np.nanmin(vals)
        else:
            min_val = self.min_heatmap_val
        if self.max_heatmap_val is None or not self.fixed_heatmap_scale:
            max_val = np.nanmax(vals)
        else:
            max_val = self.max_heatmap_val
        # scale and translate the values to make the heatmap most useful
        # 1 - vals to reverse the heatmap colors to make red==bad and blue==good
        vals = ((vals - min_val) / (max_val - min_val))
        vals *= 0.75
        # vals = 1 - vals
        vals = self.colormap(vals)
        vals[:, :, 3] = 0.7  # opacity
        vals = (vals * 255.).astype('uint8').flatten()  # convert to RGBA
        vals = (gl.GLubyte * vals.size)(*vals)
        img = pyglet.image.ImageData(size[0], size[1], 'RGBA', vals,
                                     pitch=size[1] * 4)
        self.heatmap = img.get_texture()
        self.heatmap_valid = True

        gl.glClearColor(1., 1., 1., 1.)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)
        min_coord = self.heatmap_min_coord
        max_coord = self.heatmap_max_coord
        graphics.draw(4, gl.GL_QUADS,
                      ('v2f', (min_coord[0], min_coord[1], max_coord[0], min_coord[1],
                               max_coord[0], max_coord[1], min_coord[0], max_coord[1])),
                      ('t2f', (0., 0., 1., 0., 1., 1., 0., 1.)),
                      # ('t2f', (0., 0., size[0], 0., size[0], size[1], 0., size[1]))
                      )
        gl.glDisable(self.heatmap.target)

    def _draw_world(self):
        """Draws the world into the pyglet buffer."""
        self.window.clear()

        # start image drawing mode
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()

        self._center_camera()

        self._draw_background()
        for lane in self.world.lanes:
            self._draw_lane(lane)

        for car in self.world.cars[::-1]: # main robot car is generally first; draw them backwards to draw it on top
            #if car.debug:
            #    self._draw_trajectory(car)
            #else:

            self._draw_car(car)


        if self.main_car is not None:
            self.label.text = "Speed: {0:.2f}".format(
                self.world.cars[0].state.numpy()[2]
            )
            if self.heatmap_show:
                if not self.heat:
                    self._set_heat(self.main_car.reward_fn, self.main_car)
                self._draw_heatmap()
        # end image drawing mode
        gl.glPopMatrix()

        self.label.draw()

    def _draw_background(self):
        """Draws the grass background."""
        gl.glEnable(self._grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self._grass.target, self._grass.id)

        # specifies radius of grass area in the visualization
        # we assume that no car environment is going to have cars going outside
        radius = 20.0

        # increase this to decrease the size of the grass tiles
        texture_shrink_factor = 5.0

        # v2f: tells gl to draw a background with vertices in 2d, where
        #       coordinates are in float
        # t2f: tells gl how the texture of the background should be read from
        #       the grass file.
        graphics.draw(
            4,
            gl.GL_QUADS,
            (
                "v2f",
                (
                    -radius,
                    -radius,
                    radius,
                    -radius,
                    radius,
                    radius,
                    -radius,
                    radius,
                ),
            ),
            (
                "t2f",
                (
                    0.0,
                    0.0,
                    texture_shrink_factor * radius,
                    0.0,
                    texture_shrink_factor * radius,
                    texture_shrink_factor * radius,
                    0.0,
                    texture_shrink_factor * radius,
                ),
            ),
        )
        gl.glDisable(self._grass.target)

    def _draw_car(self, car: Car):
        """Draws a `Car`."""
        state = car.state.numpy()
        color = car.color
        opacity = 255

        sprite = self.car_sprites[color]
        sprite.x, sprite.y = state[0], state[1]
        sprite.rotation = -state[3] * 180.0 / math.pi
        sprite.opacity = opacity
        sprite.draw()

    def _draw_lane(self, lane: StraightLane):
        """Draws a `StraightLane`."""
        # first, draw the asphalt
        gl.glColor3f(0.4, 0.4, 0.4)  # set color to gray
        graphics.draw(
            4,
            gl.GL_QUAD_STRIP,
            (
                "v2f",
                np.hstack(
                    [
                        lane.p - 0.5 * lane.w * lane.n,
                        lane.p + 0.5 * lane.w * lane.n,
                        lane.q - 0.5 * lane.w * lane.n,
                        lane.q + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )
        # next, draw the white lines between lanes

        gl.glColor3f(1.0, 1.0, 1.0)  # set color to white
        graphics.draw(
            4,
            gl.GL_LINES,
            (
                "v2f",
                np.hstack(
                    [
                        lane.p - 0.5 * lane.w * lane.n,
                        lane.q - 0.5 * lane.w * lane.n,
                        lane.p + 0.5 * lane.w * lane.n,
                        lane.q + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )

    def _center_camera(self):
        """Sets the camera coordinates."""
        x, y = self._get_center()
        z = 0.0
        # set the camera to be +1/-1 from the center coordinates
        gl.glOrtho(x - 1., x + 1.,
                   y - 1., y + 1.,
                   z - 1., z + 1.)

    def _get_center(self):
        if self.main_car is not None and self.follow_main_car:
            return self.main_car.state[0], self.main_car.state[1]
        else:
            return 0.0, 0.0


    def _draw_trajectory(self, car: Car, opacity=0.91, use_subframes=False):
        """Draws the past trajectory of the `Car`."""
        if len(car.past_traj) > 0:
            color = car.color
            for t, (state_tf, control_tf) in enumerate(car.past_traj):
                if t%4 != 0:
                    continue
                state = state_tf.numpy()

                sprite = self.car_sprites[color]
                sprite.x, sprite.y = state[0], state[1]
                sprite.rotation = -state[3] * 180.0 / math.pi
                sprite.opacity = opacity ** (len(car.past_traj) - t) * 255
                sprite.draw()

                if use_subframes:
                    if t < len(car.past_traj) - 1:
                        next_state_tf, next_control_tf = car.past_traj[t + 1]
                        next_state = next_state_tf.numpy()
                        middle_state = (state + next_state) / 2

                        sprite = self.car_sprites[color]
                        sprite.x, sprite.y = middle_state[0], middle_state[1]
                        sprite.rotation = -middle_state[3] * 180.0 / math.pi
                        sprite.opacity = (
                            car.opacity
                            * opacity ** (len(car.past_traj) - t - 0.5)
                            * 255
                        )
                        sprite.draw()
                    else:
                        next_state = car.state.numpy()
                        middle_state = (state + next_state) / 2

                        sprite = self.car_sprites[color]
                        sprite.x, sprite.y = middle_state[0], middle_state[1]
                        sprite.rotation = -middle_state[3] * 180.0 / math.pi
                        sprite.opacity = (
                            car.opacity
                            * opacity ** (len(car.past_traj) - t - 0.5)
                            * 255
                        )
                        sprite.draw()
##


def main():
    """Visualizes three cars in a merging scenario."""
    from experiments.merging import ThreeLaneTestCar
    from interact_drive.world import ThreeLaneCarWorld
    from interact_drive.car import FixedVelocityCar

    world = ThreeLaneCarWorld()
    our_car = ThreeLaneTestCar(
        world,
        np.array([0, -0.5, 0.8, np.pi / 2]),
        horizon=5,
        weights=np.array([0.1, 0.0, 0.0, -10.0, -1.0, -10, -10]),
        debug=True
    )
    other_car_1 = FixedVelocityCar(
        world,
        np.array([0.1, -0.7, 0.8, np.pi / 2]),
        horizon=5,
        color="gray",
        opacity=1.0,
        debug=True
    )
    other_car_2 = FixedVelocityCar(
        world,
        np.array([0.1, -0.2, 0.8, np.pi / 2]),
        horizon=5,
        color="gray",
        opacity=1.0,
        debug=True
    )
    world.add_cars([our_car, other_car_1, other_car_2])
    world.reset()

    # vis.render(heatmap_show=False)
    for i in range(5):
        world.step()
        print(i)
        # vis.render(heatmap_show=False)
    print([our_car.past_traj, other_car_1.past_traj, other_car_2.past_traj])
    frames = [world.render(mode='rgb_array')]

    import contextlib
    with contextlib.redirect_stdout(None):  # disable the pygame import print
        from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(frames, fps=int(1 / world.dt))
    clip.speedx(0.5).write_gif('test_traj.gif', program="ffmpeg")
if __name__ == "__main__":
    main()
