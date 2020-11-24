"""old visualizer (from Dorsa)"""
# flake8: noqa

import math
import pickle
import time
from os.path import dirname, join

import matplotlib.cm
import numpy as np
import pyglet
import pyglet.gl as gl
import pyglet.graphics as graphics
from pyglet.window import key

from interact_drive.car import Car

# Note: Visualizer class is now deprecated in favor of CarVisualizer

default_asset_dir = join(dirname(__file__), "assets")


class Visualizer(object):
    def __init__(
        self, dt=0.5, fullscreen=False, name="car_sim", iters=1000, magnify=1.0
    ):
        self.autoquit = False
        self.frame = None
        self.subframes = None
        self.visible_cars = []
        self.magnify = magnify
        self.camera_center = None
        self.name = name
        self.output = None
        self.iters = iters
        self.obstacles = []
        self.event_loop = pyglet.app.EventLoop()
        self.window = pyglet.window.Window(
            600, 600, fullscreen=fullscreen, caption=name
        )
        # self.grass = pyglet.resource.texture(join(asset_dir, 'grass.png'))
        self.grass = pyglet.image.load(
            join(default_asset_dir, "grass.png")
        ).get_texture()
        self.window.on_draw = self.on_draw
        self.lanes = []
        self.cars = []
        self.dt = dt
        self.anim_x = {}
        self.prev_x = {}
        self.feed_u = None
        self.feed_x = None
        self.prev_t = None
        self.joystick = None
        self.keys = key.KeyStateHandler()
        self.window.push_handlers(self.keys)
        self.window.on_key_press = self.on_key_press
        self.main_car = None
        self.heat = None
        self.heatmap = None
        self.heatmap_valid = False
        self.heatmap_show = False
        self.cm = matplotlib.cm.jet
        self.paused = False
        self.label = pyglet.text.Label(
            "Speed: ",
            font_name="Times New Roman",
            font_size=24,
            x=30,
            y=self.window.height - 30,
            anchor_x="left",
            anchor_y="top",
        )
        self.world = None

        def centered_image(filename):
            # img = pyglet.resource.image(filename)
            img = pyglet.image.load(filename).get_texture()
            img.anchor_x = img.width / 2.0
            img.anchor_y = img.height / 2.0
            return img

        def car_sprite(color, scale=0.15 / 600.0):
            sprite = pyglet.sprite.Sprite(
                centered_image(
                    join(default_asset_dir, "car-{}.png".format(color))
                ),
                subpixel=True,
            )
            sprite.scale = scale
            return sprite

        def object_sprite(name, scale=0.15 / 600.0):
            sprite = pyglet.sprite.Sprite(
                centered_image(join(default_asset_dir, "{}.png".format(name))),
                subpixel=True,
            )
            sprite.scale = scale
            return sprite

        self.sprites = {
            c: car_sprite(c)
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
        self.obj_sprites = {c: object_sprite(c) for c in ["cone", "firetruck"]}

    def use_world(self, world):
        self.cars = [c for c in world.cars]
        self.lanes = [c for c in world.lanes]
        self.obstacles = [c for c in world.obstacles]
        self.world = world
        assert len(self.cars) > 0
        self.main_car = world.cars[0]

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.event_loop.exit()
        if symbol == key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                "screenshots/screenshot-%.2f.png" % time.time()
            )
        if symbol == key.SPACE:
            self.paused = not self.paused
        if symbol == key.T:
            self.heatmap_show = not self.heatmap_show
            if self.heatmap_show:
                self.heatmap_valid = False
        if symbol == key.J:
            joysticks = pyglet.input.get_joysticks()
            if joysticks and len(joysticks) >= 1:
                self.joystick = joysticks[0]
                self.joystick.open()
        if symbol == key.D:
            self.reset()
        if symbol == key.S:
            with open(
                "data/%s-%d.pickle" % (self.name, int(time.time())), "w"
            ) as f:
                pickle.dump((self.history_u, self.history_x), f)
            self.reset()

    def control_loop(self, _=None):
        # print "Time: ", time.time()
        if self.paused:
            return
        if self.iters is not None and len(self.history_x[0]) >= self.iters:
            if self.autoquit:
                self.event_loop.exit()
            return
        if self.feed_u is not None and len(self.history_u[0]) >= len(
            self.feed_u[0]
        ):
            if self.autoquit:
                self.event_loop.exit()
            return
        if (
            self.pause_every is not None
            and self.pause_every > 0
            and len(self.history_u[0]) % self.pause_every == 0
        ):
            self.paused = True
        steer = 0.0
        gas = 0.0
        if self.keys[key.UP]:
            gas += 1.0
        if self.keys[key.DOWN]:
            gas -= 1.0
        if self.keys[key.LEFT]:
            steer += 1.5
        if self.keys[key.RIGHT]:
            steer -= 1.5
        if self.joystick:
            steer -= self.joystick.x * 3.0
            gas -= self.joystick.y
        self.heatmap_valid = False
        for car in self.cars:
            self.prev_x[car] = car.state
        if self.feed_u is None:
            for car in reversed(self.cars):
                car.control(steer, gas)
        else:
            for car, fu, hu in zip(self.cars, self.feed_u, self.history_u):
                car.control = fu[len(hu)]
        for car, hist in zip(self.cars, self.history_u):
            hist.append(car.control)
        for car, hist in zip(self.cars, self.history_x):
            hist.append(car.state)
        self.prev_t = time.time()

    def center(self):
        if self.main_car is None:
            return np.asarray([0.0, 0.0])
        elif self.camera_center is not None:
            return np.asarray(self.camera_center[0:2])
        else:
            return np.array([0, self.anim_x[self.main_car][1]])

    def camera(self):
        o = self.center()
        gl.glOrtho(
            o[0] - 1.0 / self.magnify,
            o[0] + 1.0 / self.magnify,
            o[1] - 1.0 / self.magnify,
            o[1] + 1.0 / self.magnify,
            -1.0,
            1.0,
        )

    # def set_heat(self, f):
    #     x = optimizer.vector(4)
    #     u = optimizer.vector(2)
    #     func = th.function([], f(0, x, u))
    #
    #     def val(p):
    #         x.set_value(np.asarray([p[0], p[1], 0., 0.]))
    #         return func()
    #
    #     self.heat = val

    def draw_heatmap(self):
        if not self.heatmap_show:
            return
        SIZE = (256, 256)
        if not self.heatmap_valid:
            o = self.center()
            x0 = o - np.asarray([1.5, 1.5]) / self.magnify
            x0 = np.asarray(
                [
                    x0[0] - x0[0] % (1.0 / self.magnify),
                    x0[1] - x0[1] % (1.0 / self.magnify),
                ]
            )
            x1 = x0 + np.asarray([4.0, 4.0]) / self.magnify
            x0 = o - np.asarray([1.0, 1.0]) / self.magnify
            x1 = o + np.asarray([1.0, 1.0]) / self.magnify
            self.heatmap_x0 = x0
            self.heatmap_x1 = x1
            vals = np.zeros(SIZE)
            for i, x in enumerate(np.linspace(x0[0], x1[0], SIZE[0])):
                for j, y in enumerate(np.linspace(x0[1], x1[1], SIZE[1])):
                    vals[j, i] = self.heat(np.asarray([x, y]))
            vals = (vals - np.min(vals)) / (np.max(vals) - np.min(vals) + 1e-6)
            vals = self.cm(vals)
            vals[:, :, 3] = 0.7
            vals = (vals * 255.99).astype("uint8").flatten()
            vals = (gl.GLubyte * vals.size)(*vals)
            img = pyglet.image.ImageData(
                SIZE[0], SIZE[1], "RGBA", vals, pitch=SIZE[1] * 4
            )
            self.heatmap = img.get_texture()
            self.heatmap_valid = True
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        gl.glEnable(self.heatmap.target)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glBindTexture(self.heatmap.target, self.heatmap.id)
        gl.glEnable(gl.GL_BLEND)
        x0 = self.heatmap_x0
        x1 = self.heatmap_x1
        graphics.draw(
            4,
            gl.GL_QUADS,
            ("v2f", (x0[0], x0[1], x1[0], x0[1], x1[0], x1[1], x0[0], x1[1])),
            ("t2f", (0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0)),
            # ('t2f', (0., 0., SIZE[0], 0., SIZE[0], SIZE[1], 0., SIZE[1]))
        )
        gl.glDisable(self.heatmap.target)

    def output_loop(self, _):
        if self.frame % self.subframes == 0:
            self.control_loop()
        alpha = float(self.frame % self.subframes) / float(self.subframes)
        for car in self.cars:
            self.anim_x[car] = (1 - alpha) * self.prev_x[
                car
            ] + alpha * car.state
        self.frame += 1

    def animation_loop(self, _):
        t = time.time()
        alpha = min((t - self.prev_t) / self.dt, 1.0)
        for car in self.cars:
            self.anim_x[car] = (1 - alpha) * self.prev_x[
                car
            ] + alpha * car.state

    def draw_lane_surface(self, lane):
        gl.glColor3f(0.4, 0.4, 0.4)
        W = 1000

        graphics.draw(
            4,
            gl.GL_QUAD_STRIP,
            (
                "v2f",
                np.hstack(
                    [
                        lane.p - lane.m * W - 0.5 * lane.w * lane.n,
                        lane.p - lane.m * W + 0.5 * lane.w * lane.n,
                        lane.q + lane.m * W - 0.5 * lane.w * lane.n,
                        lane.q + lane.m * W + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )

    def draw_lane_lines(self, lane):
        gl.glColor3f(1.0, 1.0, 1.0)
        W = 1000
        graphics.draw(
            4,
            gl.GL_LINES,
            (
                "v2f",
                np.hstack(
                    [
                        (lane.p + lane.q) / 2
                        - lane.m * W
                        - 0.5 * lane.w * lane.n,
                        (lane.p + lane.q) / 2
                        + lane.m * W
                        - 0.5 * lane.w * lane.n,
                        (lane.p + lane.q) / 2
                        - lane.m * W
                        + 0.5 * lane.w * lane.n,
                        (lane.p + lane.q) / 2
                        + lane.m * W
                        + 0.5 * lane.w * lane.n,
                    ]
                ),
            ),
        )

    def draw_car(self, x, color="yellow", opacity=255):
        sprite = self.sprites[color]
        sprite.x, sprite.y = x[0], x[1]
        sprite.rotation = -x[3] * 180.0 / math.pi
        sprite.opacity = opacity
        sprite.draw()

    def draw_object(self, obj):
        print(obj.name)
        sprite = self.obj_sprites[obj.name]
        sprite.x, sprite.y = obj.state[0], obj.state[1]
        sprite.rotation = obj.state[3] if len(obj.state) >= 4 else 0.0
        sprite.draw()

    def on_draw(self):
        self.window.clear()
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glPushMatrix()
        gl.glLoadIdentity()
        self.camera()
        gl.glEnable(self.grass.target)
        gl.glEnable(gl.GL_BLEND)
        gl.glBindTexture(self.grass.target, self.grass.id)
        W = 10000.0
        graphics.draw(
            4,
            gl.GL_QUADS,
            ("v2f", (-W, -W, W, -W, W, W, -W, W)),
            ("t2f", (0.0, 0.0, W * 5.0, 0.0, W * 5.0, W * 5.0, 0.0, W * 5.0)),
        )
        gl.glDisable(self.grass.target)
        for lane in self.lanes:
            self.draw_lane_surface(lane)
        for lane in self.lanes:
            self.draw_lane_lines(lane)
        for obj in self.obstacles:
            self.draw_object(obj)
        for car in self.cars:
            if car != self.main_car and car not in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        if self.heat is not None:
            self.draw_heatmap()
        for car in self.cars:
            if car == self.main_car or car in self.visible_cars:
                self.draw_car(self.anim_x[car], car.color)
        gl.glPopMatrix()
        if isinstance(self.main_car, Car):
            self.label.text = "Speed: %.2f" % self.anim_x[self.main_car][2]
            self.label.draw()
        if self.output is not None:
            pyglet.image.get_buffer_manager().get_color_buffer().save(
                self.output.format(self.frame)
            )
        # pyglet.image.get_buffer_manager().get_color_buffer().save(
        #    '%s-%d.png' % (self.name, int(time.time())))

    def reset(self):
        self.prev_t = time.time()
        for car in self.cars:
            self.prev_x[car] = car.state
            self.anim_x[car] = car.state
        self.paused = True
        self.history_x = [[] for car in self.cars]
        self.history_u = [[] for car in self.cars]

    def visualize(self):
        """Dispatch pyglet event manually"""
        self.control_loop(None)
        self.animation_loop(None)
        pyglet.clock.tick()
        for window in pyglet.app.windows:
            window.switch_to()
            window.dispatch_events()
            window.dispatch_event("on_draw")
            window.flip()

    def run(self, filename=None, pause_every=None):
        self.pause_every = pause_every
        self.reset()
        if filename is not None:
            with open(filename) as f:
                self.feed_u, self.feed_x = pickle.load(f)
        if self.output is None:
            pyglet.clock.schedule_interval(self.animation_loop, 0.02)
            pyglet.clock.schedule_interval(self.control_loop, self.dt)
        else:
            self.paused = False
            self.subframes = 6
            self.frame = 0
            self.autoquit = True
            pyglet.clock.schedule(self.output_loop)
        self.event_loop.run()
