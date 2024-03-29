import sys
import math
import numpy as np

import Box2D
from Box2D.b2 import fixtureDef
from Box2D.b2 import polygonShape
from Box2D.b2 import contactListener
import rendering

import gym
from gym import spaces
from car_dynamics import Car
from gym.utils import seeding, EzPickle
from copy import deepcopy
import pyglet, random
import matplotlib.pyplot as plt
pyglet.options["debug_gl"] = False
from pyglet import gl
from neural_netwok import NeuralNetwork

STATE_W = 96  # less than Atari 160x192
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 700 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 2000 / SCALE  # Game over boundary
FPS = 30  # Frames per second
ZOOM = 1    #2.7  # Camera zoom
ZOOM_FOLLOW = False  # Set to False for fixed view (don't use zoom)


TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 60 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4

ROAD_COLOR = [0.4, 0.4, 0.4]

FRICTION = 0.5
MAX_ANGLE = 90.0
MIN_SPEED = 0.03
MAX_SPEED = 50.0
FUTURE_SIGHT = 10
TIMEPENALTY = 0.4

class FrictionDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[0] = ROAD_COLOR[0]
        tile.color[1] = ROAD_COLOR[1]
        tile.color[2] = ROAD_COLOR[2]
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            self.env.car.curren_tile = tile.road_id
            if not tile.road_visited:
                tile.road_visited = True
                self.env.reward += 1000.0 / len(self.env.track)
                self.env.tile_visited_count += 1
        else:
            obj.tiles.remove(tile)


class CarRacing(gym.Env, EzPickle):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second": FPS,
    }

    def __init__(self, verbose=1):
        EzPickle.__init__(self)
        self.seed()
        self.contactListener_keepref = FrictionDetector(self)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.viewer = None
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.car = None
        self.reward = 0.0
        self.prev_reward = 0.0
        self.verbose = verbose
        gaussian = np.random.normal(FRICTION, 0.2, 1000)
        self.friction_values = gaussian[(gaussian > 0.1) & (gaussian < 1.0)]
        self.friction_change = random.randrange(5, 15)/10.0
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        self.action_space = spaces.Box(
            np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32
        )  # steer, gas, brake

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        self.car.destroy()

    def _create_track(self, noises=[], rads=[]):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            if len(noises) <= c:
                noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
                noises.append(noise)
            else:
                noise = noises[c]
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            if len(rads) <= c:
                rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
                rads.append(rad)
            else:
                rad = rads[c]

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        for i in range(len(track)):
            _, beta1, x1, y1 = track[i]
            _, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.road_id = i
            t.center_p = (x1, y1)
            t.userData = t
            c = 0.01 * (i % 3)
            t.color = [ROAD_COLOR[0] + c, ROAD_COLOR[1] + c, ROAD_COLOR[2] + c]
            t.road_visited = False
            if i % 10 == 0: self.friction_change = random.randrange(5, 15)/10.0
            t.road_friction = np.random.choice(self.friction_values)*self.friction_change
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (1, 1, 1) if i % 2 == 0 else (1, 0, 0))
                )
        self.track = track
        self.track_pack = {'noises':noises, 'rads':rads}
        return True

    def reset(self, track=None):
        self._destroy()
        self.reward = 0.0
        self.prev_reward = 0.0
        self.tile_visited_count = 0
        self.t = 0.0
        self.road_poly = []
        self.track_pack = None
        if track is None:
            while True:
                success = self._create_track([], [])
                if success:
                    break
        else:
            self._create_track(track['noises'], track['rads'])
        self.car = Car(self.world, *self.track[0][1:4])

        return self.step(None)[0]

    def step(self, action):
        if action is not None:
            self.car.last_action = action
            if self.tile_visited_count <= 4 and action[1] < 0.1:
                action[1] = 0.1
                action[2] = 0
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])
        self.car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS
        self.calcInputs()
        self.state = self.render("state_pixels")
        step_reward = 0
        done = False
        if action is not None:  # First step without action, called from reset()
            self.reward -= TIMEPENALTY
            if self.car.drifting:
                self.reward += 0#1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100
            if abs(self.car.offset) > 1:
                self.reward -= 1
                done = True
            if abs(self.car.angle) == 1:
                self.reward -= 1
            if speed < MIN_SPEED:
                if self.tile_visited_count > 4:
                    done = True #too slow
            if self.reward < 0:
                done = True
            #drift reward

        return self.state, step_reward, done, {}

    def calcInputs(self):
        #calculate stuff
        current_tile_center = self.road[self.car.curren_tile].center_p
        previous_tile_center = self.road[self.car.curren_tile - 2].center_p
        self.car.offset = self.calcOffset(previous_tile_center, current_tile_center, self.car.hull.position)/TRACK_WIDTH
        rear_tire = self.car.wheels[2].position
        front_tire = self.car.wheels[0].position
        if self.car.angle > 1:
            self.car.angle = 1
        elif self.car.angle < -1:
            self.car.angle = -1
        if (self.car.curren_tile + FUTURE_SIGHT) < len(self.road):
            front_tire = self.road[self.car.curren_tile + int(FUTURE_SIGHT/2)].center_p
            future_tile1_center = self.road[self.car.curren_tile + int(FUTURE_SIGHT/2)].center_p
            future_tile2_center = self.road[self.car.curren_tile + FUTURE_SIGHT].center_p
        else:
            front_tire = self.road[-1].center_p
            future_tile1_center = current_tile_center
            future_tile2_center = front_tire

        # set input variables
        self.car.angle = self.calcAngle(previous_tile_center, current_tile_center, rear_tire, front_tire)/MAX_ANGLE
        self.car.curve1 = self.calcAngle(previous_tile_center, current_tile_center, current_tile_center, front_tire)/MAX_ANGLE
        self.car.curve2 = self.calcAngle(previous_tile_center, current_tile_center, future_tile1_center, future_tile2_center)/MAX_ANGLE
        self.car.angle_offset = self.calcAngle(self.car.wheels[2].position, self.car.wheels[0].position, self.car.hull.position, future_tile1_center)/MAX_ANGLE
        self.car.slip_rate = self.calcSlipRate()
        self.car.yaw_velocity = self.car.hull.angularVelocity/400.0 #tested
        self.car.speed = np.linalg.norm(self.car.hull.linearVelocity)/MAX_SPEED

    def calcSlipRate(self):
        f_speed = (self.car.wheels[0].omega + self.car.wheels[1].omega)/2
        r_speed = (self.car.wheels[2].omega + self.car.wheels[3].omega)/2
        if r_speed > f_speed:
            traction = f_speed/r_speed
            slip = 1-traction
            return slip
        else:
            return 0

    def calcOffset(self, p1, p2, p3): #normalized with tack width
        p1 = np.array(p1)
        p2 = np.array(p2)
        p3 = np.array(p3)
        side = ((p3[0]-p2[0])*(p2[1]-p1[1]) - (p2[0]-p1[0])*(p3[1]-p2[1]))
        side /= -abs(side)
        offset = np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)*side
        return offset

    def calcAngle(self, p1, p2, p3, p4):
        v1 = -np.array(p1) + np.array(p2)
        v2 = -np.array(p3) + np.array(p4)
        signed_angle = math.atan2( v1[0]*v2[1]- v1[1]*v2[0], v1[0]*v2[0] + v1[1]*v2[1] )
        return np.rad2deg(signed_angle)

    def debug_line(self, p1, p2, color=(0.0,  255, 0.0), linewidth=2):
        #debug line
        class Particle:
            pass
        p = Particle()

        p.poly = [(p1[0], p1[1]), (p2[0], p2[1]) ]
        p.color = color
        self.viewer.draw_polyline(p.poly, color=p.color, linewidth=linewidth)


    def render(self, mode="human"):
        assert mode in ["human", "state_pixels", "rgb_array"]
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self.score_label = pyglet.text.Label("0000",font_size=36,x=20,y=WINDOW_H * 2.5 / 40.00,anchor_x="left",anchor_y="center",color=(255, 255, 255, 255),)
            self.throttle_label = pyglet.text.Label("0000",font_size=36,x=150,y=WINDOW_H * 2.5 / 40.00,anchor_x="left",anchor_y="center",color=(0, 255, 0, 255),)
            self.brake_label = pyglet.text.Label("0000",font_size=36,x=300,y=WINDOW_H * 2.5 / 40.00,anchor_x="left",anchor_y="center",color=(255, 0, 0, 255),)
            self.steer_label = pyglet.text.Label("0000",font_size=36,x=450,y=WINDOW_H * 2.5 / 40.00,anchor_x="left",anchor_y="center",color=(0, 0, 255, 255),)
            self.speed_label = pyglet.text.Label("0000",font_size=36,x=580,y=WINDOW_H * 2.5 / 40.00,anchor_x="left",anchor_y="center",color=(0, 0, 0, 255),)
            self.transform = rendering.Transform()
        if "t" not in self.__dict__:
            return  # reset() not called yet
        # Animate zoom first second:
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = self.car.hull.position[0]
        scroll_y = self.car.hull.position[1]
        angle = -self.car.hull.angle
        vel = self.car.hull.linearVelocity
        if np.linalg.norm(vel) > 0.5:
            angle = math.atan2(vel[0], vel[1])
        self.transform.set_scale(zoom, zoom)
        self.transform.set_translation(
            WINDOW_W / 2
            - (scroll_x * zoom * math.cos(angle) - scroll_y * zoom * math.sin(angle)),
            WINDOW_H / 4
            - (scroll_x * zoom * math.sin(angle) + scroll_y * zoom * math.cos(angle)),
        )
        self.transform.set_rotation(angle)
        self.car.draw(self.viewer, mode != "state_pixels")
        # draw input lines
        p2 = self.road[self.car.curren_tile].center_p
        p1 = self.road[self.car.curren_tile - 2].center_p
        self.debug_line(p1, p2)
        p3 = p2
        if (self.car.curren_tile + FUTURE_SIGHT) < len(self.road):
            p4 = self.road[self.car.curren_tile + int(FUTURE_SIGHT/2)].center_p
            p5 = self.road[self.car.curren_tile + int(FUTURE_SIGHT/2)].center_p
            p6 = self.road[self.car.curren_tile + FUTURE_SIGHT].center_p
        else:
            p4 = self.road[-1].center_p
            p5 = p3
            p6 = p4
        self.debug_line(p3, p4, color=(50, 50, 0))
        self.debug_line(p5, p6, color=(0, 100, 50))
        self.debug_line(self.car.hull.position, p5, color=(50, 0, 100))
        # draw output indicators
        self.debug_line((300, 300), (350, 350), color=(50, 50, 0))

        arr = None
        win = self.viewer.window
        win.switch_to()
        win.dispatch_events()

        win.clear()
        t = self.transform
        if mode == "rgb_array":
            VP_W = VIDEO_W
            VP_H = VIDEO_H
        elif mode == "state_pixels":
            VP_W = STATE_W
            VP_H = STATE_H
        else:
            pixel_scale = 1
            if hasattr(win.context, "_nscontext"):
                pixel_scale = (
                    win.context._nscontext.view().backingScaleFactor()
                )  # pylint: disable=protected-access
            VP_W = int(pixel_scale * WINDOW_W)
            VP_H = int(pixel_scale * WINDOW_H)

        gl.glViewport(0, 0, VP_W, VP_H)
        t.enable()
        self.render_road()
        for geom in self.viewer.onetime_geoms:
            geom.render()
        self.viewer.onetime_geoms = []
        t.disable()
        self.render_indicators(WINDOW_W, WINDOW_H)

        if mode == "human":
            win.flip()
            return self.viewer.isopen

        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(VP_H, VP_W, 4)
        arr = arr[::-1, :, 0:3]

        return arr

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def render_road(self):
        colors = [0.4, 0.8, 0.4, 1.0] * 4
        polygons_ = [
            +PLAYFIELD,
            +PLAYFIELD,
            0,
            +PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            -PLAYFIELD,
            0,
            -PLAYFIELD,
            +PLAYFIELD,
            0,
        ]

        k = PLAYFIELD / 20.0
        colors.extend([0.4, 0.9, 0.4, 1.0] * 4 * 20 * 20)
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                polygons_.extend(
                    [
                        k * x + k,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + 0,
                        0,
                        k * x + 0,
                        k * y + k,
                        0,
                        k * x + k,
                        k * y + k,
                        0,
                    ]
                )
        for poly, color in self.road_poly:
            colors.extend([color[0], color[1], color[2], 1] * len(poly))
            for p in poly:
                polygons_.extend([p[0], p[1], 0])

        vl = pyglet.graphics.vertex_list(
            len(polygons_) // 3, ("v3f", polygons_), ("c4f", colors)  # gl.GL_QUADS,
        )
        vl.draw(gl.GL_QUADS)
        vl.delete()

    def render_indicators(self, W, H):
        self.score_label.text = "%04i" % self.reward
        self.throttle_label.text = str(round(self.car.last_action[1], 1))
        self.brake_label.text = str(round(self.car.last_action[2], 1))
        self.steer_label.text = str(round(self.car.last_action[0], 1))
        self.speed_label.text = str(round(self.car.speed, 1))
        self.score_label.draw()
        self.throttle_label.draw()
        self.brake_label.draw()
        self.steer_label.draw()
        self.speed_label.draw()
        
def controller(output):
    a = np.array([0.0, 0.0, 0.0])
    throttle = output[1]
    if throttle > 0:
        a[1] = throttle
    else:
        a[2] = -throttle
    a[0] = output[0]
    return a

# ----------------------MAIN----------------------

if __name__ == "__main__":
    from pyglet.window import key
    # training parameters
    MutationChance = 0.6
    MutationStrength = 0.7
    genSize = 100
    show = True # speed up the simulation by disabling rendering
    def key_press(k, mod):
        global restart
        global MutationChance
        global MutationStrength
        global genSize
        global show
        if k == key.ENTER:
            restart = True
        if k == key.LEFT:
            MutationChance -= 0.1
            MutationStrength -= 0.1
        if k == key.RIGHT:
            MutationChance += 0.1
            MutationStrength += 0.1
        if k == key.UP:
            genSize += 5
        if k == key.DOWN:
            genSize -= 5
        if k == key.S: 
            show = not show
    # initialize environment
    env = CarRacing()
    env.render()
    env.viewer.window.on_key_press = key_press
    isopen = True
    # initialize neural networks
    networks = []
    if False: # use old networks
        old_networks = np.load('20.5.networks.npy', allow_pickle=True)#[[22, 20, 33, 19, 16, 48, 66, 67,  8,  6]][[7]]
        for i in range(genSize):
            current_network = deepcopy(np.random.choice(old_networks))
            current_network.mutate(MutationChance, MutationStrength)
            networks.append(current_network)
    else:
        for i in range(genSize):
            current_network = NeuralNetwork(2, [4], 2)
            # current_network.network = np.load('ok_agent_low_fric.npy', allow_pickle=True)
            current_network.mutate(MutationChance, MutationStrength)
            networks.append(current_network)
    # start driving simulation and training
    current_network = networks[0]
    rounds = 0
    gen = 1
    avgs = []
    while isopen:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            #input("step") # frame by frame debug
            speed = env.car.speed
            offset = env.car.offset
            body_angle = env.car.angle
            curve1 = env.car.curve1
            curve2 = env.car.curve2
            slip = env.car.slip_rate
            yaw = env.car.yaw_velocity
            angle_offset = env.car.angle_offset
            inputs = (speed, angle_offset) # choose inputs for neural network. Match the input nodes in the neural network.

            output = current_network.forward_propagate(inputs)
            actions = controller(output)

            s, r, done, info = env.step(actions)

            total_reward += r
            if done:
                current_network.fitness = total_reward
                avgs.append(total_reward)
                current_network = networks[rounds]
                rounds += 1
                if rounds % 5 == 0 and rounds >= 5:
                    avgfit = np.array(avgs).mean()
                    avgs = []
                    print("avg reward {:+0.2f}, mutation: ({:0.1f}, {:0.1f}), genSize: {}, generation: {}, round: {}".format(
                        avgfit, MutationChance, MutationStrength, genSize, gen, rounds)
                        )
            
            if rounds >= genSize: # create a pool of neural networks and generate a new generation
                pool = []
                for i in range(len(networks)):
                    pool_size = int(networks[i].fitness)
                    for ii in range(pool_size):
                        pool.append(i)
                selection = np.random.choice(pool, genSize)
                networks = np.array(networks)
                temp = networks[selection]
                networks = []
                for i in range(temp.shape[0]):
                    networks.append(deepcopy(temp[i]))
                for i in range(len(networks)):
                    networks[i].mutate(MutationChance, MutationStrength)
                rounds = 0
                gen += 1
            if show:
                isopen = env.render()
            steps += 1
            if done or restart or isopen == False:
                break
    env.close()

