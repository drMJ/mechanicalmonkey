import math
import random
import numpy as np
import os
import pybullet as p
from mmky import SimScene
from mmky import primitives

CUPMODELS = [
    # name, size in m (must match the obj file)
    ("cup_no_tex", [0.100, 0.100, 0.150]), # this is also the target cup's model
    ("cup_tex", [0.100, 0.100, 0.150]),
    ("barrel_cup", [0.080, 0.080, 0.100]),
    ("cone_cup", [0.100, 0.100, 0.100]),
    ("cone_cup_modified", [0.100, 0.100, 0.100]),
    ("cone_cup_smooth", [0.120, 0.120, 0.100]),
    ("cup", [0.100, 0.100, 0.150]),
    ("cup_wine", [0.100, 0.100, 0.100]),
    ("gift_box", [0.100, 0.100, 0.100]),
    ("mug", [0.150, 0.100, 0.100]),
    ("ring_cup", [0.100, 0.100, 0.100]),
    ("rocket_cup", [0.100, 0.100, 0.100]),
    ("round_cup", [0.100, 0.100, 0.100]),
    ("square_cup", [0.140, 0.090, 0.100]),
    ("star_cup", [0.130, 0.130, 0.100]),
    ("trophy_cup", [0.080, 0.050, 0.100]),
    ("twisted_cup", [0.120, 0.120, 0.100]),
]

TARGET_CUP_MODEL = 0
OBJ_MODEL_UNIT = 0.001 # obj files are in mm, we need to convert to meters
CUP_BOTTOM_LIMIT = 0.05

class PourSim(SimScene):
    def __init__(self,
                 robot,
                 obs_res,
                 workspace,
                 ball_count=3,
                 ball_radius=0.02,
                 rand_size=False,
                 rand_tex=False,
                 rand_mesh=False,
                 rand_light=False,
                 cameras={}, 
                 **kwargs):
        super().__init__(robot=robot, obs_res=obs_res, cameras=cameras, workspace=workspace, **kwargs)
        self.ball_radius = ball_radius
        self.rand_size = rand_size
        self.rand_tex = rand_tex
        self.rand_mesh = rand_mesh
        self.ball_count = ball_count
        self.rand_light = rand_light

    def reset(self, **kwargs):
        min_angle_in_rad, max_angle_in_rad = self.workspace_span
        min_dist, max_dist = self.workspace_radius

        sx, sy = primitives.generate_random_xy(min_angle_in_rad,
                                        (max_angle_in_rad + min_angle_in_rad) / 2 - 0.1,
                                        min_dist,
                                        max_dist)

        tx, ty = primitives.generate_random_xy((max_angle_in_rad + min_angle_in_rad) / 2 + 0.1,
                                        max_angle_in_rad,
                                        min_dist,
                                        max_dist)
        self.cup_position = np.array([sx, sy, self.workspace_height])
        self.target_cup_position = np.array([tx, ty, self.workspace_height])

        self.robot.active_force_limit = (None, None)
        super().reset(**kwargs)

    def move_home(self, home_pose):
        primitives.pivot_dxdy(self.robot, 0, 0, 0, reference_pose=home_pose)

    def setup_scene(self):
        super().setup_scene()
        self.target_cup = self._load_target_cup(self.target_cup_position, p.getQuaternionFromEuler([0, 0, math.pi * random.random()]))


        orientation = p.getQuaternionFromEuler([0, 0, (random.random() - 0.5) * math.pi * 4])
        (self.cup, self.cup_name, self.cup_size) = self._load_any_cup(self.cup_position,
                                                                      orientation,
                                                                      rand_size = self.rand_size,
                                                                      rand_tex=self.rand_tex,
                                                                      rand_mesh=self.rand_mesh,
                                                                      tag="source")

        self.balls = np.zeros((self.ball_count), int)
        for i in range(self.ball_count):
            self.balls[i] = self.make_ball(self.ball_radius,
                                           self.cup_position + [0, 0, CUP_BOTTOM_LIMIT + 0.05],
                                           mass=0.005,
                                           color=[1, 0.75, 0.25, 1],
                                           restitution=0,
                                           spinningFriction=0.2)
            # let the ball fall in the cup, so we can create another one at the same position
            for i in range(5):
                p.stepSimulation()

        return self.get_world_state()

    def get_ball_counts(self):
        spilled = 0
        in_target_cup = 0
        in_source_cup = 0
        for ball in self.balls:
            ball_pos = np.array(p.getBasePositionAndOrientation(ball)[0])
            # in target cup?
            bpos = ball_pos - self.target_cup_position
            if bpos[2] > self.workspace_height \
               and bpos[2] < self.target_cup_size[2] \
               and math.sqrt(bpos[0] * bpos[0] + bpos[1] * bpos[1]) < self.target_cup_size[0] / 2:
                in_target_cup = in_target_cup + 1
            elif bpos[2] < CUP_BOTTOM_LIMIT:
                # still in source cup? (this is approximate - only works while the cup is upright)
                bpos = ball_pos - self.cup_position
                if bpos[2] > 0 \
                   and bpos[2] < self.cup_size[2] \
                   and math.sqrt(bpos[0] * bpos[0] + bpos[1] * bpos[1]) < self.cup_size[0] / 2:
                    in_source_cup = in_source_cup + 1
                else:
                    spilled = spilled + 1

        return {"poured": in_target_cup, "remaining": len(self.balls) - in_target_cup - spilled, "spilled": spilled}

    def get_world_state(self, force_state_refresh=False):
        ws = super().get_world_state(force_state_refresh)
        ws["ball_data"] = self.get_ball_counts()
        ws["target"]["size"] = np.array(self.target_cup_size)
        ws["source"]["size"] = np.array(self.cup_size)
        return ws

    def eval_state(self, world_state):
        rew = world_state["ball_data"]["poured"]
        success = rew == self.ball_count
        done = world_state["ball_data"]["remaining"] == 0
        return rew, success, done

    def _load_any_cup(self, position, orientation, rand_size=False, rand_color=True, rand_tex=False, rand_mesh=False, tag=None):
        model = random.choice(CUPMODELS) if rand_mesh else CUPMODELS[TARGET_CUP_MODEL]
        scale = 1
        if rand_size:
            scale += (random.random() - 0.5) / 2

        id = self._load_cup(model[0],
                            position=position,
                            orientation=orientation,
                            scale=[scale * OBJ_MODEL_UNIT] * 3,
                            mass=.05,
                            rand_tex=rand_tex,
                            rand_color=rand_color,
                            tag=tag)
        return (id, model[0], np.array(model[1]) * scale)

    def _load_target_cup(self, position=[0, 0, 0], orientation=[0, 0, 0, 1], rand_color=False, rand_tex=False):
        model = CUPMODELS[TARGET_CUP_MODEL]
        self.target_cup_size = np.array(model[1])
        return self._load_cup(model[0],
                              position=position,
                              orientation=orientation,
                              scale=[OBJ_MODEL_UNIT] * 3,
                              mass=1,
                              rand_tex=rand_tex,
                              rand_color=rand_color,
                              tag="target")

    def _load_cup(self, name, position, orientation, scale, mass, rand_color=False, rand_tex=False, tag=None):
        color = [random.random(), random.random(), random.random(), 1] if rand_color else [1, 1, 1, 1]
        tex = random.choice(self.textures) if rand_tex else None
        mesh = os.path.join("Cup", name + ".obj")
        vhcad = os.path.join("Cup", name + "_vhacd.obj")
        return self.load_obj(mesh_file=mesh,
                             vhacd_file=vhcad,
                             position=position,
                             orientation=orientation,
                             scale=scale,
                             mass=mass,
                             tex=tex,
                             color=color,
                             tag=tag, 
                             restitution=0)
