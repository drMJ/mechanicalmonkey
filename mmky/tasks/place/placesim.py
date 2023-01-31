import math
import numpy as np
import os
import pybullet as pb
from mmky import SimScene
from mmky import primitives


#TODO:  OUTDATED

class PlaceSim(SimScene):
    def __init__(self, robot, obs_res, workspace, obj_size=0.05, obj_kind="box", rand_colors=False, rand_textures=False, cameras={}, **kwargs):
        super().__init__(robot, obs_res, workspace, cameras, **kwargs)
        self.obj_size = obj_size
        self.obj_kind = obj_kind
        self.cup_model = "cup_no_tex"
        self.cup_size = np.array([0.100, 0.100, 0.150]) # this must match the model
        self.rand_colors = rand_colors
        self.rand_textures = rand_textures

    def reset(self, **kwargs):
        super().reset(**kwargs)
        
        obj_pose = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height]
        color = [random.random(), random.random(), random.random(), 1] if self.rand_colors else [1, 0.75, 0.25, 1]
        tex = random.choice(self.textures) if self.rand_textures else None
    
        if self.obj_kind == "box":
            size = [self.obj_size] * 3
            self.make_box(size, obj_pose, color=color, tex=tex, mass=0.1, tag="obj")
        else:
            self.make_ball(self.obj_size/2, obj_pose, color=color, tex=tex, mass=0.1, tag="obj", restitution=0.2)

        color = [random.random(), random.random(), random.random(), 1] if self.rand_colors else [1, 1, 1, 1]
        tex = random.choice(self.textures) if self.rand_textures else None
        mesh = os.path.join("Cup", self.cup_model + ".obj")
        vhcad = os.path.join("Cup", self.cup_model + "_vhacd.obj")
        position = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius) + [self.workspace_height + self.cup_size[2]]
        
        self.load_obj(mesh_file=mesh,
            vhacd_file=vhcad,
            position=position,
            orientation=pb.getQuaternionFromEuler([math.pi,0,0]),
            scale=[0.001]*3, # mm
            mass=1,
            tex=tex,
            color=color,
            tag="cup", 
            restitution=0)

    def eval_state(self, world_state):
        rew = 0
        obj_pos = world_state["obj"]["position"]
        cup_pos = world_state["cup"]["position"]
        delta = cup_pos - obj_pos

        in_cup = obj_pos[2] > self.workspace_height \
                 and obj_pos[2] < self.cup_size[2] \
                 and math.sqrt(delta[0] * delta[0] + delta[1] * delta[1]) < self.cup_size[0] / 2
                 
        rew = 1 if in_cup else 0
        success = (rew == 1) and not self.robot.has_object
        done = success
        return rew, success, done
