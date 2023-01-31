from mmky import RealScene
from mmky import primitives
import random
GRASP_HEIGHT = 0.12
RELEASE_HEIGHT = 0.04

class PlaceReal(RealScene):
    def reset(self, **kwargs):
        super().reset(**kwargs)
        ws = self.get_world_state(force_state_refresh=False)
        while len(ws) == 1:
            self.robot.pinch(0)
            # the object is stacked, unstack
            target = self.robot.tool_pose
            target[:2] = ws["cup"]["position"][:2]
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)

             # pick the object
            primitives.pick(self.robot, self.workspace_height + GRASP_HEIGHT, pre_grasp_size=0, max_speed=0.5, max_acc=0.5)

            # pick a random spot and set the object there
            new_spot = target.clone()
            x, y = primitives.generate_random_xy(*self.workspace_span, *self.workspace_radius)
            new_spot[:2] = x, y
            self.robot.move(new_spot, timeout=10, max_speed=0.5, max_acc=0.5)
            primitives.place(self.robot, self.workspace_height + RELEASE_HEIGHT, max_speed=0.5, max_acc=0.5)

            # go back over cup
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)

            # lower the hand and move sideways at random 
            home_z = target[2]
            target[2] = self.workspace_height + GRASP_HEIGHT / 2
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            self.robot.grasp(60)
            dx = (random.random() - 0.5) / 10
            dy = (random.random() - 0.5) / 10
            target[:2] += dx, dy
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
            self.robot.grasp(0)
            target[2] = home_z
            self.robot.move(target, timeout=10, max_speed=0.5, max_acc=0.5)
           

            ws = self.get_world_state(force_state_refresh=True)

    def get_world_state(self, force_state_refresh):
        ws = {}
        raw_state = super().get_world_state(force_state_refresh)
        for k,v in raw_state.items():
            #TODO fix container critera
            if v["size"][2] > 0.06:
                # this is a cup
                ws["cup"] = v          
            else:
                ws["obj"] = v             
        return ws

    def eval_state(self, obs):
        rew = 2 - len(world_state)
        success = rew == 1
        done = success
        return rew, success, done
