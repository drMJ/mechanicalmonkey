import torch

class SimpleAgent:
    def __init__(self):
        self._cmd = {'cmd':'move_and_grasp', 'args': 
            {'max_speed':0.5, 'max_acc':0.5,'force_limit':None, 'grasp_speed':0, 'grasp_force':0}
        }

    def start_episode(self):
        pass

    def get_action(self, observation):
        self._cmd['args']['arm_target'] = observation['tool_pose']
        self._cmd['args']['grasp_target'] = observation['hand_position']
        return self._cmd
    
    def end_episode(self):
        pass

            

            
        


    
