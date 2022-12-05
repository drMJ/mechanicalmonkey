import argparse
import random
from roman import Robot, GraspMode, Joints, Tool, JointSpeeds
from roman.ur import arm
from roman.rq import hand
from mmky import RomanEnv, RealScene, SimScene
from mmky.writers import HDF5Writer
from mmky.tasks import *
from mmky.env import RomanEnv, ProtoSkillEnv, MacroActionEnv
import os
import yaml

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run a mechanicalmonkey agent.")
    parser.add_argument(
        "-w",
        "--workcell",
        type=str,
        default="ws1",
        help="The workcell definition (robot and camera setup). Can be the name of one of the config files in the workcell folder, e.g. 'ws1', of a full config file path.")

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        #nargs='+',
        required=True,
        help="The task to execute. Must resolve to imported task/scene classes, e.g. 'PickReal' or 'StackSim'.")

    parser.add_argument(
        "-a",
        "--agent",
        type=str,
        required=True,
        help="The agent to run. Must be resolve to an imported agent class, e.g. 'PourExpert'.")

    parser.add_argument(
        "-e",
        "--env",
        type=str,
        required=True,
        help="The type of gym environment to instantiate. Must resolve to imported env classes, e.g. RomanEnv, MacroActionEnv, ProtoSkillEnv")

    parser.add_argument(
        "-c",
        "--episode_count",
        type=int,
        default=1,
        help="The number of episodes to run.")

    parser.add_argument(
        "--seed",
        type=int,
        help="The seed for the random number generators.")

    parser.add_argument(
        "--cpu_rendering",
        action="store_true",
        default=False,
        help="Sim only. Whether to use CPU or GPU for rendering. Defaults to GPU rendering. GPU rendering in headless mode uses EGL (Linux only).")

    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Sim only. Whether to run in headless mode or show full GUI. In headless mode, multiple instances can be running in parallel.")

    return parser

def create_env(env_type, task_type, workcell, seed=5, gpu_rendering=True, headless=False):
    # load workspace definition
    rootdir = os.path.dirname(__file__)
    templates = [[workcell], [rootdir, workcell], [rootdir, workcell], [rootdir, workcell+'.yaml'], [rootdir, 'workcells', workcell], [rootdir, 'workcells', workcell+'.yaml']]
    for template in templates:
        ws_config_file = os.path.join(*template)
        if os.path.isfile(ws_config_file):
            break
    if not os.path.isfile(ws_config_file):
        raise Exception(f"Workcell {workcell} could not be resolved to a yaml config file.")
    with open(ws_config_file) as f:
        ws_config = yaml.safe_load(f)

    # instantiate the scene object
    try:
        task_type = eval(task_type) if isinstance(task_type, str) else task_type
        scene = task_type(**ws_config, use_gpu=gpu_rendering, use_gui=not headless)
    except Exception as e:
        raise Exception(f"Task name {task_type} could not be resolved to a valid scene class or function. {e}")

    # instantiate the environment
    #env = env_type(scene, seed, **scene.env_config)
    try:
        env_type = eval(env_type) if isinstance(env_type, str) else env_type
        env = env_type(scene, seed, **scene.env_config)
    except Exception as e:
        raise Exception(f"Env name {env_type} could not be resolved to a valid env class or function. {e}")
    return env


def create_agent(agent):
    # instantiate the agent as needed
    try:
        agent = eval(agent)() if isinstance(agent, str) else agent
    except Exception as e:
        raise Exception(f"Agent name {agent} could not be resolved to a valid agent or expert class. {e}")

    return agent

def run(env, agent, writer, episodes):
    for eix in range(episodes):
        done = False
        agent.start_episode()
        obs = env.reset()
        writer.start_episode(obs)
        while not done:
            act = agent.get_action(obs)
            if act is None:
                break
            (obs, rew, done, info) = env.step(act, logger=writer.log)
        agent.end_episode()
        writer.end_episode()
    env.close()
    writer.close()

def parse_and_run():
    args = create_parser().parse_args()
    env = create_env(args.env, args.task, args.workcell, args.seed, not args.cpu_rendering, args.headless)
    agent = create_agent(args.agent)
    writer = HDF5Writer(f"{args.task}_{args.env}_{args.agent}_{args.workcell}")
    run(env, agent, writer, args.episode_count)


if __name__ == '__main__':
    parse_and_run()
    # python run.py -w ws1 -a PickExpert -t PickReal -e ProtoSkillEnv -c 1000