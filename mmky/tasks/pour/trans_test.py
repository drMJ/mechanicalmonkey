'''Given a policy, evaluate on several episodes'''
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import time

from mmky.tasks.pour.pourenv import PourEnv

from transformers import TransPolicy
import math
import os
import argparse

img_x = 224
img_y = 224
step_size = 0.01
max_traj_length=300
#scene = Scene(use_gui=False, width = img_x, height = img_y )
#scene = Scene(use_gui=True, width = img_x, height = img_y,num_frames=16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_weights', type=str)
    parser.add_argument('--model', default=None, type=str)
    parser.add_argument('--model_type',choices=['single','dino','dino_concat','concat','swinvideo','clip'], type=str)
    args = parser.parse_args()
    return args


def test(env, count = 100):
    args = parse_args()
    policy = TransPolicy(args.model_type, args.pretrained_weights, args.model)

    episode_count = 0
    total_reward = 0.
    while episode_count<count:
        episode_count += 1
        print(f"Episode: {episode_count} ... ")

        #scene.random_reset(ball_count = 3)
        obs = env.reset()
        
#        objects = obs["cameras"][0]
#        print(objects.size)
#        print(objects.shape)
        
        start_time = time.time()
        (dx, dy, drx) = (step_size,step_size,step_size)
        step_count = 0
        while step_count < max_traj_length:
            
            #if args.model_type == 'single' or args.model_type == 'dino' or args.model_type == 'clip' :
            rgb = obs["cameras"][0]
        #    print(rgb)
            print(rgb.shape)

            action = policy.get_action(rgb, rgb, 0) # 0 for rgb
            print(action)

            #obs, reward, done, info = env.step([action[0], action[1], action[2]*3]) # order: x, y, r
            obs, reward, done, info = env.step([action[1], action[2], action[0]]) # order: x, y, r
            step_count = step_count + 1
            
        env.finalize()
        #total_reward += env.get_ball_counts()[1]
        
        #print(f"Reward: {scene.get_ball_counts()[1]}")
        #print(f"Steps: {scene.get_frame_count()}")
        #print(f"FPS: {scene.get_frame_count() / (time.time() - start_time)}")

    #print(f"Avg reward {total_reward/episode_count} for {episode_count} episodes")
    return

if __name__ == '__main__':
    env = PourEnv()
    test(env)
