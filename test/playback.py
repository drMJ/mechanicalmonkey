from mmky import writers
import os
import cv2
import argparse
import numpy as np
import h5py
import pickle

def play_hdf5(episode):
    cams = list(episode['/observation/cameras/'].keys())
    length = len(episode['observation/cameras'][cams[0]]['color'])
    print(f"Episode length: {length}")

    for frame_ix in range(length):
        rgb = np.concatenate(list(np.array(cam["color"][frame_ix]) for cam in episode['/observation/cameras'].values()), axis = 1)
        depth = np.concatenate(list(np.array(cam["depth"][frame_ix]) for cam in episode['/observation/cameras'].values()), axis = 1).astype(np.uint8)
        cv2.imshow('rgb observation', rgb)
        cv2.imshow('depth observation', depth)
        cv2.waitKey(1)

def play_pickle(episode):
    length = len(episode)
    for frame_ix in range(length):
        rgb = np.concatenate(list(np.array(cam["color"]) for cam in episode[frame_ix]["observation"]["cameras"].values()), axis = 1)
        depth = np.concatenate(list(np.array(cam["depth"]) for cam in episode[frame_ix]["observation"]["cameras"].values()), axis = 1).astype(np.uint8)
        cv2.imshow('rgb observation', rgb)
        cv2.imshow('depth observation', depth)
        cv2.waitKey(1)
    print(f"Success: {episde[length-1]["info"]}")


parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Tool to inspect MechanicalMonkey recordings.")
parser.add_argument(
    "-f",
    "--file",
    type=str,
    help="The file to play back.")

args = parser.parse_args()
ext = os.path.splitext(args.file)[-1]
if ext == '.hdf5':
    episode =h5py.File(args.file, 'r')
    play_hdf5(episode)
elif ext == '.pickle':
    with open(args.file, "rb") as f: 
        episode = pickle.load(f)
        play_pickle(episode)
else:
    raise Exception(f"Unsuported file type {ext}. Supported files are hdf5 and npy")
