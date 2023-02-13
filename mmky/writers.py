import h5py
import numpy as np
import os
import time
import threading
import pickle
import cv2

class Writer:
    def __init__(self, file_name_prefix="", file_name_ext="", file_path="trajectories"):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.path = file_path
        self.file_name_prefix = file_name_prefix
        self.file_name_ext = file_name_ext
        self.episode_count = 0
        self.writer_thread = None

    def close(self):
        if self.writer_thread:
            self.writer_thread.join()
            self.writer_thread = None

    def get_file_name(self):
        return os.path.join(self.path, f"{self.file_name_prefix}_{self.episode_count}_{time.time()}.{self.file_name_ext}")

    def start_episode(self, observation):
        self.step = 0
        self.data = [{"observation":observation, 'action':{}, 'info':{}}]
         
    def log(self, action, observation, info):
        self.data[self.step]['action'] = action
        self.data[self.step]['info'] = info
        self.data +=  [{"observation":observation, 'action':{}, 'info':{}}]
        self.step += 1

    def end_episode(self, discard=False):
        # wait for previous flush to finish
        if self.writer_thread:
            self.writer_thread.join()

        # flush to file
        f = self.get_file_name()
        self.writer_thread = threading.Thread(target=self.thread_fn, args=(f, self.data))
        self.writer_thread.start()
        self.episode_count += 1

class PickleWriter(Writer):
    def __init__(self, file_name_ext="pickle", **kwargs):
        super().__init__(file_name_ext="pickle", **kwargs)
        self.thread_fn = PickleWriter._write_file

    def _write_file(filename, data):
        # flush to file
        with open(filename, "wb") as f: 
            pickle.dump(data, f) 
        print("trajectory saved.")

class HDF5Writer(Writer):
    def __init__(self, **kwargs):
        super().__init__(file_name_ext="hdf5", **kwargs)
        self.thread_fn = HDF5Writer._write_file

    def __append(dest, src, ix, size):
        assert isinstance(src, dict)
        for k,v in src.items():
            if not isinstance(k, str):
                k = str(k)
            if isinstance(v, dict):
                dest.require_group(k) 
                HDF5Writer.__append(dest[k], v, ix, size)
            else:
                if k not in dest:
                    if isinstance(v, np.ndarray):
                        dest.create_dataset(k, (size,)+np.shape(v), dtype=v.dtype, maxshape=(None,)+np.shape(v), compression="gzip")
                    elif np.ndim(v) > 0:
                        dt = np.dtype(type(np.take(v,0))) 
                        dest.create_dataset(k, (size,)+np.shape(v), dtype=dt, maxshape=(None,)+np.shape(v), compression="gzip")
                    else:
                        dt = h5py.string_dtype() if isinstance(v, str) else np.dtype(type(v))
                        dest.create_dataset(k, (size,), dtype=dt, maxshape=(None,), compression="gzip")
                while ix >= len(dest[k]):
                    dest[k].resize(len(dest[k]) + size, 0)
                dest[k][ix] = v

    def _write_file(filename, data):
        # flush to file
        with h5py.File(filename, 'w') as file:
            for i in range(len(data)):
                HDF5Writer.__append(file, data[i], i, len(data))
        print(f"{filename} saved.")


def convert_dir(dir):
    while True:    
        dir_entries = os.listdir(dir)
        for entry in dir_entries:
            entry = os.path.join(dir, entry)
            if not os.path.isfile(entry):
                continue

            try:
                with open(entry, "rb") as f: 
                    data = pickle.load(f) 
            except:
                continue
            print(f"Processing {entry}")
            filename, _ = os.path.splitext(entry) 
            filename += ".hdf5"
            writer = HDF5Writer._write_file(filename, data)
            os.remove(entry)

def fix_dir(dir):
    import roman
    import cv2
    print(f"processing {dir}")
    dir_entries = os.listdir(dir)
    for entry in dir_entries:
        entry = os.path.join(dir, entry)
        if not os.path.isfile(entry):
            continue

        filename, ext = os.path.splitext(entry) 
        try:
            data = h5py.File(entry, 'r+')
        except Exception as e:
            print(f"Skipping {entry}: {e}")
            continue
        
        print(f"Converting {entry}")

        # convert to rgb
        for cam_name in ['left_cam', 'overhead_cam', 'right_cam']:
            print(cam_name) 
            cam = data['observation']['cameras'][cam_name]['color']
            for i in range(len(cam)):   
                cam[i, :, :, :] = cv2.cvtColor(cam[i], code=cv2.COLOR_BGR2RGB)

        # add metadata
        data["meta/task"] = ['pick']
        data["meta/arm"] = ['Universal Robots UR5e']
        data["meta/force_torque_sensor"] = ['builtin']
        data["meta/gripper"] = ['Robotiq 3-Finger Gripper']
        data["meta/cameras"] = ['AzureKinect', 'AzureKinect', 'RealSense D405', 'AzureKinect']
        data["meta/fps"] = [30]
        data["meta/resolution"] = [256, 256]
        data["meta/color_format"] = ['RGB']
        data["meta/color_range"] = [0, 255]
        data["meta/depth_unit"] = ['mm']
        data["meta/tool_pose_format"] = ['x, y, z, rx, ry, rz']
        data["meta/tool_pose_angle_format"] = ['axis angle']
        data["meta/tool_pose_abs_limit"] = [1, 1, 1, 2*np.pi, 2*np.pi, 2*np.pi]
        data["meta/joint_positions_format"] = ['[base shoulder elbow wrist1 wrist2 wrist3]']
        data["meta/joint_positions_abs_limit"] = [2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi]
        data["meta/hand_position_range"] = [0, 255]
        data["meta/hand_position_format"] = ['0 => fully opened, 255 => fully closed']
        data["meta/possible_actions"] = ["stop", "move", "touch", "open", "release", "grasp", "pinch"]

        data.flush()

if __name__ == '__main__':
    import sys
    #fix_dir(sys.argv[1])
    convert_dir(sys.argv[1])
