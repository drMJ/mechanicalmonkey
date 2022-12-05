import h5py
import numpy as np
import os
import time
import threading
import pickle


class Writer:
    def __init__(self, file_name_prefix="", file_name_ext="", file_path="trajectories"):
        if not os.path.isabs(file_path):
            file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.path = file_path
        self.file_name_prefix = file_name_prefix
        self.file_name_ext = file_name_ext
        self._writer_threads = {}
        self.episode_count = 0

    def close(self):
        i = 0
        for n, t in self._writer_threads.items():
            print(f"Flushing {i} / {len(self._writer_threads)}: {n}")
            t.join()
            i += 1

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
        # flush to file
        f = self.get_file_name()
        t = threading.Thread(target=self.thread_fn, args=(f, self.data))
        t.start()
        self._writer_threads[f] = t
        self.episode_count += 1

class PickleWriter(Writer):
    def __init__(self, file_name_ext="pickle", **kwargs):
        super().__init__(**kwargs)
        self.thread_fn = PickleWriter.__write_file

    def __write_file(filename, data):
        # flush to file
        with open(filename, "wb") as f: 
            pickle.dump(data, f) 
        print("trajectory saved.")

class HDF5Writer(Writer):
    def __init__(self, file_name_ext="hdf5", **kwargs):
        super().__init__(**kwargs)
        self.thread_fn = HDF5Writer.__write_file

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

    def __write_file(filename, data):
        # flush to file
        with h5py.File(filename, 'w') as file:
            for i in range(len(data)):
                HDF5Writer.__append(file, data[i], i, len(data))
        print("trajectory saved.")


if __name__ == '__main__':
    writer = HDF5Writer("test")
    obs = {
            'cameras':np.zeros((100, 100)), 
            'world':{"0":"zero"}, 
            'arm':[0]*78, 
            'action_completed': True,
            'time': time.perf_counter()
        }

    writer.start_episode(obs)
    writer.log({"cmd":"pick", "args":{"p1":0, "p1":1}}, obs, {})
    writer.end_episode()