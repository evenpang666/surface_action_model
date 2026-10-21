"""Memory-mapped array-directory shards; NPZ stays readable for migration."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np


class ArrayDirectory:
    def __init__(self, path):
        self.path = Path(path)
        self.files = sorted(p.stem for p in self.path.glob("*.npy"))

    def __contains__(self, name):
        return name in self.files

    def __getitem__(self, name):
        return np.load(self.path / f"{name}.npy", mmap_mode="r", allow_pickle=False)


@contextmanager
def open_shard(path):
    if Path(path).is_dir():
        yield ArrayDirectory(path)
    else:
        with np.load(path, allow_pickle=False) as data:
            yield data


class SampleArray:
    """Map frame/point selections at access time, not for a whole episode."""
    def __init__(self, source, *, frames=None, points=None, point_axis=1):
        self.source, self.frames, self.points, self.point_axis = source, frames, points, point_axis
        shape = list(source.shape)
        if frames is not None: shape[0] = len(frames)
        if points is not None: shape[point_axis] = len(points)
        self.shape = tuple(shape)

    def __getitem__(self, key):
        first, *rest = key if isinstance(key, tuple) else (key,)
        frame = first if self.frames is None else self.frames[first]
        value = self.source[frame]
        if self.points is not None:
            value = np.take(value, self.points, axis=self.point_axis - 1)
        return value[tuple(rest)] if rest else value
