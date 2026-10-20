"""A frozen selection of world-space voxel cells, shared by replay and display."""

import threading

import numpy as np

from safety_module.occupancy import OUTSIDE


class VoxelSelectionScope:
    def __init__(self, reference, voxel_ids):
        self.reference = reference
        self.ids = np.asarray(voxel_ids, dtype=np.int64)
        self.centers = reference.centers(self.ids)
        self.selected = np.zeros(reference.size, bool)
        self.selected[self.ids] = True
        self._key = None
        self._mask = None
        self._lock = threading.Lock()

    @staticmethod
    def geometry_key(grid):
        return (tuple(grid.origin), tuple(grid.axes.reshape(-1)),
                tuple(grid.shape), grid.resolution_m)

    def mask_for(self, live):
        """Use exact IDs on the same grid; otherwise test live centers in saved cells."""
        mapped, inside = live.indices(self.centers)
        if not np.all(inside) or np.any(live.state[mapped] == OUTSIDE):
            raise ValueError("Saved manual obstacle falls outside the live occupancy workspace; recapture and reselect")
        key = self.geometry_key(live)
        with self._lock:
            if key != self._key:
                if key == self.geometry_key(self.reference):
                    mask = self.selected.copy()
                else:
                    ids, inside = self.reference.indices(live.centers(np.arange(live.size)))
                    mask = np.zeros(live.size, bool)
                    mask[inside] = self.selected[ids[inside]]
                if len(self.ids) and not np.any(mask & (live.state != OUTSIDE)):
                    raise ValueError("Saved manual voxel selection has no live cells; recapture and reselect")
                mask.setflags(write=False)
                self._key, self._mask = key, mask
            return self._mask
