"""Native OctoMap occupancy with the existing bounded viewer/snapshot contract.

The OcTree is expressed in grid-local metres. The exported placement transform
maps it back into the scene coordinate frame. RGB-D filtering and conservative
ray clipping are shared with the NumPy backend, before evidence enters OctoMap.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random

import numpy as np

from safety_module.occupancy import LocalOccupancyGrid


def require_octomap():
    """Load only when selected, and fail before opening any hardware."""
    try:
        import octomap
    except ImportError as exc:
        raise RuntimeError(
            "OctoMap backend requires octomap-python; install in the runtime environment: "
            "python -m pip install 'octomap-python==1.10.0.0'"
        ) from exc
    return octomap


class OctomapOccupancyGrid(LocalOccupancyGrid):
    """Accumulate native log-odds; fresh accepted hits become occupied immediately.

    Unlike the default current-frame map, historical occupied cells persist
    until raw free-space rays reduce their probability below 0.5. Completion
    contributes weaker hits only. Uncertain depth never clears history.
    Dense arrays exist for compatibility with bounded snapshots/manual selection;
    the native octree is the authority for probabilities and occupied cells.
    """

    coordinate_frame = "left_base"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._octomap = require_octomap()
        self.tree = self._octomap.OcTree(self.resolution_m)
        # OcTree keys have a finite range. Our positive local domain must fit.
        limit = (2 ** (self.tree.getTreeDepth() - 1)) * self.resolution_m
        if np.any(self.shape * self.resolution_m >= limit):
            raise ValueError("Workspace exceeds the OctoMap key range")
        self.tree.setProbHit(float(1 / (1 + np.exp(-self.hit_log))))
        self.tree.setProbMiss(float(1 / (1 + np.exp(-self.miss_log))))
        self.tree.setOccupancyThres(.5)
        self.tree.setClampingThresMin(float(1 / (1 + np.exp(2.))))
        self.tree.setClampingThresMax(float(1 / (1 + np.exp(-3.5))))
        self.known = np.zeros(self.size, bool)

    def _local_centers(self, ids):
        cells = np.column_stack(np.unravel_index(np.asarray(ids, np.int64), tuple(self.shape)))
        return np.ascontiguousarray((cells + .5) * self.resolution_m, dtype=np.float64)

    def _update_nodes(self, ids, increment, *, immediate_hit=False):
        for voxel, point in zip(ids, self._local_centers(ids)):
            delta = float(increment)
            if immediate_hit and self.known[voxel] and self.log_odds[voxel] < 0:
                # updateNode expands a pruned leaf before changing this voxel.
                # Mutating search(point) directly would reset all its siblings.
                delta -= float(self.log_odds[voxel])
            # Native recursive updates also prune eight equal-probability
            # children into a parent, preserving probabilities and unknown gaps.
            node = self.tree.updateNode(point, delta, lazy_eval=False)
            self.log_odds[voxel] = node.getLogOdds()
            self.known[voxel] = True

    def update(self, *, timestamp_ns, raw_hits, free_ids, completion_hits=(), uncertain_hits=()):
        if not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns <= self.timestamp_ns:
            raise ValueError("Occupancy frames require strictly increasing capture timestamps")
        hit, completed, uncertain, occupied, free, conflicts = self._frame_evidence(
            raw_hits, free_ids, completion_hits, uncertain_hits)
        self._update_nodes(free, self.miss_log)
        self._update_nodes(np.setdiff1d(completed, hit), .405465, immediate_hit=True)
        self._update_nodes(hit, self.hit_log, immediate_hit=True)
        self.tree.updateInnerOccupancy()
        seen = np.union1d(np.union1d(occupied, uncertain), free)
        self.last_observed_ns[seen] = timestamp_ns
        self.sources[free], self.sources[completed], self.sources[hit], self.sources[uncertain] = 2, 4, 1, 8
        self.current_hits = np.flatnonzero(self.active & self.known & (self.log_odds >= self.tree.getOccupancyThresLog()))
        self.timestamp_ns, self.frame = int(timestamp_ns), self.frame + 1
        self.diagnostics = {
            "raw_hit_voxels": len(hit), "completion_hit_voxels": len(completed),
            "uncertain_raw_voxels": len(uncertain), "raw_free_voxels": len(free),
            "conflicting_voxels": len(conflicts),
            "octomap_nodes": int(self.tree.size()), "octomap_leaf_nodes": int(self.tree.getNumLeafNodes()),
            "octomap_memory_bytes": int(self.tree.memoryUsage()),
            "octomap_unknown_voxels": int((self.active & ~self.known).sum()),
            "historical_occupied_voxels": int(len(np.setdiff1d(self.current_hits, occupied))),
        }

    def placement(self, coordinate_frame):
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = self.axes, self.origin
        return {"coordinate_frame": coordinate_frame, "octomap_frame": "occupancy_grid_local",
                "parent_T_octomap": transform.tolist(), "resolution_m": self.resolution_m,
                "shape": self.shape.tolist(), "capture_timestamp_ns": self.timestamp_ns}

    def viewer_payload(self, *, max_leaves=6000):
        """Read real native leaves, including pruned sizes; never voxelize them.

        Deterministic per-state reservoir sampling bounds geometry/JSON size.
        Counts and histograms describe the complete tree, even when sampled.
        No native node wrappers survive the iteration or enter HTTP threads.
        """
        if not isinstance(max_leaves, int) or max_leaves < 2:
            raise ValueError("OctoMap viewer requires a budget of at least two leaves")
        limits = {"occupied": max_leaves - max_leaves // 3, "free": max_leaves // 3}
        # A budget of two still reserves one node for each state.
        limits["free"] = max(1, limits["free"])
        limits["occupied"] = max_leaves - limits["free"]
        groups = {name: {"count": 0, "depth_counts": {}, "samples": []} for name in limits}
        rng = random.Random(0)
        threshold = self.tree.getOccupancyThres()
        for leaf in self.tree.begin_leafs():
            probability = float(leaf.getOccupancy())
            name = "occupied" if probability >= threshold else "free"
            group = groups[name]
            group["count"] += 1
            depth = int(leaf.getDepth())
            group["depth_counts"][str(depth)] = group["depth_counts"].get(str(depth), 0) + 1
            index = group["count"] - 1
            if index >= limits[name]:
                index = rng.randrange(group["count"])
                if index >= limits[name]:
                    continue
            local_center = np.asarray(leaf.getCoordinate(), dtype=float)
            center = self.origin + local_center @ self.axes.T
            sample = (center.round(7).tolist(), float(leaf.getSize()), probability, depth)
            if len(group["samples"]) < limits[name]:
                group["samples"].append(sample)
            else:
                group["samples"][index] = sample
        result = {"schema": "octomap_native_leaves_v1", "frame": self.frame,
                  "capture_timestamp_ns": self.timestamp_ns, "coordinate_frame": self.coordinate_frame,
                  "axes": self.axes.tolist(), "resolution_m": self.resolution_m,
                  "max_depth": int(self.tree.getTreeDepth()), "max_leaves": max_leaves}
        for name, group in groups.items():
            samples = group.pop("samples")
            result[name] = {**group, "shown": len(samples), "sampled": len(samples) < group["count"],
                            "centers": [item[0] for item in samples],
                            "sizes_m": [item[1] for item in samples],
                            "probabilities": [item[2] for item in samples],
                            "depths": [item[3] for item in samples]}
        return result

    def snapshot(self):
        # Preserve the v1 consumer policy: unmapped active cells are exported
        # as nonblocking. Native unknown counts remain explicit in metadata.
        result = super().snapshot()
        result.update({"backend": "octomap", "temporal_policy": "accumulated_raw_free_clearing",
                       "octomap": {**self.placement(self.coordinate_frame), **self.diagnostics}})
        return result

    def export(self, path, *, coordinate_frame="left_base"):
        """Save .ot probabilities or a .bt copy, plus the scene placement JSON.

        writeBinary mutates its tree into a maximum-likelihood map. Serialize
        a clone to avoid changing probabilities in the running detector.
        """
        path = Path(path).expanduser()
        if path.suffix.lower() not in (".bt", ".ot"):
            raise ValueError("OctoMap output must end in .bt or .ot")
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = self.tree.write()
        if not isinstance(blob, bytes):
            raise RuntimeError("OctoMap serialization failed")
        if path.suffix.lower() == ".bt":
            clone = self._octomap.OcTree.read(blob)
            blob = clone.writeBinary()
            if not isinstance(blob, bytes):
                raise RuntimeError("OctoMap binary serialization failed")
        path.write_bytes(blob)
        metadata = {**self.placement(coordinate_frame), "format": path.suffix.lower()[1:],
                    "backend": "octomap", "frame": self.frame}
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return os.fspath(path)
