"""Native Voxblox TSDF/ESDF and bounded surface voxels, without ROS."""
from __future__ import annotations

import json
import importlib
import time
import threading
from pathlib import Path

import numpy as np

from real_scripts.camera_geometry import pixel_rays
from safety_module.occupancy import GridGeometry, LocalOccupancyGrid, _pack


def require_voxblox():
    try:
        _voxblox_native = importlib.import_module("real_scripts._voxblox_native")
    except ImportError as exc:
        raise RuntimeError("Voxblox native binding is missing; build with the runtime Python: "
                           "python scripts/build_voxblox.py (see real_scripts/README.md)") from exc
    if getattr(_voxblox_native, "api_version", 0) < 2:
        raise RuntimeError("Voxblox native binding is outdated; rebuild with the runtime Python: "
                           "python scripts/build_voxblox.py")
    return _voxblox_native


class VoxbloxOccupancyGrid(LocalOccupancyGrid):
    """Native TSDF solid/near-surface cells plus accepted endpoint coverage.

    Historical surface cells persist until measured rays update the TSDF.
    Completion is current-frame hit-only evidence and never generates TSDF rays.
    Uncertain endpoint cells retain their pre-frame native distance/weight.
    """
    coordinate_frame = "left_base"

    def __init__(self, *args, truncation_m=None, surface_band_m=None, min_weight=.1,
                 max_weight=32., threads=1, max_blocks=2048, esdf=True,
                 esdf_max_distance_m=2., esdf_min_distance_m=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.truncation_m = float(4*self.resolution_m if truncation_m is None else truncation_m)
        self.surface_band_m = float(self.resolution_m if surface_band_m is None else surface_band_m)
        self.esdf_enabled = bool(esdf)
        self.esdf_max_distance_m = float(esdf_max_distance_m)
        self.esdf_min_distance_m = float(self.resolution_m if esdf_min_distance_m is None else esdf_min_distance_m)
        if (not np.isfinite([self.truncation_m, self.surface_band_m, min_weight, max_weight]).all()
                or not self.resolution_m <= self.truncation_m < 10.
                or not 0 < self.surface_band_m <= self.truncation_m
                or not 0 < min_weight <= max_weight or max_weight < 1
                or not 1 <= threads <= 32 or max_blocks < 1):
            raise ValueError("Invalid Voxblox TSDF settings")
        if (not np.isfinite([self.esdf_max_distance_m, self.esdf_min_distance_m]).all()
                or not 0 < self.esdf_min_distance_m <= self.truncation_m
                or self.esdf_max_distance_m <= max(self.esdf_min_distance_m, self.truncation_m)):
            raise ValueError("Invalid Voxblox ESDF settings")
        self.min_weight, self.max_weight, self.max_blocks = float(min_weight), float(max_weight), int(max_blocks)
        self.native = require_voxblox()
        self.volume = self.native.TsdfVolume(self.resolution_m, 16, self.truncation_m, max_weight, 10., threads)
        self.distance_m = np.full(self.size, np.nan, np.float32)
        self.weight = np.zeros(self.size, np.float32)
        self.esdf_distance_m = np.full(self.size, np.nan, np.float32)
        self.esdf_observed = np.zeros(self.size, bool)
        if self.esdf_enabled:
            self.volume.configure_esdf(self.esdf_max_distance_m, self.esdf_min_distance_m, self.min_weight)

    def _local_centers(self, ids):
        cells = np.column_stack(np.unravel_index(np.asarray(ids, np.int64), tuple(self.shape)))
        return np.ascontiguousarray((cells + .5)*self.resolution_m)

    def update(self, **_kwargs):
        raise ValueError("Voxblox requires calibrated RGB-D views; use integrate_depth_views")

    def integrate_rgbd_views(self, views, *, timestamp_ns, raw_hits, free_ids, completion_hits=(),
                            uncertain_hits=(), ray_stride=6, free_margin_m=.015, max_depth_m=2.5):
        if not isinstance(timestamp_ns, (int, np.integer)) or timestamp_ns <= self.timestamp_ns:
            raise ValueError("Occupancy frames require strictly increasing capture timestamps")
        hit, completed, uncertain, occupied, free, conflicts = self._frame_evidence(
            raw_hits, free_ids, completion_hits, uncertain_hits)
        tsdf_start = time.perf_counter()
        protected_points = self._local_centers(uncertain)
        protected = self.volume.query(protected_points)
        ray_count = 0
        for camera, depth, keep, rendered in views:
            depth = np.asarray(depth)[::ray_stride, ::ray_stride]
            keep = np.asarray(keep, bool)[::ray_stride, ::ray_stride]
            rendered = np.asarray(rendered)[::ray_stride, ::ray_stride]
            valid = keep & np.isfinite(depth) & (depth > .01) & (depth <= max_depth_m)
            robot = np.isfinite(rendered) & (rendered > 0)
            # Do not invent an endpoint on a robot mask or on the max-range cap.
            valid &= ~robot | (depth < rendered-free_margin_m)
            v, u = np.nonzero(valid)
            if not len(u):
                continue
            rays = pixel_rays(np.c_[u*ray_stride, v*ray_stride], camera.intrinsics,
                              camera.distortion, camera.distortion_model)
            world = (rays * depth[v, u, None]) @ camera.camera_to_world[:3,:3].T + camera.camera_to_world[:3,3]
            local = (world-self.origin) @ self.axes
            origin = (camera.camera_to_world[:3,3]-self.origin) @ self.axes
            self.volume.integrate(np.ascontiguousarray(origin), np.ascontiguousarray(local))
            ray_count += len(local)
        self.volume.restore(protected_points, protected)
        self.volume.crop(np.zeros(3), self.shape*self.resolution_m, self.max_blocks)
        esdf_start = time.perf_counter()
        tsdf_seconds = esdf_start - tsdf_start
        self.esdf_distance_m.fill(np.nan); self.esdf_observed.fill(False)
        if self.esdf_enabled:
            self.volume.update_esdf()
            esdf = self.volume.esdf_observed()
            cells = np.floor(esdf[:,:3]/self.resolution_m + 1e-7).astype(np.int64)
            inside = np.all((cells>=0) & (cells<self.shape), axis=1)
            esdf_ids = cells[inside] @ self.strides
            allowed = self.active[esdf_ids]
            esdf_ids = esdf_ids[allowed]
            self.esdf_distance_m[esdf_ids] = np.clip(esdf[inside,3][allowed],
                                                    -self.esdf_max_distance_m, self.esdf_max_distance_m)
            self.esdf_observed[esdf_ids] = True
        esdf_seconds = time.perf_counter() - esdf_start
        extract_start = time.perf_counter()
        observed = self.volume.observed()
        cells = np.floor(observed[:,:3]/self.resolution_m + 1e-7).astype(np.int64)
        inside = np.all((cells>=0) & (cells<self.shape), axis=1)
        ids = cells[inside] @ self.strides
        values = observed[inside,3:]
        allowed = self.active[ids]
        ids, values = ids[allowed], values[allowed]
        self.distance_m.fill(np.nan); self.weight.fill(0)
        self.distance_m[ids], self.weight[ids] = values[:,0], values[:,1]
        band = (self.weight >= self.min_weight) & (np.abs(self.distance_m) <= self.surface_band_m)
        solid = (self.weight >= self.min_weight) & (self.distance_m <= self.surface_band_m)
        self.current_hits = np.union1d(np.flatnonzero(self.active & solid), occupied)
        # v1 wire compatibility only: these are not TSDF-derived probabilities.
        self.log_odds.fill(0.)
        self.log_odds[ids] = self.miss_log
        self.log_odds[self.current_hits] = self.hit_log
        self.sources[ids] = 16
        self.sources[free], self.sources[completed], self.sources[hit], self.sources[uncertain] = 2,4,1,8
        seen = np.union1d(np.union1d(occupied, uncertain), free)
        self.last_observed_ns[seen] = timestamp_ns
        first_seen = ids[self.last_observed_ns[ids]==0]
        self.last_observed_ns[first_seen] = timestamp_ns
        self.timestamp_ns, self.frame = int(timestamp_ns), self.frame+1
        self.diagnostics = {"raw_hit_voxels":len(hit), "completion_hit_voxels":len(completed),
                            "uncertain_raw_voxels":len(uncertain), "raw_free_voxels":len(free),
                            "conflicting_voxels":len(conflicts), "voxblox_integrated_rays":ray_count,
                            "voxblox_blocks":int(self.volume.blocks), "tsdf_observed_voxels":len(ids),
                            "tsdf_surface_voxels":int((self.active & band).sum()),
                            "tsdf_interior_voxels":int(np.count_nonzero(
                                self.active & solid & (self.distance_m < -self.surface_band_m))),
                            "esdf_observed_voxels":int(self.esdf_observed.sum()),
                            "esdf_saturated_voxels":int(np.count_nonzero(
                                self.esdf_observed & (np.abs(self.esdf_distance_m)>=self.esdf_max_distance_m-1e-6))),
                            "esdf_update_seconds":esdf_seconds,
                            "tsdf_integration_seconds":tsdf_seconds,
                            "tsdf_extract_seconds":time.perf_counter()-extract_start,
                            "current_endpoint_only_voxels":int(np.count_nonzero(~solid[occupied]))}

    def query_esdf(self, points, *, interpolate=True):
        """Query world-frame points; metres and world-frame distance gradients.

        Interpolation requires all eight observed, active voxel centers. Unknown
        or out-of-domain queries return NaN and observed=False. If any stencil
        value is capped, saturated=True and the gradient is invalid. Capped
        estimates are not exact clearance.
        With interpolate=False return the containing voxel and no gradient.
        """
        if not self.esdf_enabled:
            raise ValueError("Voxblox ESDF is disabled")
        from safety_module.distance_field import query_esdf_grid
        return query_esdf_grid(self, self.esdf_distance_m, self.esdf_observed,
                               self.active, self.esdf_max_distance_m, points, interpolate=interpolate)

    def freeze_view(self, *, max_voxels=6000):
        """Capture arrays now; sample/convert only on a viewer HTTP request."""
        return FrozenVoxbloxView(self, max_voxels)

    def viewer_payload(self, *, max_voxels=6000):
        if not isinstance(max_voxels, int) or max_voxels < 2:
            raise ValueError("Voxblox viewer requires a budget of at least two voxels")
        known = self.active & (self.weight>=self.min_weight)
        surface = known & (np.abs(self.distance_m)<=self.surface_band_m)
        result = {"schema":"voxblox_tsdf_view_v1", "frame":self.frame,
                  "capture_timestamp_ns":self.timestamp_ns, "coordinate_frame":self.coordinate_frame,
                  "axes":self.axes.tolist(), "resolution_m":self.resolution_m,
                  "truncation_m":self.truncation_m, "surface_band_m":self.surface_band_m,
                  "max_weight":self.max_weight, "esdf_enabled":self.esdf_enabled,
                  "esdf_max_distance_m":self.esdf_max_distance_m}
        field_limit = max(1, max_voxels//3)
        def group(mask, limit, distances, weights):
            ids = np.flatnonzero(mask)
            count = len(ids)
            if count>limit:
                ids = ids[np.linspace(0,count-1,limit,dtype=int)]
            return {"count":count,"shown":len(ids),"sampled":len(ids)<count,
                            "centers":self.centers(ids).round(7).tolist(),
                            "sizes_m":[self.resolution_m]*len(ids),
                            "distances_m":distances[ids].astype(float).tolist(),
                            "weights":weights[ids].astype(float).tolist()}
        result["surface"] = group(surface,max_voxels-field_limit,self.distance_m,self.weight)
        result["field"] = group(known & ~surface,field_limit,self.distance_m,self.weight)
        if self.esdf_enabled:
            esdf_surface = self.esdf_observed & (np.abs(self.esdf_distance_m)<=self.surface_band_m)
            result["esdf"] = {
                "surface":group(esdf_surface,max_voxels-field_limit,self.esdf_distance_m,self.weight),
                "field":group(self.esdf_observed & ~esdf_surface,field_limit,self.esdf_distance_m,self.weight)}
        return result

    def snapshot(self):
        result = super().snapshot()
        result.update(backend="voxblox", field_kind="TSDF", log_odds_semantics="compatibility_state_only",
                      occupied_policy="observed_tsdf_le_surface_band_plus_current_endpoints",
                      available_fields=["TSDF","ESDF"] if self.esdf_enabled else ["TSDF"],
                      temporal_policy="native_tsdf_fusion_with_current_endpoint_coverage",
                      voxblox={"truncation_m":self.truncation_m, "surface_band_m":self.surface_band_m,
                               "min_weight":self.min_weight,"max_weight":self.max_weight,
                               "distance_dtype":"float32-le", "weight_dtype":"float32-le",
                               "distance_m_data":_pack(self.distance_m.astype('<f4')),
                               "weight_data":_pack(self.weight.astype('<f4')),
                               "upstream_revision":self.native.upstream_revision})
        result["voxblox"]["upstream_patch"] = self.native.upstream_patch
        result["voxblox"]["esdf"] = self._esdf_metadata()
        result["voxblox"]["esdf"].update(
            distance_dtype="float32-le", observed_dtype="uint8",
            distance_m_data=_pack(self.esdf_distance_m.astype('<f4')),
            observed_data=_pack(self.esdf_observed.astype('u1')))
        result["source_codes"]["native_tsdf"] = 16
        return result

    def _esdf_metadata(self):
        return {"enabled":self.esdf_enabled, "full_euclidean_distance":True,
                "max_distance_m":self.esdf_max_distance_m, "min_distance_m":self.esdf_min_distance_m,
                "update_policy":"batch_after_complete_frame", "unknown_policy":"unobserved_nan",
                "saturation_policy":"signed_max_distance", "add_occupied_crust":False,
                "robot_clearance_sphere":False}

    def export(self, path, *, coordinate_frame=None):
        path = Path(path).expanduser()
        if path.suffix.lower()!=".npz":
            raise ValueError("Voxblox distance-field export requires .npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"schema":"voxblox_workspace_tsdf_v1", "coordinate_frame":coordinate_frame or self.coordinate_frame,
                    "capture_timestamp_ns":self.timestamp_ns,"frame":self.frame,
                    "truncation_m":self.truncation_m,"surface_band_m":self.surface_band_m,
                    "min_weight":self.min_weight, "upstream_revision":self.native.upstream_revision}
        metadata["upstream_patch"] = self.native.upstream_patch
        metadata["esdf"] = self._esdf_metadata()
        np.savez_compressed(path, distance_m=self.distance_m.reshape(tuple(self.shape)),
                            esdf_distance_m=self.esdf_distance_m.reshape(tuple(self.shape)),
                            esdf_observed=self.esdf_observed.reshape(tuple(self.shape)),
                            weight=self.weight.reshape(tuple(self.shape)), active=self.active.reshape(tuple(self.shape)),
                            state=self.states().reshape(tuple(self.shape)), origin=self.origin, axes=self.axes,
                            resolution_m=self.resolution_m, metadata=json.dumps(metadata))


class FrozenVoxbloxView(GridGeometry):
    """Frame-aligned display data with lazy, thread-safe materialization."""

    def __init__(self, grid, max_voxels):
        super().__init__(grid.origin.copy(), grid.axes.copy(), grid.shape.copy(), grid.resolution_m)
        for name in ("active", "distance_m", "weight", "esdf_distance_m", "esdf_observed"):
            value = getattr(grid, name).copy()
            value.setflags(write=False)
            setattr(self, name, value)
        for name in ("frame", "timestamp_ns", "coordinate_frame", "min_weight", "surface_band_m",
                     "truncation_m", "max_weight", "esdf_enabled", "esdf_max_distance_m"):
            setattr(self, name, getattr(grid, name))
        self.max_voxels = max_voxels
        self._payload = None
        self._lock = threading.Lock()

    def materialize(self):
        with self._lock:
            if self._payload is None:
                self._payload = VoxbloxOccupancyGrid.viewer_payload(self, max_voxels=self.max_voxels)
            return self._payload
