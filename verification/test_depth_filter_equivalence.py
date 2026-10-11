"""Keep accelerated depth dilation equivalent to the previous pixel rules."""

import numpy as np
import pytest

from real_scripts.real_robot_adapter import robot_depth_keep_mask
from real_scripts.ur7e_collision_mesh import render_surface_points_depth


def _old_min_dilation(source: np.ndarray, radius: int) -> np.ndarray:
    result = source.copy()
    for _ in range(radius):
        padded = np.pad(result, 1, mode="constant", constant_values=np.inf)
        result = np.minimum.reduce([
            padded[y:y + result.shape[0], x:x + result.shape[1]]
            for y in range(3) for x in range(3)
        ])
    return result


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_rendered_depth_splat_matches_minimum_dilation(radius: int) -> None:
    rng = np.random.default_rng(19)
    width, height = 43, 29
    points = np.column_stack((
        rng.integers(0, width, 120),
        rng.integers(0, height, 120),
        rng.uniform(0.3, 2.0, 120),
    ))
    intrinsics = np.eye(3)
    single_pixels = render_surface_points_depth(
        points, np.eye(4), intrinsics, width=width, height=height, splat_radius_pixels=0,
    )
    expected = _old_min_dilation(np.where(single_pixels > 0, single_pixels, np.inf), radius)
    expected[~np.isfinite(expected)] = 0.0
    actual = render_surface_points_depth(
        points, np.eye(4), intrinsics, width=width, height=height, splat_radius_pixels=radius,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("radius", [0, 1, 3])
def test_robot_mask_matches_previous_dilation(radius: int) -> None:
    rng = np.random.default_rng(22)
    rendered = rng.uniform(0.3, 2.0, (19, 31)).astype(np.float32)
    measured = rendered + rng.normal(0, 0.025, rendered.shape).astype(np.float32)
    rendered[rng.random(rendered.shape) < 0.65] = 0.0
    rendered[0, 0] = 0.8
    rendered[1, 1] = np.nan
    measured[4, 4] = 0.0
    expanded = _old_min_dilation(np.where(np.isfinite(rendered) & (rendered > 0), rendered, np.inf), radius)
    visible = np.isfinite(expanded)
    valid = np.isfinite(measured) & (measured > 0)
    expected = ~(visible & valid & (np.abs(measured - expanded) <= 0.008 + 0.01 * expanded))
    actual = robot_depth_keep_mask(measured, rendered, dilation_pixels=radius)
    np.testing.assert_array_equal(actual, expected)
