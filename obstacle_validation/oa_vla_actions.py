"""Explicit policy-to-controller action contracts and bounded execution admission."""

from dataclasses import dataclass
import math
import time
import numpy as np

DELTA_LAYOUT = "interleaved_joint_delta_gripper_target"
ABSOLUTE_LAYOUT = "absolute_q12_gripper2"


class ObservationHold(RuntimeError):
    """No motion this cycle; acquire a fresh observation and infer again."""


def validate_server_metadata(metadata):
    if (metadata.get("test_only") is True or metadata.get("model_type") != "oa-vla" or metadata.get("model_format") != 2
            or metadata.get("state_dim") != 12 or metadata.get("action_dim") != 14
            or metadata.get("policy_views") != ["front", "side", "right_wrist", "left_wrist"]
            or metadata.get("detection_views") != ["front", "side"]
            or metadata.get("detection_box_format") != "cxcywh_angle_deg_original_pixels"
            or metadata.get("pair_id_scope") != "current_prediction"
            or metadata.get("detection_trained") is not True):
        raise ValueError("Execution requires a trained four-view OA-VLA format-2 paired-OBB policy")
    if metadata.get("action_semantics") not in (DELTA_LAYOUT, ABSOLUTE_LAYOUT):
        raise ValueError("Server must declare its trained action semantics; cannot infer from dimension 14")
    if metadata["action_semantics"] == DELTA_LAYOUT and metadata.get("action_source") not in ("commanded", "executed"):
        raise ValueError("Delta policy must declare commanded/executed action source")
    period = metadata.get("action_period_s")
    if not isinstance(period, (float, int)) or not math.isfinite(period) or not .002 <= period <= 2:
        raise ValueError("Server must declare a valid trained action_period_s")
    return float(period)


def first_action_target(prediction, observed_q, semantics, *, lower, upper):
    """Use only the first predicted action. Never cumsum future commanded deltas."""
    actions = np.asarray(prediction.get("actions"), dtype=float)
    observed_q = np.asarray(observed_q, float)
    if (observed_q.shape != (12,) or not np.isfinite(observed_q).all()
            or actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= 1024
            or not np.isfinite(actions).all()):
        raise ValueError("Policy action chunk must be finite [H,14]")
    if semantics == DELTA_LAYOUT:
        action = np.r_[observed_q + np.r_[actions[0, :6], actions[0, 7:13]], actions[0, 6], actions[0, 13]]
    elif semantics == ABSOLUTE_LAYOUT:
        action = actions[0].copy()
    else:
        raise ValueError("Unknown action layout")
    if (np.any(action[:12] < lower) or np.any(action[:12] > upper)
            or np.any((action[12:] < 0) | (action[12:] > 1))):
        raise ObservationHold("Predicted joint/gripper target exceeds hardware contract")
    return action


def validate_age_and_drift(bundle, feedback, *, max_age_s, max_drift_rad, max_feedback_age_s, now_ns=None):
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if not 0 <= (now_ns - bundle.capture_ns) / 1e9 <= max_age_s:
        raise ObservationHold("VLA result/camera frame expired")
    if not 0 <= (now_ns - feedback.timestamp_ns) / 1e9 <= max_feedback_age_s:
        raise ObservationHold("Robot feedback expired")
    if np.max(np.abs(feedback.q - bundle.feedback.q)) > max_drift_rad:
        raise ObservationHold("Robot moved too far since inference observation")


def check_optimized_step(info, q, target, *, max_step, max_slack_m=1e-5, barrier_tolerance_m=1e-5):
    target, q = np.asarray(target, float), np.asarray(q, float)
    if (target.shape != (14,) or not np.isfinite(target).all() or not info.get("success")
            or np.max(np.abs(target[:12] - q)) > max_step + 1e-8
            or np.any((target[12:] < 0) | (target[12:] > 1))):
        raise ObservationHold("Optimizer failed or exceeded a bounded control step")
    steps = info.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ObservationHold("Optimizer did not report constraint diagnostics")
    detail = steps[0]
    slack = detail.get("max_distance_slack_m")
    barrier = detail.get("minimum_predicted_barrier_after_m")
    if (not isinstance(slack, (float, int)) or not math.isfinite(slack) or slack < 0 or slack > max_slack_m
            or not isinstance(barrier, (float, int)) or math.isnan(barrier) or barrier < -barrier_tolerance_m):
        raise ObservationHold("Optimized step has unacceptable distance slack or predicted barrier")
    return target
