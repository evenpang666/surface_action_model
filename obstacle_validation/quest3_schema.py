"""Quest3 dual-arm joint naming shared by LeRobot conversion."""

import json


JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
NAMES = [f"{arm}/{joint}" for arm in ("left_arm", "right_arm") for joint in JOINTS]


def _joint_order(file, attribute):
    names = json.loads(file.attrs[attribute])
    if not isinstance(names, list) or len(names) != 12 or set(names) != set(NAMES):
        raise ValueError(f"{attribute} must identify both UR arms and all six joints exactly once")
    return names, [names.index(name) for name in NAMES]


