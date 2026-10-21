"""Camera and output contract for the point-free SAM 3 + PI0.5 policy."""

POLICY_VIEWS = ("front", "side", "right_wrist", "left_wrist")
DETECTION_VIEWS = ("front", "side")
CAMERA_TO_MODEL = {
    "front": "base_0_rgb",
    "side": "base_1_rgb",
    "right_wrist": "right_wrist_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
}
POLICY_IMAGE_KEYS = tuple(CAMERA_TO_MODEL.values())
DETECTION_IMAGE_KEYS = tuple(CAMERA_TO_MODEL[view] for view in DETECTION_VIEWS)
MODEL_TYPE = "Sam3VlaModelPytorch"
SCHEMA_VERSION = 5
BOX_FORMAT = "cxcywh_normalized_padded"
IMAGE_RESOLUTION = 224
