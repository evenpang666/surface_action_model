"""Public camera and output contract. All names are physical view names."""

POLICY_VIEWS = ("front", "side", "right_wrist", "left_wrist")
DETECTION_VIEWS = ("front", "side")
CAMERA_TO_MODEL = {name: name for name in POLICY_VIEWS}
POLICY_IMAGE_KEYS = POLICY_VIEWS
DETECTION_IMAGE_KEYS = DETECTION_VIEWS
MODEL_TYPE = "oa-vla"
SCHEMA_VERSION = 2
BOX_FORMAT = "cxcywh_cos2sin2_normalized_padded"
IMAGE_RESOLUTION = 224
