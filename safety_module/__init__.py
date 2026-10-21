"""Trainable safety modules for robot link-point prediction."""

# Geometry/perception callers must not import model/GPU dependencies merely
# by importing safety_module. Preserve the original public model exports lazily.
def __getattr__(name):
    if name in ("SafetyPointDecoder", "SafetyPointDecoderConfig"):
        from importlib import import_module
        value = getattr(import_module("safety_module.point_decoder"), name)
    elif name in __all__ and name != "predicted_link_points_collision":
        from importlib import import_module
        value = getattr(import_module("safety_module.safety_flow_point_model"), name)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


def predicted_link_points_collision(*args, **kwargs):
    from safety_module.geometric_safety import predicted_link_points_collision as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "ArmPointTokenEmbedding",
    "FlowPointDecoderLayer",
    "FlowPointHead",
    "MLP",
    "PrefixPointEncoder",
    "PrefixTokenAdapter",
    "SafetyFlowPointModel",
    "SafetyPointDecoder",
    "SafetyPointDecoderConfig",
    "SinusoidalTimeEmbedding",
    "euler_sample",
    "flow_matching_loss",
    "predicted_link_points_collision",
    "sample_flow_matching_batch",
]
