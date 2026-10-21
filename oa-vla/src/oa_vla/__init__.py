"""OA-VLA: standalone multi-view actions and obstacle-aware visual grounding."""

from .config import OAVLAConfig

__version__ = "0.3.0"
__all__ = ["OAVLA", "OAVLAConfig", "OAVLAPolicy", "Observation"]


def __getattr__(name):
    if name == "OAVLA":
        from .model import OAVLA
        return OAVLA
    if name == "OAVLAPolicy":
        from .policy import OAVLAPolicy
        return OAVLAPolicy
    if name == "Observation":
        from .observation import Observation
        return Observation
    raise AttributeError(name)
