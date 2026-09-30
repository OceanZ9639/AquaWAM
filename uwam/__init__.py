"""Package init: keep torch-free so ROS collection scripts can import Cfg/sim."""

from .config import Cfg, ensure_dirs

__all__ = ["Cfg", "ensure_dirs"]
