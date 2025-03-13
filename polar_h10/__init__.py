import importlib.metadata

from .core import NotificationType, PolarH10

__version__ = importlib.metadata.version(__package__ or __name__)
__version_info__ = tuple(int(i) for i in __version__.split(".") if i.isdigit())

__all__ = ["NotificationType", "PolarH10"]
