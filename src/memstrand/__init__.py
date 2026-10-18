"""memstrand: local-first typed memory for AI agents."""

from importlib.metadata import PackageNotFoundError, version

from .api import Memory, StoreLocked
from .config import ConfigError, Settings, get_settings
from .schema import Category, MemoryAtom, MemoryEdge, Provenance, RelationType

try:
    __version__ = version("memstrand")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0+local"
__all__ = [
    "Category",
    "ConfigError",
    "Memory",
    "MemoryAtom",
    "MemoryEdge",
    "Provenance",
    "RelationType",
    "Settings",
    "StoreLocked",
    "get_settings",
]
