"""memstrand: local-first typed memory for AI agents."""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from .config import ConfigError, Settings, get_settings
from .schema import Category, MemoryAtom, MemoryEdge, Provenance, RelationType

if TYPE_CHECKING:
    # Only for type checkers -- see __getattr__ below for why these two
    # aren't imported eagerly at runtime.
    from .api import Memory, StoreLocked

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

# `Memory`/`StoreLocked` live in `.api`, which transitively imports
# `qdrant_client` -> fastembed -> onnxruntime (~0.4s and a stderr
# hardware-probe warning on some Linux VMs). Every other symbol here is
# cheap (pydantic/dataclasses only), but a plain `import memstrand` --
# which every CLI invocation does, including hooks and remote-mode
# commands that never touch local storage -- must not pay that cost just
# because `Memory` is exported. PEP 562 module `__getattr__` defers the
# `.api` import until `Memory`/`StoreLocked` are actually accessed.
_LAZY_API_ATTRS = ("Memory", "StoreLocked")


def __getattr__(name: str):
    if name not in _LAZY_API_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import api

    value = getattr(api, name)
    globals()[name] = value  # cache: subsequent access skips __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
