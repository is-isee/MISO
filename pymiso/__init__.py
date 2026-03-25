"""Compatibility package for the in-repo ``src`` layout.

This keeps ``import pymiso`` working from the repository root while the
installable package lives under ``pymiso/src/pymiso``.
"""

from pathlib import Path

_SRC_PACKAGE_DIR = Path(__file__).resolve().parent / "src" / "pymiso"
if _SRC_PACKAGE_DIR.is_dir():
    __path__.append(str(_SRC_PACKAGE_DIR))

__all__ = ["Conf", "Data", "Grid", "MPI", "Time"]

from .conf import Conf
from .data import Data
from .grid import Grid
from .mpi import MPI
from .time import Time

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"
