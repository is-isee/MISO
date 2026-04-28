""":mod:`pymiso`
Utilities for handling MISO simulation data.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["Conf", "Data", "Grid", "MPI", "Time"]

from .conf import Conf
from .data import Data
from .grid import Grid
from .mpi import MPI
from .time import Time

try:
    __version__ = version("pymiso")
except PackageNotFoundError:
    __version__ = "unknown"
