"""
Package for dealing with MISO simulation data.
"""

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
