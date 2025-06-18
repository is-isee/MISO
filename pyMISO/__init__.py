"""
Package for dealing with rmhdlab data.
"""

from .conf import Conf
from .grid import Grid
from .time import Time
from .data import Data
from .mpi import MPI

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'