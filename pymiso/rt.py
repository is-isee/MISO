from pathlib import Path

from .conf import Conf
from .grid import Grid
from .mpi import MPI

# import numpy as np


class RT:
    """
    Class for handling the Radiative Transfer (RT) data
    """

    def __init__(self, conf: Conf, mpi: MPI, grid: Grid):
        """
        Initialize the :class:`~pymiso.RT` class instance

        Parameters
        ----------
        data_dir : str
            The directory where the ``config.yaml`` file is located
        """
        self.conf = conf
        self.mpi = mpi
        self.grid = grid

    def load(self, data_file: Path):
        """
        Load data file

        Parameters
        ----------
        data_file : Path
            The path to the data file to load
        """
        assert data_file.is_file()
        pass  # To be implemented in the future
