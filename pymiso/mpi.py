import numpy as np

from .conf import Conf


class MPI:
    """
    Class for handling MPI data from the ``coords.csv`` file
    """

    def __init__(self, conf: Conf):
        """
        Initialize the :class:`~pymiso.MPI` class instance

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """
        self.coords_file = conf.mpi_data_dir / "coords.csv"
        self.load(conf)

    def load(self, conf: Conf):
        """
        Load the ``coords.csv`` file

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """
        for group, values in conf.mpi.items():
            setattr(self, group, values)
        self.n_procs = self.x_procs * self.y_procs * self.z_procs
        self.coords = np.genfromtxt(
            self.coords_file, delimiter=",", names=True, dtype=int
        )
