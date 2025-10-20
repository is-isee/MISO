import numpy as np

import pymiso


class MPI:
    """
    Class to handle MPI data.
    """

    def __init__(self, conf: pymiso.Conf):
        """
        Initialize the pymiso.Grid class instance

        Parameters
        ----------
        conf : pymiso.Conf
            Instance of pymiso.Conf class
        """

        self.load(conf)

    def load(self, conf: pymiso.Conf):
        """
        Load the coords.csv file in the save_dir

        Parameters
        ----------
        conf : pymiso.Conf
            Instance of pymiso.Conf class
        """

        for group, values in conf.mpi.items():
            setattr(self, group, values)
        self.n_procs = self.x_procs * self.y_procs * self.z_procs
        self.coords = np.genfromtxt(
            (conf.mpi_data_dir / "coords.csv"), delimiter=",", names=True, dtype=int
        )
