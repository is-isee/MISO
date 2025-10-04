import numpy as np

import pyMISO


class MPI:
    """
    Class to handle MPI data.
    """

    def __init__(self, conf: pyMISO.Conf):
        """
        Initialize the pyMISO.Grid class instance

        Parameters
        ----------
        conf : pyMISO.Conf
            Instance of pyMISO.Conf class
        """

        self.load(conf)

    def load(self, conf: pyMISO.Conf):
        """
        Load the coords.csv file in the save_dir

        Parameters
        ----------
        conf : pyMISO.Conf
            Instance of pyMISO.Conf class
        """

        for group, values in conf.mpi.items():
            setattr(self, group, values)
        self.n_procs = self.x_procs * self.y_procs * self.z_procs
        self.coords = np.genfromtxt(
            (conf.mpi_data_dir / "coords.csv"), delimiter=",", names=True, dtype=int
        )
