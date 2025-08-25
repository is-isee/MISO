import numpy as np


class MPI:
    def __init__(self, conf):
        """
        Initialize the pyMISO.Grid class instance

        Parameters
        ----------
        conf : pyMISO.Conf
            Instance of pyMISO.Conf class
        """

        self.load(conf)

    def load(self, conf):
        """
        Load the coords.csv file in the save_dir
        """

        self.n_procs = conf.x_procs * conf.y_procs * conf.z_procs
        self.coords = np.genfromtxt(
            conf.mpi_data_dir + "coords.csv", delimiter=",", names=True, dtype=int
        )
