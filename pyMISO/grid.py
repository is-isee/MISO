import numpy as np


class Grid:
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
        Load the grid.dat file in the save_dir and set the grid points as attributes
        """

        dtype = np.dtype(
            [
                ("x", conf.endian + str(conf.i_total) + conf.dtype),
                ("y", conf.endian + str(conf.j_total) + conf.dtype),
                ("z", conf.endian + str(conf.k_total) + conf.dtype),
            ]
        )
        with open(conf.data_dir + "grid.bin", "rb") as f:
            data = np.fromfile(f, dtype=dtype)

            # geometry is defined at cell center
            self.x = data["x"].reshape((conf.i_total), order="C")[
                conf.margin : -conf.margin
            ]
            self.y = data["y"].reshape((conf.j_total), order="C")[
                conf.margin : -conf.margin
            ]
            self.z = data["z"].reshape((conf.k_total), order="C")[
                conf.margin : -conf.margin
            ]

            # geometry at cell edge
            self.x_edge = np.empty(conf.i_size + 1, dtype=conf.dtype)
            self.y_edge = np.empty(conf.j_size + 1, dtype=conf.dtype)
            self.z_edge = np.empty(conf.k_size + 1, dtype=conf.dtype)

            self.x_edge[1:-1] = 0.5 * (self.x[1:] + self.x[:-1])
            self.y_edge[1:-1] = 0.5 * (self.y[1:] + self.y[:-1])
            self.z_edge[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])

            self.x_edge[0] = conf.xmin
            self.x_edge[-1] = conf.xmax
            self.y_edge[0] = conf.ymin
            self.y_edge[-1] = conf.ymax
            self.z_edge[0] = conf.zmin
            self.z_edge[-1] = conf.zmax
