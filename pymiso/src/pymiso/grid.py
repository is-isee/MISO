import numpy as np

from .conf import Conf


class Grid:
    """
    Class for handling simulation grids from the ``grid.bin`` file
    """

    def __init__(self, conf: Conf):
        """
        Initialize the :class:`~pymiso.Grid` class instance

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """
        self.grid_file = conf.data_dir / "grid.bin"
        self.load(conf)

    def load(self, conf: Conf):
        """
        Load the ``grid.bin`` file and set the grid points as attributes

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """
        for group, values in conf.grid.items():
            setattr(self, group, values)
        self.set_ijk_params(conf)

        with self.grid_file.open(mode="rb") as f:
            elem_size = np.fromfile(f, dtype=conf.endian + "u4", count=1)[0]
            if elem_size == 4:
                base = "f4"
            elif elem_size == 8:
                base = "f8"
            else:
                raise ValueError(f"Unexpected element size: {elem_size}")
            dtype = np.dtype(
                [
                    ("x", conf.endian + base, (self.i_total,)),
                    ("y", conf.endian + base, (self.j_total,)),
                    ("z", conf.endian + base, (self.k_total,)),
                ]
            )
            data = np.fromfile(f, dtype=dtype, count=1)[0]

            # geometry is defined at cell center
            self.x = data["x"][self.i_margin : self.i_total - self.i_margin]
            self.y = data["y"][self.j_margin : self.j_total - self.j_margin]
            self.z = data["z"][self.k_margin : self.k_total - self.k_margin]

            # geometry at cell edge
            self.x_edge = np.empty(self.i_size + 1, dtype=self.x.dtype)
            self.y_edge = np.empty(self.j_size + 1, dtype=self.y.dtype)
            self.z_edge = np.empty(self.k_size + 1, dtype=self.z.dtype)

            self.x_edge[1:-1] = 0.5 * (self.x[1:] + self.x[:-1])
            self.y_edge[1:-1] = 0.5 * (self.y[1:] + self.y[:-1])
            self.z_edge[1:-1] = 0.5 * (self.z[1:] + self.z[:-1])

            self.x_edge[0] = self.x_min
            self.x_edge[-1] = self.x_max
            self.y_edge[0] = self.y_min
            self.y_edge[-1] = self.y_max
            self.z_edge[0] = self.z_min
            self.z_edge[-1] = self.z_max

    def set_ijk_params(self, conf: Conf):
        """
        Set start, end, margin, size, etc. for each direction

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """
        self.i_stride = 1 if self.i_size > 1 else 0
        self.j_stride = 1 if self.j_size > 1 else 0
        self.k_stride = 1 if self.k_size > 1 else 0

        self.i_margin = self.margin * self.i_stride
        self.j_margin = self.margin * self.j_stride
        self.k_margin = self.margin * self.k_stride

        self.i_total = self.i_size + 2 * self.i_margin
        self.j_total = self.j_size + 2 * self.j_margin
        self.k_total = self.k_size + 2 * self.k_margin

        assert self.i_size % conf.mpi.x_procs == 0
        assert self.j_size % conf.mpi.y_procs == 0
        assert self.k_size % conf.mpi.z_procs == 0
        self.i_size_local = self.i_size // conf.mpi.x_procs
        self.j_size_local = self.j_size // conf.mpi.y_procs
        self.k_size_local = self.k_size // conf.mpi.z_procs
        self.i_total_local = self.i_size_local + 2 * self.i_margin
        self.j_total_local = self.j_size_local + 2 * self.j_margin
        self.k_total_local = self.k_size_local + 2 * self.k_margin
