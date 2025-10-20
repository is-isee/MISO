import numpy as np

from .conf import Conf
from .grid import Grid
from .mpi import MPI


class MHD:
    """
    Class for handling the MHD data
    """

    def __init__(self, conf: Conf, mpi: MPI, grid: Grid):
        """
        Initialize the :class:`~pymiso.MHD` class instance

        Parameters
        ----------
        data_dir : str
            The directory where the ``config.yaml`` file is located
        """
        self.conf = conf
        self.mpi = mpi
        self.grid = grid

    def load(self, n_output: int):
        """
        Load a snapshot at a specified time index

        Parameters
        ----------
        n_output : int
            The output number to load the data from
        """
        shape_global = (self.grid.i_size, self.grid.j_size, self.grid.k_size)
        self.ro = np.zeros(shape_global, dtype=self.conf.dtype)
        self.vx = np.zeros_like(self.ro)
        self.vy = np.zeros_like(self.ro)
        self.vz = np.zeros_like(self.ro)
        self.bx = np.zeros_like(self.ro)
        self.by = np.zeros_like(self.ro)
        self.bz = np.zeros_like(self.ro)
        self.ei = np.zeros_like(self.ro)
        self.ph = np.zeros_like(self.ro)

        shape = (
            self.grid.i_total_local,
            self.grid.j_total_local,
            self.grid.k_total_local,
        )
        count = np.prod(shape)

        for rank in range(self.mpi.n_procs):
            n_output_str = f"{n_output:0{self.conf.time.n_output_digits}d}"
            rank_str = f"{rank:0{self.mpi.n_procs_digits}d}"
            filename = self.conf.mhd_data_dir / f"mhd.{n_output_str}.{rank_str}.bin"

            if self.mpi.n_procs > 1:
                ijk_global = [
                    slice(
                        self.mpi.coords["x"][rank] * self.grid.i_size_local,
                        (self.mpi.coords["x"][rank] + 1) * self.grid.i_size_local,
                    ),
                    slice(
                        self.mpi.coords["y"][rank] * self.grid.j_size_local,
                        (self.mpi.coords["y"][rank] + 1) * self.grid.j_size_local,
                    ),
                    slice(
                        self.mpi.coords["z"][rank] * self.grid.k_size_local,
                        (self.mpi.coords["z"][rank] + 1) * self.grid.k_size_local,
                    ),
                ]
            else:
                ijk_global = [
                    slice(0, self.grid.i_size_local),
                    slice(0, self.grid.j_size_local),
                    slice(0, self.grid.k_size_local),
                ]

            ijk_local = [
                slice(self.grid.i_margin, self.grid.i_total_local - self.grid.i_margin),
                slice(self.grid.j_margin, self.grid.j_total_local - self.grid.j_margin),
                slice(self.grid.k_margin, self.grid.k_total_local - self.grid.k_margin),
            ]

            with open(filename, "rb") as f:
                self.ro[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vx[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vy[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vz[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.bx[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.by[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.bz[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.ei[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.ph[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.conf.endian + self.conf.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]

        self.ro = np.squeeze(self.ro)
        self.vx = np.squeeze(self.vx)
        self.vy = np.squeeze(self.vy)
        self.vz = np.squeeze(self.vz)
        self.bx = np.squeeze(self.bx)
        self.by = np.squeeze(self.by)
        self.bz = np.squeeze(self.bz)
        self.ei = np.squeeze(self.ei)
        self.ph = np.squeeze(self.ph)

        self.load_n_output = n_output

    def write_scalar_vtk(self, n_output: int, var: str, output_path: str):
        """
        Write a scalar variable to a VTK file for visualization using PyVista.

        Parameters
        ----------
        n_output : int
            The output number to load the data from.
        var : str
            The variable name to write (e.g., 'ro', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'ei', 'ph').
        output_path : str
            The path to save the VTK file.

        Notes
        -----
        In this version, uniform grid is assumed.
        """
        import pyvista as pv

        if not isinstance(n_output, int):
            raise TypeError("n_output must be an integer")

        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")

        if var not in ["ro", "vx", "vy", "vz", "bx", "by", "bz", "ei", "ph"]:
            raise ValueError(
                "var must be one of 'ro', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'ei', 'ph'"
            )

        read_flag = True
        if hasattr(self, "load_n_output"):
            if n_output == self.load_n_output:
                read_flag = False
        if read_flag:
            self.load(n_output)

        pv_grid = pv.ImageData()
        pv_grid.dimensions = (self.i_size + 1, self.j_size + 1, self.k_size + 1)
        pv_grid.origin = (self.grid.xmin, self.grid.ymin, self.grid.zmin)
        pv_grid.spacing = (
            self.grid.x[1] - self.grid.x[0],
            self.grid.y[1] - self.grid.y[0],
            self.grid.z[1] - self.grid.z[0],
        )
        # TODO: Retain the following alternative implementation for future support of non-uniform grids.
        # pv_grid = pv.RectilinearGrid(
        #     self.grid.x_edge, self.grid.y_edge, self.grid.z_edge
        # )
        pv_grid.cell_data["scalar"] = getattr(self, var).flatten(order="F")
        pv_grid = pv_grid.cell_data_to_point_data()

        pv_grid.save(output_path)

    def write_vector_vtk(self, n_output: int, var: str, output_path: str):
        """
        Write a vector variable to a VTK file for visualization using PyVista.

        Parameters
        ----------
        n_output : int
            The output number to load the data from.
        var : str
            The variable name to write ('v' for velocity, 'b' for magnetic field).
        output_path : str
            The path to save the VTK file.

        Notes
        -----
        In this version, uniform grid is assumed.
        """
        import pyvista as pv

        if not isinstance(n_output, int):
            raise TypeError("n_output must be an integer")

        if not isinstance(output_path, str):
            raise TypeError("output_path must be a string")

        if var not in ["v", "b"]:
            raise ValueError("var must be 'v' or 'b'")

        read_flag = True
        if hasattr(self, "load_n_output"):
            if n_output == self.load_n_output:
                read_flag = False
        if read_flag:
            self.load(n_output)

        pv_grid = pv.ImageData()
        pv_grid.dimensions = (self.i_size + 1, self.j_size + 1, self.k_size + 1)
        pv_grid.origin = (self.grid.xmin, self.grid.ymin, self.grid.zmin)
        pv_grid.spacing = (
            self.grid.x[1] - self.grid.x[0],
            self.grid.y[1] - self.grid.y[0],
            self.grid.z[1] - self.grid.z[0],
        )

        pv_grid.cell_data["vector"] = np.stack(
            (
                getattr(self, var + "x").flatten(order="F"),
                getattr(self, var + "y").flatten(order="F"),
                getattr(self, var + "z").flatten(order="F"),
            ),
            axis=-1,
        )
        pv_grid = pv_grid.cell_data_to_point_data()

        pv_grid.save(output_path)
