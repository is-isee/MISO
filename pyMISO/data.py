import numpy as np

import pyMISO


class Data:
    def __init__(self, data_dir: str):
        """
        Initialize the pyMISO.Data class instance

        Parameters
        ----------
        data_dir : str
            The directory where the config.nml file is located
        """

        self.conf = pyMISO.Conf(data_dir)
        self.mpi = pyMISO.MPI(self.conf)
        self.grid = pyMISO.Grid(self.conf)
        self.time = pyMISO.Time(self.conf)

    def load(self, n_output: int):
        """
        Load mhd quantities

        Parameters
        ----------
        n_output : int
            The output number to load the data from
        """

        self.ro = np.zeros(
            (self.i_size, self.j_size, self.k_size), dtype=self.conf.dtype
        )
        self.vx = np.zeros_like(self.ro)
        self.vy = np.zeros_like(self.ro)
        self.vz = np.zeros_like(self.ro)
        self.bx = np.zeros_like(self.ro)
        self.by = np.zeros_like(self.ro)
        self.bz = np.zeros_like(self.ro)
        self.ei = np.zeros_like(self.ro)
        self.ph = np.zeros_like(self.ro)

        self.time.load(n_output)

        shape = (
            self.conf.i_total_local,
            self.conf.j_total_local,
            self.conf.k_total_local,
        )
        count = np.prod(shape)

        for rank in range(self.mpi.n_procs):
            filename = self.conf.mhd_data_dir / (
                "mhd."
                + str(n_output).zfill(self.conf.n_output_digits)
                + "."
                + str(rank).zfill(self.conf.n_procs_digits)
                + ".bin"
            )

            if self.mpi.n_procs > 1:
                ijk_global = [
                    slice(
                        self.mpi.coords["x"][rank] * self.conf.i_size_local,
                        (self.mpi.coords["x"][rank] + 1) * self.conf.i_size_local,
                    ),
                    slice(
                        self.mpi.coords["y"][rank] * self.conf.j_size_local,
                        (self.mpi.coords["y"][rank] + 1) * self.conf.j_size_local,
                    ),
                    slice(
                        self.mpi.coords["z"][rank] * self.conf.k_size_local,
                        (self.mpi.coords["z"][rank] + 1) * self.conf.k_size_local,
                    ),
                ]
            else:
                ijk_global = [
                    slice(0, self.conf.i_size_local),
                    slice(0, self.conf.j_size_local),
                    slice(0, self.conf.k_size_local),
                ]

            ijk_local = [
                slice(self.i_margin, self.conf.i_total_local - self.i_margin),
                slice(self.j_margin, self.conf.j_total_local - self.j_margin),
                slice(self.k_margin, self.conf.k_total_local - self.k_margin),
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

    def write_scalar_vtk(self, n_output, var, output_path):
        """
        Write a scalar variable to a VTK file for visualization using PyVista.

        Parameters
        ----------
        self : pyMISO.Data
            Instance of pyMISO.Data class containing the simulation data.
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

        pv_grid.save(output_path)

    def __getattr__(self, name):
        """
        When an attribute is not found in pyMISO.Data, it is searched in grid, time, and mpi.
        """
        for obj in [self.grid, self.time, self.mpi]:
            if hasattr(obj, name):
                # Only when the attribute is not a function, return it
                if not callable(getattr(obj, name)):
                    return getattr(obj, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
