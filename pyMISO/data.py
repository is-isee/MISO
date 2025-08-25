import numpy as np

import pyMISO


class Data:
    def __init__(self, data_dir):
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

    def load(self, n_output):
        """
        Load the config.json file in the data_dir and set the parameters as attributes
        using np.memmap for efficient access to large binary data.
        """
        self.ro = np.zeros((self.i_size, self.j_size, self.k_size), dtype=self.dtype)
        self.vx = np.zeros_like(self.ro)
        self.vy = np.zeros_like(self.ro)
        self.vz = np.zeros_like(self.ro)
        self.bx = np.zeros_like(self.ro)
        self.by = np.zeros_like(self.ro)
        self.bz = np.zeros_like(self.ro)
        self.ei = np.zeros_like(self.ro)
        self.ph = np.zeros_like(self.ro)

        self.time.load(n_output)

        shape = (self.i_total_local, self.j_total_local, self.k_total_local)
        count = np.prod(shape)

        for rank in range(self.mpi.n_procs):
            filename = (
                self.mhd_data_dir
                + "mhd."
                + str(n_output).zfill(self.n_output_digits)
                + "."
                + str(rank).zfill(self.n_procs_digits)
                + ".bin"
            )

            if self.mpi.n_procs > 1:
                ijk_global = [
                    slice(
                        self.mpi.coords["x"][rank] * self.i_size_local,
                        (self.mpi.coords["x"][rank] + 1) * self.i_size_local,
                    ),
                    slice(
                        self.mpi.coords["y"][rank] * self.j_size_local,
                        (self.mpi.coords["y"][rank] + 1) * self.j_size_local,
                    ),
                    slice(
                        self.mpi.coords["z"][rank] * self.k_size_local,
                        (self.mpi.coords["z"][rank] + 1) * self.k_size_local,
                    ),
                ]
            else:
                ijk_global = [
                    slice(0, self.i_size_local),
                    slice(0, self.j_size_local),
                    slice(0, self.k_size_local),
                ]

            ijk_local = [
                slice(self.i_margin, self.i_total_local - self.i_margin),
                slice(self.j_margin, self.j_total_local - self.j_margin),
                slice(self.k_margin, self.k_total_local - self.k_margin),
            ]

            with open(filename, "rb") as f:
                self.ro[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vx[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vy[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.vz[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.bx[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.by[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.bz[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.ei[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]
                self.ph[tuple(ijk_global)] = np.fromfile(
                    f, dtype=self.endian + self.dtype, count=count
                ).reshape(shape, order="C")[tuple(ijk_local)]

        # squeezeはオプション（必要なら）
        self.ro = np.squeeze(self.ro)
        self.vx = np.squeeze(self.vx)
        self.vy = np.squeeze(self.vy)
        self.vz = np.squeeze(self.vz)
        self.bx = np.squeeze(self.bx)
        self.by = np.squeeze(self.by)
        self.bz = np.squeeze(self.bz)
        self.ei = np.squeeze(self.ei)
        self.ph = np.squeeze(self.ph)

    def __getattr__(self, name):
        """
        When an attribute is not found in pyMISO.Data, it is searched in pyMISO.MHD.conf and pyMISO.MHD.grid
        """
        for obj in [self.conf, self.grid, self.time]:
            if hasattr(obj, name):
                # Only when the attribute is not a function, return it
                if not callable(getattr(obj, name)):
                    return getattr(obj, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
