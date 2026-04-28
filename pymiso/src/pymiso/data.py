import numpy as np

from .conf import Conf
from .grid import Grid
from .mpi import MPI
from .time import Time


class Data:
    """
    Class for handling the simulation data
    """

    def __init__(self, data_dir: str):
        """
        Initialize the :class:`~pymiso.Data` class instance

        Parameters
        ----------
        data_dir : str
            The directory where the ``config.yaml`` file is located
        """
        self.conf = Conf(data_dir)
        self.mpi = MPI(self.conf)
        self.grid = Grid(self.conf)
        self.time = Time(self.conf)
        self.model = self._detect_model()

    def _detect_model(self) -> str:
        """
        Detect the model type from the ``physics`` config section.
        """
        if self.conf.physics.rt and self.conf.physics.mhd:
            raise ValueError("physics.mhd and physics.rt cannot both be true")
        if self.conf.physics.rt:
            return "rt"
        if self.conf.physics.mhd:
            return "mhd"
        raise ValueError("Either physics.mhd or physics.rt must be true")

    def load(self, n_output: int):
        """
        Load snapshot at a specified time index

        Parameters
        ----------
        n_output : int
            The output number to load the data from
        """
        if self.model == "mhd":
            self._load_mhd(n_output)
        elif self.model == "rt":
            self._load_rt(n_output)
        else:
            raise ValueError(f"Unsupported model type: {self.model}")

    def _global_slice(self, rank: int):
        """
        Return the global-domain slice for a rank.
        """
        if self.mpi.n_procs > 1:
            return (
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
            )

        return (
            slice(0, self.grid.i_size_local),
            slice(0, self.grid.j_size_local),
            slice(0, self.grid.k_size_local),
        )

    def _local_slice(self):
        """
        Return the local physical-domain slice excluding guard cells.
        """
        return (
            slice(self.grid.i_margin, self.grid.i_total_local - self.grid.i_margin),
            slice(self.grid.j_margin, self.grid.j_total_local - self.grid.j_margin),
            slice(self.grid.k_margin, self.grid.k_total_local - self.grid.k_margin),
        )

    def _dtype_from_elem_size(self, elem_size: np.uint32) -> str:
        """
        Map stored element size to a NumPy dtype suffix.
        """
        if elem_size == 4:
            return "f4"
        if elem_size == 8:
            return "f8"
        raise ValueError(f"Unexpected element size: {elem_size}")

    def _load_mhd(self, n_output: int):
        """
        Load MHD snapshot data.
        """
        shape_global = (self.grid.i_size, self.grid.j_size, self.grid.k_size)

        self.time.load(n_output)

        shape = (
            self.grid.i_total_local,
            self.grid.j_total_local,
            self.grid.k_total_local,
        )
        count = np.prod(shape)

        for rank in range(self.mpi.n_procs):
            filename = self.conf.mhd_data_dir / (
                f"mhd.{n_output:0{self.conf.io.n_output_digits}d}"
                f".{rank:0{self.conf.io.n_output_digits}d}.bin"
            )

            ijk_global = self._global_slice(rank)
            ijk_local = self._local_slice()

            with open(filename, "rb") as f:
                elem_size = np.fromfile(f, dtype=self.conf.endian + "u4", count=1)[0]
                elem_base = self._dtype_from_elem_size(elem_size)
                nvar = 9  # ro, vx, vy, vz, bx, by, bz, ei, ph

                if rank == 0:
                    self.ro = np.zeros(shape_global, dtype=elem_base)
                    self.vx = np.zeros_like(self.ro)
                    self.vy = np.zeros_like(self.ro)
                    self.vz = np.zeros_like(self.ro)
                    self.bx = np.zeros_like(self.ro)
                    self.by = np.zeros_like(self.ro)
                    self.bz = np.zeros_like(self.ro)
                    self.ei = np.zeros_like(self.ro)
                    self.ph = np.zeros_like(self.ro)

                data = np.fromfile(
                    f, dtype=self.conf.endian + elem_base, count=nvar * count
                ).reshape((nvar, *shape), order="C")
                self.ro[ijk_global] = data[0][ijk_local]
                self.vx[ijk_global] = data[1][ijk_local]
                self.vy[ijk_global] = data[2][ijk_local]
                self.vz[ijk_global] = data[3][ijk_local]
                self.bx[ijk_global] = data[4][ijk_local]
                self.by[ijk_global] = data[5][ijk_local]
                self.bz[ijk_global] = data[6][ijk_local]
                self.ei[ijk_global] = data[7][ijk_local]
                self.ph[ijk_global] = data[8][ijk_local]

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

    def _load_rt(self, n_output: int):
        """
        Load RT snapshot data.
        """
        shape_global = (self.grid.i_size, self.grid.j_size, self.grid.k_size)
        shape_local = (
            self.grid.i_total_local,
            self.grid.j_total_local,
            self.grid.k_total_local,
        )
        count_local = np.prod(shape_local)

        self.time.load(n_output)

        for rank in range(self.mpi.n_procs):
            filename = self.conf.rt_data_dir / f"rank_{rank:06d}.bin"
            ijk_global = self._global_slice(rank)
            ijk_local = self._local_slice()

            with open(filename, "rb") as f:
                num_rays = int(
                    np.fromfile(f, dtype=self.conf.endian + "i4", count=1)[0]
                )

                if rank == 0:
                    self.num_rays = num_rays
                    elem_probe = np.fromfile(f, dtype=self.conf.endian + "f4", count=1)
                    if elem_probe.size == 0:
                        raise ValueError("RT file is missing angular weights")
                    f.seek(-elem_probe.dtype.itemsize, 1)
                    elem_base = "f4"
                    self.weights = np.empty(self.num_rays, dtype=elem_base)
                    self.mu_x = np.empty(self.num_rays, dtype=elem_base)
                    self.mu_y = np.empty(self.num_rays, dtype=elem_base)
                    self.mu_z = np.empty(self.num_rays, dtype=elem_base)
                    self.src_func = np.zeros(shape_global, dtype=elem_base)
                    self.abs_coeff = np.zeros(shape_global, dtype=elem_base)
                    self.rint = np.zeros(
                        (self.num_rays, *shape_global), dtype=elem_base
                    )
                else:
                    if num_rays != self.num_rays:
                        raise ValueError(
                            f"Inconsistent num_rays: {num_rays} != {self.num_rays}"
                        )
                    elem_base = self.weights.dtype.str[1:]

                dtype = np.dtype(self.conf.endian + elem_base)
                weights = np.fromfile(f, dtype=dtype, count=num_rays)
                mu_x = np.fromfile(f, dtype=dtype, count=num_rays)
                mu_y = np.fromfile(f, dtype=dtype, count=num_rays)
                mu_z = np.fromfile(f, dtype=dtype, count=num_rays)
                src_func = np.fromfile(f, dtype=dtype, count=count_local).reshape(
                    shape_local, order="C"
                )
                abs_coeff = np.fromfile(f, dtype=dtype, count=count_local).reshape(
                    shape_local, order="C"
                )
                rint = np.fromfile(
                    f, dtype=dtype, count=num_rays * count_local
                ).reshape((num_rays, *shape_local), order="C")

                if rank == 0:
                    self.weights[:] = weights
                    self.mu_x[:] = mu_x
                    self.mu_y[:] = mu_y
                    self.mu_z[:] = mu_z
                else:
                    if not (
                        np.array_equal(self.weights, weights)
                        and np.array_equal(self.mu_x, mu_x)
                        and np.array_equal(self.mu_y, mu_y)
                        and np.array_equal(self.mu_z, mu_z)
                    ):
                        raise ValueError(
                            "Inconsistent RT angular quadrature across ranks"
                        )

                self.src_func[ijk_global] = src_func[ijk_local]
                self.abs_coeff[ijk_global] = abs_coeff[ijk_local]
                self.rint[(slice(None),) + ijk_global] = rint[
                    (slice(None),) + ijk_local
                ]

        self.src_func = np.squeeze(self.src_func)
        self.abs_coeff = np.squeeze(self.abs_coeff)
        self.rint = np.squeeze(self.rint)
        self.weights = np.squeeze(self.weights)
        self.mu_x = np.squeeze(self.mu_x)
        self.mu_y = np.squeeze(self.mu_y)
        self.mu_z = np.squeeze(self.mu_z)

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
        pv_grid.origin = (self.grid.x_min, self.grid.y_min, self.grid.z_min)
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
        pv_grid.origin = (self.grid.x_min, self.grid.y_min, self.grid.z_min)
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

    def __getattr__(self, name):
        """
        When an attribute is not found, it is searched in grid, time, and mpi.
        """
        for obj in [self.grid, self.time, self.mpi]:
            if hasattr(obj, name):
                # Only when the attribute is not a function, return it
                if not callable(getattr(obj, name)):
                    return getattr(obj, name)

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )
