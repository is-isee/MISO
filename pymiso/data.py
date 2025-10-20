from .conf import Conf
from .grid import Grid
from .mhd import MHD
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
        self.mhd = MHD(self.conf, self.mpi, self.grid)

    def load(self, n_output: int):
        """
        Alias of :meth:`~pymiso.Time.load` and  :meth:`~pymiso.MHD.load`

        Parameters
        ----------
        n_output : int
            The output number to load the data from
        """
        self.time.load(n_output)
        self.mhd.load(n_output)

    def __getattr__(self, name):
        """
        Get access to (non-callable) attributes
        """
        for obj in [self.mpi, self.time, self.grid, self.mhd]:
            if hasattr(obj, name):
                # Only when the attribute is not a function, return it
                attr = getattr(obj, name)
                if not callable(attr):
                    return attr

        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __dir__(self):
        """
        List available attributes
        """
        attrs = set(super().__dir__())
        for obj in [self.mpi, self.time, self.grid, self.mhd]:
            attrs.update(
                attr
                for attr in dir(obj)
                if not callable(getattr(obj, attr)) and not attr.startswith("_")
            )
        return list(attrs)
