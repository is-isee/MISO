from .conf import Conf


class Time:
    """
    Class for handling the simulation time
    """

    def __init__(self, conf: Conf):
        """
        Initialize :class:`~pymiso.Time` class instance

        Parameters
        ----------
        conf : Conf
            Instance of :class:`~pymiso.Conf` class
        """

        for group, values in conf.time.items():
            setattr(self, group, values)

        self.time_data_dir = conf.time_data_dir
        with (self.time_data_dir / "n_output.txt").open("r") as f:
            self.n_output = int(f.readline())

    def load(self, n_output: int):
        """
        Load the simulation time at a specified time index

        Parameters
        ----------
        n_output : int
            The output number to load the time data from
        """
        n_output_str = f"{n_output:0{self.n_output_digits}d}"
        self.time_data_file = self.time_data_dir / f"time.{n_output_str}.txt"
        with self.time_data_file.open(mode="r") as f:
            self.time = float(f.readline())
            self.load_n_output = int(f.readline())
            self.load_n_step = int(f.readline())
