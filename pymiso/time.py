import pymiso


class Time:
    """
    Class to handle the time data
    """

    def __init__(self, conf: pymiso.Conf):
        """
        Initialize the pymiso.Time class instance

        Parameters
        ----------
        conf : pymiso.Conf
            Instance of pymiso.Conf class
        """

        for group, values in conf.time.items():
            setattr(self, group, values)

        self.time_data_dir = conf.time_data_dir
        with (self.time_data_dir / "n_output.txt").open("r") as f:
            self.n_output = int(f.readline())

    def load(self, n_output: int):
        """
        Load the time data from n_output

        Parameters
        ----------
        n_output : int
            The output number to load the time data from
        """
        with (
            self.time_data_dir
            / ("time." + str(n_output).zfill(self.n_output_digits) + ".txt")
        ).open(mode="r") as f:
            self.time = float(f.readline())
            self.load_n_output = int(f.readline())
            self.load_n_step = int(f.readline())
