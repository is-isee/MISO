class Time:
    def __init__(self, conf):
        """
        Initialize the pyMISO.Time class instance

        Parameters
        ----------
        data_dir : str
            The directory where the config.nml file is located
        """

        self.time_data_dir = conf.time_data_dir
        self.n_output_digits = conf.n_output_digits
        with (self.time_data_dir / "n_output.txt").open("r") as f:
            self.n_output = int(f.readline())

    def load(self, n_output):
        with (
            self.time_data_dir / "time."
            + str(n_output).zfill(self.n_output_digits)
            + ".txt",
        ).open(mode="r") as f:
            self.time = float(f.readline())
            self.n_output = int(f.readline())
            self.n_step = int(f.readline())
