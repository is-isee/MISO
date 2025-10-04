from pathlib import Path

import yaml
from box import Box


class Conf:
    """
    Class to handle the configuration file
    """

    def __init__(self, data_dir: str):
        """
        Initialize the pyrmhd.Conf class instance

        Parameters
        ----------
        data_dir : str
            The directory where the config.json file is located
        """
        self.data_dir = Path(data_dir)
        self.load()

    def load(self):
        """
        Load the config.json file in the save_dir and set the parameters as attributes
        """
        with (self.data_dir / "config.yaml").open(mode="r") as f:
            config = Box(yaml.safe_load(f))

        for group, values in config.items():
            setattr(self, group, values)

        self.time_data_dir = self.data_dir / self.time.time_save_dir
        self.mhd_data_dir = self.data_dir / self.mhd.mhd_save_dir
        self.mpi_data_dir = self.data_dir / self.mpi.mpi_save_dir

        self.endian = "<" if self.data_type.Endian == "little" else ">"
        self.dtype = "d" if self.data_type.Real == "double" else "f"
