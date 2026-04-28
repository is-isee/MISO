from pathlib import Path

import yaml
from box import Box


class Conf:
    """
    Class for handling model configuration from the ``config.yaml`` file
    """

    def __init__(self, data_dir: str):
        """
        Initialize the :class:`~pymiso.Conf` class instance

        Parameters
        ----------
        data_dir : str
            The directory where the ``config.yaml`` file is located
        """
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "config.yaml"
        self.load()

    def load(self):
        """
        Load the ``config.yaml`` file and set the parameters as attributes
        """
        with self.config_file.open(mode="r") as f:
            config = Box(yaml.safe_load(f))

        if "physics" not in config:
            config.physics = Box({"mhd": True, "rt": False})
        else:
            rt_enabled = bool(config.physics.get("rt", False))
            mhd_enabled = bool(config.physics.get("mhd", not rt_enabled))
            config.physics = Box({"mhd": mhd_enabled, "rt": rt_enabled})

        for group, values in config.items():
            setattr(self, group, values)

        if hasattr(self, "time"):
            self.time_data_dir = self.data_dir / self.io.time_save_dir
        if hasattr(self, "mhd"):
            self.mhd_data_dir = self.data_dir / self.io.mhd_save_dir
        if hasattr(self, "rt"):
            self.rt_data_dir = self.data_dir / self.io.rt_save_dir
        self.mpi_data_dir = self.data_dir / self.io.mpi_save_dir
        self.endian = "<" if self.data_type.Endian == "little" else ">"
