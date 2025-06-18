import yaml

class Conf:
    def __init__(self, data_dir):
        """
        Initialize the pyrmhd.Conf class instance
        
        Parameters
        ----------
        data_dir : str
            The directory where the config.json file is located
        """
        self.data_dir = data_dir
        
        self.load()
        self.set_ijk_params()
    
    def load(self):
        """
        Load the config.json file in the save_dir and set the parameters as attributes
        """
        with open(self.data_dir + 'config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        for group in config.keys():
            for key in config[group].keys():
                setattr(self, key, config[group][key])
                
        self.time_data_dir = self.data_dir + self.time_save_dir
        self.mhd_data_dir = self.data_dir + self.mhd_save_dir
        self.mpi_data_dir = self.data_dir + self.mpi_save_dir
        
        self.endian = '<' if self.Endian == 'little' else '>'
        self.dtype = 'd' if self.Real == 'double' else 'f'
    
    def set_ijk_params(self):
        """
        Set the i, j, and k parameters
        """
        self.i_stride = 1 if self.i_size > 1 else 0
        self.j_stride = 1 if self.j_size > 1 else 0
        self.k_stride = 1 if self.k_size > 1 else 0
    
        self.i_margin = self.margin*self.i_stride
        self.j_margin = self.margin*self.j_stride
        self.k_margin = self.margin*self.k_stride
    
        self.i_total = self.i_size + 2*self.i_margin
        self.j_total = self.j_size + 2*self.j_margin
        self.k_total = self.k_size + 2*self.k_margin
        
        self.i_size_local = int(self.i_size / self.x_procs)
        self.j_size_local = int(self.j_size / self.y_procs)
        self.k_size_local = int(self.k_size / self.z_procs)
        self.i_total_local = self.i_size_local + 2*self.i_margin
        self.j_total_local = self.j_size_local + 2*self.j_margin
        self.k_total_local = self.k_size_local + 2*self.k_margin