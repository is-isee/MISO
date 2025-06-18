import numpy as np
class Grid:    
    def __init__(self, conf):
        """
        Initialize the pyMISO.Grid class instance
        
        Parameters
        ----------
        conf : pyMISO.Conf
            Instance of pyMISO.Conf class
        """
        
        self.load(conf)
        
    def load(self, conf):
        """
        Load the grid.dat file in the save_dir and set the grid points as attributes
        """
        
        dtype = np.dtype([
            ('x', conf.endian+str(conf.i_total) + conf.dtype),
            ('y', conf.endian+str(conf.j_total) + conf.dtype),
            ('z', conf.endian+str(conf.k_total) + conf.dtype),
            ])
        with open(conf.data_dir + 'grid.bin', 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
            self.x = data['x'].reshape((conf.i_total), order='C')
            self.y = data['y'].reshape((conf.j_total), order='C')
            self.z = data['z'].reshape((conf.k_total), order='C')
