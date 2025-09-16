import pyvista as pv

import pyMISO


def write_scalar_vtk(data, n_output, var, output_path):
    """
    Write a scalar variable to a VTK file for visualization using PyVista.

    Parameters
    ----------
    data : pyMISO.Data
        Instance of pyMISO.Data class containing the simulation data.
    n_output : int
        The output number to load the data from.
    var : str
        The variable name to write (e.g., 'ro', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'ei', 'ph').
    output_path : str
        The path to save the VTK file.
    """
    if not isinstance(data, pyMISO.Data):
        raise TypeError("data must be an instance of pyMISO.Data")

    if not isinstance(n_output, int):
        raise TypeError("n_output must be an integer")

    if not isinstance(output_path, str):
        raise TypeError("output_path must be a string")

    if var not in ["ro", "vx", "vy", "vz", "bx", "by", "bz", "ei", "ph"]:
        raise ValueError(
            "var must be one of 'ro', 'vx', 'vy', 'vz', 'bx', 'by', 'bz', 'ei', 'ph'"
        )

    read_flag = True
    if hasattr(data, "n_output"):
        if n_output == data.n_output:
            read_flag = False
    if read_flag:
        data.load(n_output)

    pv_grid = pv.RectilinearGrid(data.x_edge, data.y_edge, data.z_edge)
    pv_grid.cell_data["scalar"] = getattr(data, var).flatten(order="F")

    pv_grid.save(output_path)
