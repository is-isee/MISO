#pragma once

namespace miso {
inline constexpr const char *config_default_yaml = R"yaml(
io:
  # Whether to enable I/O operations
  enabled: true
  # Directory for saving all output files
  save_dir: data/
  # Directory for saving time-related files
  time_save_dir: time/
  # Directory for saving MPI-related files
  mpi_save_dir: mpi/
  # Directory for saving MHD simulation checkpoint files
  mhd_save_dir: mhd/
  # Number of digits for output file naming (e.g., 8 means 00000001)
  n_output_digits: 8

mhd:
    # CFL number for time-stepping
    cfl_number: 0.5
    artificial_viscosity:
        # ep and fh represent epsilon and h in Rempel, 2014, ApJ, 789, 132. In Rempel, 2014, epsilon = 2 is implicitly used in eq. (8).
        ep: 1.0
        fh: 1.0
        # factors for controlling the strength of artificial viscosity for different characteristic velocities
        cs_fac: 1.0
        ca_fac: 1.0
        vv_fac: 1.0
    )yaml";
}
