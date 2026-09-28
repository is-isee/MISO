#pragma once

// Umbrella header for the common layer shared by MHD and RT.
// Include this together with a physics header, e.g.
//   #include <miso/core.hpp>
//   #include <miso/mhd_model_base.hpp>  // or <miso/rt.hpp>

// fundamentals
#include "backend.hpp"
#include "types.hpp"
#include "utility.hpp"

// infrastructure
#include "config.hpp"
#include "env.hpp"
#include "execution.hpp"
#include "mpi_util.hpp"

// data structures
#include "array1d.hpp"
#include "array2d.hpp"
#include "array3d.hpp"
#include "array4d.hpp"
#include "grid.hpp"
#include "time.hpp"

// shared by MHD and RT
#include "boundary_condition.hpp"
