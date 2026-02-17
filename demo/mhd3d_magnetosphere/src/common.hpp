#pragma once
#include <string>

#include <miso/boundary_condition.hpp>
#include <miso/config.hpp>
#include <miso/env.hpp>
#include <miso/eos.hpp>
#include <miso/execution.hpp>
#include <miso/grid.hpp>
#include <miso/mhd.hpp>
#include <miso/mpi_util.hpp>
#include <miso/time.hpp>
#include <miso/types.hpp>
#include <miso/utility.hpp>

using namespace miso;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif
