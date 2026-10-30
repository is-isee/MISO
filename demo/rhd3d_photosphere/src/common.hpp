#pragma once
#include <miso/core.hpp>
#include <miso/eos_table.hpp>
#include <miso/mhd.hpp>
#include <miso/rt_mhd.hpp>
#include <miso/stratification.hpp>

using namespace miso;

using Real = double;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif
