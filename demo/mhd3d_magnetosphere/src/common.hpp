#pragma once
#include <miso/core.hpp>
#include <miso/mhd.hpp>

using namespace miso;

using Real = float;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif
