#pragma once
#include <miso/mhd_model_base.hpp>

using namespace miso;

#ifdef USE_CUDA
using Backend = backend::CUDA;
#else
using Backend = backend::Host;
#endif
