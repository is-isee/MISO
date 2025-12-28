#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#endif
