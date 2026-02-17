#pragma once

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

#ifndef __CUDACC__
#include <cmath>
#endif
#include "cuda_compat.hpp"

#ifdef __CUDA_ARCH__

#define MISO_MATH_UTILS(name, namef)                                             \
  __host__ __device__ inline float name(float x) { return ::namef(x); }          \
  __host__ __device__ inline double name(double x) { return ::name(x); }

#define MISO_MATH_UTILS2(name, namef)                                            \
  __host__ __device__ inline float name(float x, float y) {                      \
    return ::namef(x, y);                                                        \
  }                                                                              \
  __host__ __device__ inline double name(double x, double y) {                   \
    return ::name(x, y);                                                         \
  }

#else

#define MISO_MATH_UTILS(name, namef)                                             \
  __host__ __device__ inline float name(float x) { return std::name(x); }        \
  __host__ __device__ inline double name(double x) { return std::name(x); }

#define MISO_MATH_UTILS2(name, namef)                                            \
  __host__ __device__ inline float name(float x, float y) {                      \
    return std::name(x, y);                                                      \
  }                                                                              \
  __host__ __device__ inline double name(double x, double y) {                   \
    return std::name(x, y);                                                      \
  }

#endif

namespace miso {

namespace fs = std::filesystem;

/// @brief Utility functions
namespace util {

/// @brief Create directories if they do not exist (only on root process)
/// @details Do nothing if the directory already exists.
void create_directories(const std::string &dir_path) {
  fs::create_directories(dir_path);
}

/// @brief zero-fill integer to string
/// @param num target integer number
/// @param width width of the string
/// @return zero-filled string representation of the integer
inline std::string zfill(int num, int width) {
  std::ostringstream oss;
  oss << std::setw(width) << std::setfill('0') << num;
  return oss.str();
}

/// @brief Calculate the square of a value
/// @tparam T type of the value
/// @param x target value
/// @return squared value
template <typename T> __host__ __device__ inline T pow2(T x) { return x * x; }
template <typename T> __host__ __device__ inline T pow3(T x) { return x * x * x; }
template <typename T> __host__ __device__ inline T pow4(T x) {
  return x * x * x * x;
}
template <typename T> __host__ __device__ inline T pow5(T x) {
  return x * x * x * x * x;
}

// for float min
__host__ __device__ inline float fmin_safe(float a, float b) {
  return fminf(a, b);
}

// for float max
__host__ __device__ inline float fmax_safe(float a, float b) {
  return fmaxf(a, b);
}

// for double min
__host__ __device__ inline double dmin_safe(double a, double b) {
  return fmin(a, b);
}

// for double max
__host__ __device__ inline double dmax_safe(double a, double b) {
  return fmax(a, b);
}

template <typename T> __host__ __device__ inline T max2(T a, T b) {
  return fmax_safe(a, b);
}

template <typename T> __host__ __device__ inline T min2(T a, T b) {
  return fmin_safe(a, b);
}

template <typename T> __host__ __device__ inline T min3(T a, T b, T c) {
  return fmin_safe(fmin_safe(a, b), c);
}

template <typename T> __host__ __device__ inline T max3(T a, T b, T c) {
  return fmax_safe(fmax_safe(a, b), c);
}

MISO_MATH_UTILS(sqrt, sqrtf)
MISO_MATH_UTILS(sin, sinf)
MISO_MATH_UTILS(cos, cosf)
MISO_MATH_UTILS(tan, tanf)
MISO_MATH_UTILS(tanh, tanhf)
MISO_MATH_UTILS(asin, asinf)
MISO_MATH_UTILS(acos, acosf)
MISO_MATH_UTILS(exp, expf)
MISO_MATH_UTILS(log, logf)
MISO_MATH_UTILS(log10, log10f)
MISO_MATH_UTILS(fabs, fabsf)
MISO_MATH_UTILS2(atan2, atan2f)
MISO_MATH_UTILS2(copysign, copysignf)
MISO_MATH_UTILS2(fmin, fminf)
MISO_MATH_UTILS2(fmax, fmaxf)
MISO_MATH_UTILS2(fmod, fmodf)

// data Endian checker
enum class Endian { Little, Big };
inline Endian get_endian() {
  uint32_t num = 1;
  return (*reinterpret_cast<uint8_t *>(&num) == 1) ? Endian::Little : Endian::Big;
}

// @brief Clear array (zero-fill)
template <typename VectorLike> inline void clear_array(VectorLike &arr) {
  using T = std::decay_t<decltype(*arr.data())>;
  std::fill(arr.data(), arr.data() + arr.size(), T(0));
}

}  // namespace util

}  // namespace miso
