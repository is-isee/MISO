#pragma once
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

#if defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

/// @brief Utility functions
namespace util {

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
template <typename T> inline T pow2(T x) { return x * x; }
template <typename T> inline T pow3(T x) { return x * x * x; }
template <typename T> inline T pow4(T x) { return x * x * x * x; }

// for float min
HOST_DEVICE inline float fmin_safe(float a, float b) { return fminf(a, b); }

// for float max
HOST_DEVICE inline float fmax_safe(float a, float b) { return fmaxf(a, b); }

// for double min
HOST_DEVICE inline double dmin_safe(double a, double b) { return fmin(a, b); }

// for double max
HOST_DEVICE inline double dmax_safe(double a, double b) { return fmax(a, b); }

template <typename T> HOST_DEVICE inline T max2(T a, T b) {
  return fmax_safe(a, b);
}

template <typename T> HOST_DEVICE inline T min2(T a, T b) {
  return fmin_safe(a, b);
}

template <typename T> HOST_DEVICE inline T min3(T a, T b, T c) {
  return fmin_safe(fmin_safe(a, b), c);
}

template <typename T> HOST_DEVICE inline T max3(T a, T b, T c) {
  return fmax_safe(fmax_safe(a, b), c);
}

// data Endian checker
enum class Endian { Little, Big };
inline Endian get_endian() {
  uint32_t num = 1;
  return (*reinterpret_cast<uint8_t *>(&num) == 1) ? Endian::Little : Endian::Big;
}

};  // namespace util
