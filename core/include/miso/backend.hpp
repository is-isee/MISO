#pragma once

namespace miso {
namespace backend {

/// @brief Tag type representing CPU (host) execution backend.
/// @details Used for compile-time backend dispatch and specialization.
struct Host {};

/// @brief Tag type representing CUDA (GPU device) execution backend.
/// @details Used for compile-time backend dispatch and specialization.
struct CUDA {};

}  // namespace backend

}  // namespace miso
