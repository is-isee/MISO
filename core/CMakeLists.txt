cmake_minimum_required(VERSION 3.18)
project(MISO)

set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose a type of build.")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA
option(MISO_USE_CUDA "Enable CUDA support" OFF)
if(MISO_USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Select GPU architecture (e.g., L40S/4090: Ada)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif(MISO_USE_CUDA)

# MISO as a header-only library
add_library(MISO INTERFACE)
add_library(MISO::MISO ALIAS MISO)
target_include_directories(MISO INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA Runtime and flags
if(MISO_USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(MISO INTERFACE CUDA::cudart)
    target_compile_definitions(MISO INTERFACE MISO_USE_CUDA ON)
endif(MISO_USE_CUDA)

# MPI
find_package(MPI REQUIRED)
target_link_libraries(MISO INTERFACE MPI::MPI_CXX)

# yaml-cpp
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found; fetching with FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0)
    FetchContent_MakeAvailable(yaml-cpp)
endif(NOT yaml-cpp_FOUND)
target_link_libraries(MISO INTERFACE yaml-cpp::yaml-cpp)

# Unit tests
option(ENABLE_TESTING "Enable Testing" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif(ENABLE_TESTING)
