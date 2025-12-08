cmake_minimum_required(VERSION 3.18)
project(MISO)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose a type of build.")

# CUDA
option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Select GPU architecture (e.g., L40S/4090: Ada)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# External libraries
find_package(MPI REQUIRED)
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "yaml-cpp not found; fetching with FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# MISO as a header-only library
add_library(MISO INTERFACE)
add_library(MISO::MISO ALIAS MISO)
target_include_directories(MISO INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(MISO INTERFACE MPI::MPI_CXX yaml-cpp::yaml-cpp)

# Unit tests
option(ENABLE_TESTING "Disable Testing" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
