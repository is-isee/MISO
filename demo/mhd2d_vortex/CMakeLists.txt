cmake_minimum_required(VERSION 3.18)

# Define project name by directory
get_filename_component(project_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${project_name})

set(CMAKE_BUILD_TYPE
    Release
    CACHE STRING "Choose a type of build.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_CUDA "Enable CUDA support" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # Select GPU architecture (e.g., L40S/4090: Ada)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(USE_CUDA)
    set(source_suffix "cu")
else()
    set(source_suffix "cpp")
endif()

# App sources
set(app_name ${project_name})
set(app_source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB_RECURSE app_source_files CONFIGURE_DEPENDS
     "${app_source_dir}/*.${source_suffix}")
add_executable(${app_name} ${app_source_files})
target_include_directories(${app_name} PRIVATE ${app_source_dir})
if(USE_CUDA)
    target_compile_definitions(${app_name} PRIVATE USE_CUDA)
endif()

# MPI
find_package(MPI REQUIRED)
target_link_libraries(${app_name} PRIVATE MPI::MPI_CXX)

# MISO library (also fetches dependencies)
include(FetchContent)
set(miso_src ${CMAKE_CURRENT_SOURCE_DIR}/../../core)
# ENABLE_TESTING is passed to MISO by a cache variable (for faster builds)
set(ENABLE_TESTING
    OFF
    CACHE BOOL "Disable tests in MISO")
# Other options (e.g., USE_CUDA) is passed automatically
FetchContent_Declare(MISO SOURCE_DIR ${miso_src})
FetchContent_MakeAvailable(MISO)
target_link_libraries(${app_name} PRIVATE MISO::MISO)

# TODO: SHOULD BE PASSED BY ARGS
set(config_dir ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(${app_name} PRIVATE CONFIG_DIR="${config_dir}/")
