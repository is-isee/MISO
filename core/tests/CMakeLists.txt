# tests/CMakeLists.txt

# External libraries: doctest
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12)
FetchContent_MakeAvailable(doctest)

# Uncomment the following line to use doctest_discover_tests
# include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

# Unit tests (serial)
add_subdirectory(serial)

# Unit tests (parallel)
add_subdirectory(parallel)
