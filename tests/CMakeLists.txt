# tests/CMakeLists.txt

# External libraries: doctest
include(FetchContent)
FetchContent_Declare(
    doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.12
)
FetchContent_MakeAvailable(doctest)
include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)

# # Integration tests
# add_subdirectory(integration)

# Unit tests (serial)
add_subdirectory(unit_serial)

# # Unit tests (mpi)
# add_subdirectory(unit_mpi)
