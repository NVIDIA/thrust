enable_language(CUDA)

#
# Architecture options:
#

# Since we have to filter the arch list based on target features, we don't
# currently support the convenience arch flags:
if ("all" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
    "all-major" IN_LIST CMAKE_CUDA_ARCHITECTURES OR
    "native" IN_LIST CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR
    "The Thrust dev build requires an explicit list of architectures in CMAKE_CUDA_ARCHITECTURES. "
    "The convenience flags of 'all', 'all-major', and 'native' are not supported.\n"
    "CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()

# Create a new arch list that only contains arches that support CDP:
set(THRUST_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set(THRUST_CUDA_ARCHITECTURES_RDC ${THRUST_CUDA_ARCHITECTURES})
list(FILTER THRUST_CUDA_ARCHITECTURES_RDC EXCLUDE REGEX "53|62|72|90")

message(STATUS "THRUST_CUDA_ARCHITECTURES:     ${THRUST_CUDA_ARCHITECTURES}")
message(STATUS "THRUST_CUDA_ARCHITECTURES_RDC: ${THRUST_CUDA_ARCHITECTURES_RDC}")

option(THRUST_ENABLE_RDC_TESTS "Enable tests that require separable compilation." ON)
option(THRUST_FORCE_RDC "Enable separable compilation on all targets that support it." OFF)

#
# Clang CUDA options
#
if ("Clang" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-unknown-cuda-version -Xclang=-fcuda-allow-variadic-functions")
endif ()

