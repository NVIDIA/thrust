enable_language(CUDA)

set(THRUST_KNOWN_COMPUTE_ARCHS 35 37 50 52 53 60 61 62 70 72 75 80)

# Split CUDA_FLAGS into 3 parts:
#
# THRUST_CUDA_FLAGS_BASE: Common CUDA flags for all targets.
# THRUST_CUDA_FLAGS_RDC: Additional CUDA flags for targets compiled with RDC.
# THRUST_CUDA_FLAGS_NO_RDC: Additional CUDA flags for targets compiled without RDC.
#
# This is necessary because CUDA SMs 5.3, 6.2, and 7.2 do not support RDC, but
# we want to always build some targets (e.g. testing/cuda/*) with RDC.
# We work around this by building the "always RDC" targets without support for
# those SMs. This requires two sets of CUDA_FLAGS.
#
# Enabling any of those SMs along with the ENABLE_RDC options will result in a
# configuration error.
#
# Because of how CMake handles the CMAKE_CUDA_FLAGS variables, every target
# generated in a given directory will use the same value for CMAKE_CUDA_FLAGS,
# which is determined at the end of the directory's scope. This means caution
# should be used when trying to build different targets with different flags,
# since they might not behave as expected. This will improve with CMake 3.18,
# which add the DEVICE_LINK genex, fixing the issue with using per-target
# CUDA_FLAGS: https://gitlab.kitware.com/cmake/cmake/-/issues/18265
set(THRUST_CUDA_FLAGS_BASE "${CMAKE_CUDA_FLAGS}")
set(THRUST_CUDA_FLAGS_RDC)
set(THRUST_CUDA_FLAGS_NO_RDC)

# Archs that don't support RDC:
set(no_rdc_archs 53 62 72)

# Find the highest arch:
list(SORT THRUST_KNOWN_COMPUTE_ARCHS)
list(LENGTH THRUST_KNOWN_COMPUTE_ARCHS max_idx)
math(EXPR max_idx "${max_idx} - 1")
list(GET THRUST_KNOWN_COMPUTE_ARCHS ${max_idx} highest_arch)

set(option_init OFF)
if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(option_init ON)
endif()
option(THRUST_DISABLE_ARCH_BY_DEFAULT
  "If ON, then all CUDA architectures are disabled on the initial CMake run."
  ${option_init}
)

set(option_init ON)
if (THRUST_DISABLE_ARCH_BY_DEFAULT)
  set(option_init OFF)
endif()

set(num_archs_enabled 0)
foreach (arch IN LISTS THRUST_KNOWN_COMPUTE_ARCHS)
  option(THRUST_ENABLE_COMPUTE_${arch}
    "Enable code generation for tests for sm_${arch}"
    ${option_init}
  )

  if (NOT THRUST_ENABLE_COMPUTE_${arch})
    continue()
  endif()

  math(EXPR num_archs_enabled "${num_archs_enabled} + 1")

  if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    if (NOT ${num_archs_enabled} EQUAL 1)
      message(FATAL_ERROR
        "Feta does not support compilation for multiple device architectures "
        "at once."
      )
    endif()
    set(arch_flag "-gpu=cc${arch}")
  else()
    set(arch_flag "-gencode arch=compute_${arch},code=sm_${arch}")
  endif()

  string(APPEND COMPUTE_MESSAGE " sm_${arch}")
  string(APPEND THRUST_CUDA_FLAGS_NO_RDC " ${arch_flag}")
  if (NOT arch IN_LIST no_rdc_archs)
    string(APPEND THRUST_CUDA_FLAGS_RDC " ${arch_flag}")
  endif()
endforeach()

if (NOT "Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  option(THRUST_ENABLE_COMPUTE_FUTURE
    "Enable code generation for tests for compute_${highest_arch}"
    ${option_init}
  )
  if (THRUST_ENABLE_COMPUTE_FUTURE)
    string(APPEND THRUST_CUDA_FLAGS_BASE
      " -gencode arch=compute_${highest_arch},code=compute_${highest_arch}"
    )
    string(APPEND COMPUTE_MESSAGE " compute_${highest_arch}")
  endif()
endif()

message(STATUS "Thrust: Enabled CUDA architectures:${COMPUTE_MESSAGE}")

# RDC is off by default in NVCC and on by default in Feta. Turning off RDC
# isn't currently supported by Feta. So, we default to RDC off for NVCC and
# RDC on for Feta.
set(option_init OFF)
if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(option_init ON)
endif()

option(THRUST_ENABLE_TESTS_WITH_RDC
  "Build all Thrust tests with RDC; tests that require RDC are not affected by this option."
  ${option_init}
)

option(THRUST_ENABLE_EXAMPLES_WITH_RDC
  "Build all Thrust examples with RDC; examples which require RDC are not affected by this option."
  ${option_init}
)

# Check for RDC/SM compatibility and error/warn if necessary
foreach (sm IN LISTS no_rdc_archs)
  set(sm_opt THRUST_ENABLE_COMPUTE_${sm})
  if (${sm_opt})
    foreach (opt IN ITEMS TESTS EXAMPLES)
      set(rdc_opt THRUST_ENABLE_${opt}_WITH_RDC)
      if (${rdc_opt})
        message(FATAL_ERROR
          "${rdc_opt} is incompatible with ${sm_opt}, since sm_${sm} does not "
          "support RDC."
        )
      endif()
    endforeach()

    message(NOTICE
      "sm_${sm} does not support RDC. Targets that require RDC will be built "
      "without support for this architecture."
    )
  endif()
endforeach()

# By default RDC is not used:
set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_NO_RDC}")
