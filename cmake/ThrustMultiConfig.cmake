# This file defines thrust_configure_multiconfig(), which sets up and handles
# the MultiConfig options that allow multiple host/device/dialect configurations
# to be generated from a single thrust build.

function(thrust_configure_multiconfig)
  option(THRUST_ENABLE_MULTICONFIG "Enable multiconfig options for coverage testing." OFF)

  # Dialects:
  set(THRUST_CPP_DIALECT_OPTIONS
    11 14 17
    CACHE INTERNAL "C++ dialects supported by Thrust." FORCE
  )

  if (THRUST_ENABLE_MULTICONFIG)
    # Handle dialect options:
    foreach (dialect IN LISTS THRUST_CPP_DIALECT_OPTIONS)
      set(default_value OFF)
      if (dialect EQUAL 14) # Default to just 14 on:
        set(default_value ON)
      endif()
      option(THRUST_MULTICONFIG_ENABLE_DIALECT_CPP${dialect}
        "Generate C++${dialect} build configurations."
        ${default_value}
      )
    endforeach()

    # Supported versions of MSVC do not distinguish between C++11 and C++14.
    # Warn the user that they may be generating a ton of redundant targets.
    if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}" AND
        THRUST_MULTICONFIG_ENABLE_DIALECT_CPP11)
      message(WARNING
        "Supported versions of MSVC (2017+) do not distinguish between C++11 "
        "and C++14. The requested C++11 targets will be built with C++14."
      )
    endif()

    # Systems:
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_CPP "Generate build configurations that use CPP." ON)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA "Generate build configurations that use CUDA." ON)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_OMP "Generate build configurations that use OpenMP." OFF)
    option(THRUST_MULTICONFIG_ENABLE_SYSTEM_TBB "Generate build configurations that use TBB." OFF)

    # CMake added C++17 support for CUDA targets in 3.18:
    if (THRUST_MULTICONFIG_ENABLE_DIALECT_CPP17 AND
        THRUST_MULTICONFIG_ENABLE_SYSTEM_CUDA)
      cmake_minimum_required(VERSION 3.18)
    endif()

    # Workload:
    # - `SMALL`: [3 configs] Minimal coverage and validation of each device system against the `CPP` host.
    # - `MEDIUM`: [6 configs] Cheap extended coverage.
    # - `LARGE`: [8 configs] Expensive extended coverage. Include all useful build configurations.
    # - `FULL`: [12 configs] The complete cross product of all possible build configurations.
    #
    # Config   | Workloads | Value      | Expense   | Note
    # ---------|-----------|------------|-----------|-----------------------------
    # CPP/CUDA | F L M S   | Essential  | Expensive | Validates CUDA against CPP
    # CPP/OMP  | F L M S   | Essential  | Cheap     | Validates OMP against CPP
    # CPP/TBB  | F L M S   | Essential  | Cheap     | Validates TBB against CPP
    # CPP/CPP  | F L M     | Important  | Cheap     | Tests CPP as device
    # OMP/OMP  | F L M     | Important  | Cheap     | Tests OMP as host
    # TBB/TBB  | F L M     | Important  | Cheap     | Tests TBB as host
    # TBB/CUDA | F L       | Important  | Expensive | Validates TBB/CUDA interop
    # OMP/CUDA | F L       | Important  | Expensive | Validates OMP/CUDA interop
    # TBB/OMP  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # OMP/TBB  | F         | Not useful | Cheap     | Mixes CPU-parallel systems
    # TBB/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device
    # OMP/CPP  | F         | Not Useful | Cheap     | Parallel host, serial device

    set(THRUST_MULTICONFIG_WORKLOAD SMALL CACHE STRING
      "Limit host/device configs: SMALL (up to 3 h/d combos per dialect), MEDIUM(6), LARGE(8), FULL(12)"
    )
    set_property(CACHE THRUST_MULTICONFIG_WORKLOAD PROPERTY STRINGS
      SMALL MEDIUM LARGE FULL
    )
    set(THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS
      CPP_OMP CPP_TBB CPP_CUDA
      CACHE INTERNAL "Host/device combos enabled for SMALL workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_SMALL_CONFIGS}
      CPP_CPP TBB_TBB OMP_OMP
      CACHE INTERNAL "Host/device combos enabled for MEDIUM workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_MEDIUM_CONFIGS}
      OMP_CUDA TBB_CUDA
      CACHE INTERNAL "Host/device combos enabled for LARGE workloads." FORCE
    )
    set(THRUST_MULTICONFIG_WORKLOAD_FULL_CONFIGS
      ${THRUST_MULTICONFIG_WORKLOAD_LARGE_CONFIGS}
      OMP_CPP TBB_CPP OMP_TBB  TBB_OMP
      CACHE INTERNAL "Host/device combos enabled for FULL workloads." FORCE
    )

    # Hide the single config options if they exist from a previous run:
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE INTERNAL)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE INTERNAL)
    endif()
    if (DEFINED THRUST_CPP_DIALECT)
      set_property(CACHE THRUST_CPP_DIALECT PROPERTY TYPE INTERNAL)
    endif()

  else() # Single config:
    # Restore system option visibility if these cache options already exist
    # from a previous run.
    if (DEFINED THRUST_HOST_SYSTEM)
      set_property(CACHE THRUST_HOST_SYSTEM PROPERTY TYPE STRING)
      set_property(CACHE THRUST_DEVICE_SYSTEM PROPERTY TYPE STRING)
    endif()

    set(THRUST_CPP_DIALECT 14
      CACHE STRING "The C++ standard to target: ${THRUST_CPP_DIALECT_OPTIONS}"
    )
    set_property(CACHE THRUST_CPP_DIALECT
      PROPERTY STRINGS
      ${THRUST_CPP_DIALECT_OPTIONS}
    )

    # CMake added C++17 support for CUDA targets in 3.18:
    if (THRUST_CPP_DIALECT EQUAL 17 AND
        THRUST_DEVICE_SYSTEM STREQUAL "CUDA")
      cmake_minimum_required(VERSION 3.18)
    endif()
  endif()
endfunction()
