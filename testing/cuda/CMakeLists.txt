file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

# These tests always build with RDC, so make sure that the sm_XX flags are
# compatible. See note in ThrustCudaConfig.cmake.
# TODO once we're using CUDA_ARCHITECTURES, we can setup non-rdc fallback
# tests to build for non-rdc arches. But for now, all files in a given directory
# must build with the same `CMAKE_CUDA_FLAGS` due to CMake constraints around
# how CUDA_FLAGS works.
set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_RDC}")

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  if (NOT config_device STREQUAL "CUDA")
    continue()
  endif()

  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    string(PREPEND test_name "cuda.")

    # Create two targets, one with RDC enabled, the other without. This tests
    # both device-side behaviors -- the CDP kernel launch with RDC, and the
    # serial fallback path without RDC.
    thrust_add_test(seq_test_target ${test_name}.cdp_0 "${test_src}" ${thrust_target})

    if (THRUST_ENABLE_TESTS_WITH_RDC)
      thrust_add_test(cdp_test_target ${test_name}.cdp_1 "${test_src}" ${thrust_target})
      thrust_enable_rdc_for_cuda_target(${cdp_test_target})
    endif()
  endforeach()
endforeach()
