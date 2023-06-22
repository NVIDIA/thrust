file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

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
    thrust_configure_cuda_target(${seq_test_target} RDC OFF)

    if (THRUST_ENABLE_RDC_TESTS)
      thrust_add_test(cdp_test_target ${test_name}.cdp_1 "${test_src}" ${thrust_target})
      thrust_configure_cuda_target(${cdp_test_target} RDC ON)
    endif()
  endforeach()
endforeach()
