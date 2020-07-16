# Create meta targets that build all tests for a single configuration:
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)
  set(config_meta_target ${config_prefix}.tests)
  add_custom_target(${config_meta_target})
  add_dependencies(${config_prefix}.all ${config_meta_target})
endforeach()

# Update flags to reflect RDC options. See note in ThrustCudaConfig.cmake --
# these flag variables behave unintuitively:
if (THRUST_ENABLE_TESTS_WITH_RDC)
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_RDC}")
else()
  set(CMAKE_CUDA_FLAGS "${THRUST_CUDA_FLAGS_BASE} ${THRUST_CUDA_FLAGS_NO_RDC}")
endif()

# Generate testing framework libraries:
add_subdirectory(unittest)

# List of tests that aren't implemented for all backends, but are implemented for CUDA.
set(partially_implemented_CUDA
  async_copy
  async_for_each
  async_reduce
  async_reduce_into
  async_sort
  async_transform
  event
  future
)

# List of tests that aren't implemented for all backends, but are implemented for CPP.
set(partially_implemented_CPP
)

# List of tests that aren't implemented for all backends, but are implemented for TBB.
set(partially_implemented_TBB
)

# List of tests that aren't implemented for all backends, but are implemented for OMP.
set(partially_implemented_OMP
)

# List of all partially implemented tests.
set(partially_implemented
  ${partially_implemented_CUDA}
  ${partially_implemented_CPP}
  ${partially_implemented_TBB}
  ${partially_implemented_OMP}
)
list(REMOVE_DUPLICATES partially_implemented)

## thrust_add_test
#
# Add a test executable and register it with ctest.
#
# target_name_var: Variable name to overwrite with the name of the test
#   target. Useful for post-processing target information per-backend.
# test_name: The name of the test minus "<config_prefix>.test." For example,
#   testing/vector.cu will be "vector", and testing/cuda/copy.cu will be
#   "cuda.copy".
# test_src: The source file that implements the test.
# thrust_target: The reference thrust target with configuration information.
#
function(thrust_add_test target_name_var test_name test_src thrust_target)
  thrust_get_target_property(config_host ${thrust_target} HOST)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Wrap the .cu file in .cpp for non-CUDA backends
  if ("CUDA" STREQUAL "${config_device}")
    set(real_test_src "${test_src}")
  else()
    thrust_wrap_cu_in_cpp(real_test_src "${test_src}" ${thrust_target})
  endif()

  # The actual name of the test's target:
  set(test_target ${config_prefix}.test.${test_name})
  set(${target_name_var} ${test_target} PARENT_SCOPE)

  # Related target names:
  set(config_framework_target ${config_prefix}.test.framework)
  set(config_meta_target ${config_prefix}.tests)
  set(test_meta_target thrust.all.test.${test_name})

  add_executable(${test_target} "${real_test_src}")
  target_link_libraries(${test_target} ${config_framework_target})
  target_include_directories(${test_target} PRIVATE "${Thrust_SOURCE_DIR}/testing")
  thrust_clone_target_properties(${test_target} ${thrust_target})

  # Add to the active configuration's meta target
  add_dependencies(${config_meta_target} ${test_target})

  # Meta target that builds tests with this name for all configurations:
  if (NOT TARGET ${test_meta_target})
    add_custom_target(${test_meta_target})
  endif()
  add_dependencies(${test_meta_target} ${test_target})

  add_test(NAME ${test_target}
    COMMAND "${CMAKE_COMMAND}"
    "-DTHRUST_BINARY=$<TARGET_FILE:${test_target}>"
    "-DTHRUST_SOURCE=${Thrust_SOURCE_DIR}"
    -P "${Thrust_SOURCE_DIR}/cmake/ThrustRunTest.cmake"
  )

  # Run OMP/TBB tests in serial. Multiple OMP processes will massively
  # oversubscribe the machine with GCC's OMP, and we want to test these with
  # the full CPU available to each unit test.
  set(config_systems ${config_host} ${config_device})
  if (("OMP" IN_LIST config_systems) OR ("TBB" IN_LIST config_systems))
    set_tests_properties(${test_target} PROPERTIES RUN_SERIAL ON)
  endif()

  # Check for per-test script. Script will be included in the current scope
  # to allow custom property modifications.
  get_filename_component(test_cmake_script "${test_src}" NAME_WLE)
  set(test_cmake_script "${CMAKE_CURRENT_LIST_DIR}/${test_cmake_script}.cmake")
  if (EXISTS "${test_cmake_script}")
    include("${test_cmake_script}")
  endif()
endfunction()

file(GLOB test_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

# Add common tests to all configs:
foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  foreach(test_src IN LISTS test_srcs)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    if ("${test_name}" IN_LIST partially_implemented)
      # This test is partially implemented on _some_ backends...
      if (NOT "${test_name}" IN_LIST partially_implemented_${config_device})
        # ...but not on the current one.
        continue()
      endif()
    endif()

    thrust_add_test(test_target ${test_name} "${test_src}" ${thrust_target})

    if (THRUST_ENABLE_TESTS_WITH_RDC AND ("CUDA" STREQUAL "${config_device}"))
      thrust_enable_rdc_for_cuda_target(${test_target})
    endif()
  endforeach()
endforeach()

# Add specialized tests:
add_subdirectory(cpp)
add_subdirectory(cuda)
add_subdirectory(omp)
add_subdirectory(regression)
