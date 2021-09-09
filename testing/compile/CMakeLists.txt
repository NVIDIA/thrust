thrust_update_system_found_flags()

set(cmake_opts)
set(extra_cuda_cmake_opts)
set(extra_nvcxx_cmake_opts)

set(cmake_opts
  -D "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  -D "CMAKE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
  -D "Thrust_DIR=${Thrust_SOURCE_DIR}/thrust/cmake"
  -D "CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
)

if (THRUST_CPP_FOUND AND THRUST_CUDA_FOUND)
  # Test that we can use `find_package` on an installed Thrust:
  set(extra_cuda_cmake_opts
    -D "CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
  )
endif()

# Need to pass these when testing NVC++.
if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
  set(extra_nvcxx_cmake_opts
    -D "CMAKE_CUDA_COMPILER_ID=${CMAKE_CUDA_COMPILER_ID}"
    -D "CMAKE_CUDA_COMPILER_FORCED=${CMAKE_CUDA_COMPILER_FORCED}"
  )
endif()

# Add a build-and-test CTest.
# - full_test_name_var will be set to the full name of the test.
# - test_id is used to generate a unique name for this test.
# - subdir is the relative path to the test project directory.
# - Any additional args will be passed to the project configure step.
function(thrust_add_compile_test full_test_name_var test_id subdir)
  set(test_name thrust.test.compile.${subdir}.${test_id})
  set(src_dir "${CMAKE_CURRENT_SOURCE_DIR}/${subdir}")
  set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/${subdir}/${test_id}")
  add_test(NAME ${test_name}
    COMMAND "${CMAKE_CTEST_COMMAND}"
      --build-and-test "${src_dir}" "${build_dir}"
      --build-generator "${CMAKE_GENERATOR}"
      --build-options
        ${cmake_opts}
        ${extra_cuda_cmake_opts}
        ${extra_nvcxx_cmake_opts}
        ${ARGN}
      --test-command "${CMAKE_CTEST_COMMAND}"
  )
  set(${full_test_name_var} ${test_name} PARENT_SCOPE)
endfunction()

# Test that CDP fork/join deprecation warnings are emitted only when expected:
if (THRUST_CPP_FOUND AND THRUST_CUDA_FOUND)
  # Helper to enumerate test cases:
  function(thrust_add_fork_join_cdp_test direct_call exec_space exec_policy rdc ignore_depr emit_warning)

    set(direct "indirect")
    if (direct_call)
      set(direct "direct")
    endif()

    set(warn "nowarn")
    if (emit_warning)
      set(warn "warn")
    endif()

    thrust_add_compile_test(test_name
      "${direct}.${exec_space}.${exec_policy}.rdc_${rdc}.ignoredepr_${ignore_depr}.${warn}"
      fork_join_cdp_deprecation
      "-D" "DIRECT_CALL=${direct_call}"         # YES    | NO
      "-D" "EXEC_SPACE=${exec_space}"           # HOST   | DEVICE
      "-D" "EXEC_POLICY=thrust::${exec_policy}" # device | seq
      "-D" "ENABLE_RDC=${rdc}"                  # YES    | NO
      "-D" "IGNORE_DEPRECATION=${ignore_depr}"  # YES    | NO
    )

    set(prop FAIL_REGULAR_EXPRESSION)
    if (emit_warning)
      set(prop PASS_REGULAR_EXPRESSION)
    endif()

    set_tests_properties(${test_name} PROPERTIES ${prop}
      "Device-side launch of Thrust algorithms using CUDA Dynamic Parallelism is deprecated"
    )
  endfunction()

  #                             direct | launch space | policy | rdc | ignoredepr | warn
  thrust_add_fork_join_cdp_test(YES      HOST           device   NO    NO           NO)
  thrust_add_fork_join_cdp_test(YES      HOST           device   NO    YES          NO)
  thrust_add_fork_join_cdp_test(YES      HOST           device   YES   NO           NO)
  thrust_add_fork_join_cdp_test(YES      HOST           device   YES   YES          NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         device   NO    NO           YES)
  thrust_add_fork_join_cdp_test(YES      DEVICE         device   NO    YES          NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         device   YES   NO           YES)
  thrust_add_fork_join_cdp_test(YES      DEVICE         device   YES   YES          NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         seq      NO    NO           NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         seq      NO    YES          NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         seq      YES   NO           NO)
  thrust_add_fork_join_cdp_test(YES      DEVICE         seq      YES   YES          NO)
  thrust_add_fork_join_cdp_test(NO       HOST           device   NO    NO           NO)
  thrust_add_fork_join_cdp_test(NO       HOST           device   NO    YES          NO)
  thrust_add_fork_join_cdp_test(NO       HOST           device   YES   NO           NO)
  thrust_add_fork_join_cdp_test(NO       HOST           device   YES   YES          NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         device   NO    NO           YES)
  thrust_add_fork_join_cdp_test(NO       DEVICE         device   NO    YES          NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         device   YES   NO           YES)
  thrust_add_fork_join_cdp_test(NO       DEVICE         device   YES   YES          NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         seq      NO    NO           NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         seq      NO    YES          NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         seq      YES   NO           NO)
  thrust_add_fork_join_cdp_test(NO       DEVICE         seq      YES   YES          NO)

endif()
