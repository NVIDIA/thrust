# This test should always use per-thread streams on NVCC.
set_target_properties(${test_target} PROPERTIES
  COMPILE_OPTIONS
    $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--default-stream=per-thread>
)

thrust_fix_clang_nvcc_build_for(${test_target})

# NVC++ does not have an equivalent option, and will always
# use the global stream by default.
if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVHPC")
  set_tests_properties(${test_target} PROPERTIES WILL_FAIL ON)
endif()
