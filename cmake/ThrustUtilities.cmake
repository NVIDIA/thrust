# Given a cu_file (e.g. foo/bar.cu) relative to CMAKE_CURRENT_SOURCE_DIR
# and a thrust_target, create a cpp file that includes the .cu file, and set
# ${cpp_file_var} in the parent scope to the full path of the new file. The new
# file will be generated in:
# ${CMAKE_CURRENT_BINARY_DIR}/<thrust_target_prefix>/${cu_file}.cpp
function(thrust_wrap_cu_in_cpp cpp_file_var cu_file thrust_target)
  thrust_get_target_property(prefix ${thrust_target} PREFIX)
  set(wrapped_source_file "${CMAKE_CURRENT_SOURCE_DIR}/${cu_file}")
  set(cpp_file "${CMAKE_CURRENT_BINARY_DIR}/${prefix}/${cu_file}.cpp")
  configure_file("${Thrust_SOURCE_DIR}/cmake/wrap_source_file.cpp.in" "${cpp_file}")
  set(${cpp_file_var} "${cpp_file}" PARENT_SCOPE)
endfunction()

# Enable or disable RDC for a CUDA target.
# Just using the CMake property won't work for our nvcxx builds, we need
# to manually specify flags.
# nvcc disables RDC by default, while nvc++ enables it. Thus this function
# must be called on all CUDA targets to get consistent RDC state across all
# platforms.
function(thrust_set_rdc_state target_name enable)
  if ("NVCXX" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    if (enable)
      target_compile_options(${target_name} PRIVATE "-gpu=rdc")
    else()
      target_compile_options(${target_name} PRIVATE "-gpu=nordc")
    endif()
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_SEPARABLE_COMPILATION ${enable}
    )
  endif()
endfunction()
