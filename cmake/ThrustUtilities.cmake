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

# thrust_configure_cuda_target(<target_name> RDC <ON|OFF>)
#
# Configures `target_name` with the appropriate CUDA architectures and RDC state.
function(thrust_configure_cuda_target target_name)
  set(options)
  set(one_value_args RDC)
  set(multi_value_args)
  cmake_parse_arguments(thrust_cuda "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if (thrust_cuda_UNPARSED_ARGUMENTS)
    message(AUTHOR_WARNING
      "Unrecognized arguments passed to thrust_configure_cuda_target: "
      ${thrust_cuda_UNPARSED_ARGUMENTS})
  endif()

  if (NOT DEFINED thrust_cuda_RDC)
    message(AUTHOR_WARNING "RDC option required for thrust_configure_cuda_target.")
  endif()

  if (thrust_cuda_RDC)
    set_target_properties(${target_name} PROPERTIES
      CUDA_ARCHITECTURES "${THRUST_CUDA_ARCHITECTURES_RDC}"
      POSITION_INDEPENDENT_CODE ON
      CUDA_SEPARABLE_COMPILATION ON)
  else()
    set_target_properties(${target_name} PROPERTIES
      CUDA_ARCHITECTURES "${THRUST_CUDA_ARCHITECTURES}"
      CUDA_SEPARABLE_COMPILATION OFF)
  endif()
endfunction()
