# Need to link to libm if it exists:
include(CheckLibraryExists)
check_library_exists(m pow "" THRUST_LIBM_EXISTS)

foreach(thrust_target IN LISTS THRUST_TARGETS)
  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  # Wrap the .cu file in .cpp for non-CUDA backends
  set(bench_src bench.cu)
  if ("CUDA" STREQUAL "${config_device}")
    set(real_bench_src "${bench_src}")
  else()
    thrust_wrap_cu_in_cpp(real_bench_src "${bench_src}" ${thrust_target})
  endif()

  set(bench_target ${config_prefix}.bench)
  add_executable(${bench_target} ${real_bench_src})
  target_link_libraries(${bench_target} ${thrust_target})
  thrust_clone_target_properties(${bench_target} ${thrust_target})

  # Link to libm if available:
  if (THRUST_LIBM_EXISTS)
    target_link_libraries(${bench_target} m)
  endif()

  # Meta target that builds bench target for all configurations:
  set(bench_meta_target thrust.all.bench)
  if (NOT TARGET ${bench_meta_target})
    add_custom_target(${bench_meta_target})
  endif()
  add_dependencies(${bench_meta_target} ${bench_target})

  if ("CUDA" STREQUAL "${config_device}" AND
      THRUST_ENABLE_BENCHMARKS_WITH_RDC)
    thrust_enable_rdc_for_cuda_target(${bench_target})
  endif()

  # Add to configuration meta target:
  add_dependencies(${config_prefix}.all ${bench_target})
endforeach()
