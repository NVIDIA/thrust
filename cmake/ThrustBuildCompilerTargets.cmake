#
# This file defines the `thrust_build_compiler_targets()` function, which
# creates the following interface targets:
#
# thrust.compiler_interface
# - Interface target providing compiler-specific options needed to build
#   Thrust's tests, examples, etc.
#
# thrust.promote_cudafe_warnings
# - Interface target that adds warning promotion for NVCC cudafe invocations.
# - Only exists to work around github issue #1174 on tbb.cuda configurations.
# - May be combined with thrust.compiler_interface when #1174 is fully resolved.

function(thrust_build_compiler_targets)
  set(cxx_compile_definitions)
  set(cxx_compile_options)

  thrust_update_system_found_flags()

  if (THRUST_TBB_FOUND)
    # There's a ton of these in the TBB backend, even though the code is correct.
    # TODO: silence these warnings in code instead
    append_option_if_available("-Wno-unused-parameter" cxx_compile_options)
  endif()

  if ("MSVC" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # TODO Enable /Wall instead of W3
    append_option_if_available("/W3" cxx_compile_options)

    # Treat all warnings as errors:
    append_option_if_available("/WX" cxx_compile_options)

    # Disabled loss-of-data conversion warnings.
    # TODO Re-enable.
    append_option_if_available("/wd4244" cxx_compile_options)
    append_option_if_available("/wd4267" cxx_compile_options)

    # Suppress numeric conversion-to-bool warnings.
    # TODO Re-enable.
    append_option_if_available("/wd4800" cxx_compile_options)

    # Disable warning about applying unary operator- to unsigned type.
    append_option_if_available("/wd4146" cxx_compile_options)

    # MSVC STL assumes that `allocator_traits`'s allocator will use raw pointers,
    # and the `__DECLSPEC_ALLOCATOR` macro causes issues with thrust's universal
    # allocators:
    #   warning C4494: 'std::allocator_traits<_Alloc>::allocate' :
    #      Ignoring __declspec(allocator) because the function return type is not
    #      a pointer or reference
    # See https://github.com/microsoft/STL/issues/696
    append_option_if_available("/wd4494" cxx_compile_options)

    # Some of the async tests require /bigobj to fit all their sections into the
    # object files:
    append_option_if_available("/bigobj" cxx_compile_options)

    # "Oh right, this is Visual Studio."
    list(APPEND cxx_compile_definitions "NOMINMAX")
  else()
    append_option_if_available("-Werror" cxx_compile_options)
    append_option_if_available("-Wall" cxx_compile_options)
    append_option_if_available("-Wextra" cxx_compile_options)
    append_option_if_available("-Winit-self" cxx_compile_options)
    append_option_if_available("-Woverloaded-virtual" cxx_compile_options)
    append_option_if_available("-Wcast-qual" cxx_compile_options)
    append_option_if_available("-Wno-cast-align" cxx_compile_options)
    append_option_if_available("-Wno-long-long" cxx_compile_options)
    append_option_if_available("-Wno-variadic-macros" cxx_compile_options)
    append_option_if_available("-Wno-unused-function" cxx_compile_options)
    append_option_if_available("-Wno-unused-variable" cxx_compile_options)
  endif()

  if ("GNU" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.5)
      # In GCC 4.4, the CUDA backend's kernel launch templates cause
      # impossible-to-decipher "'<anonymous>' is used uninitialized in this
      # function" warnings, so we disable uninitialized variable warnings.
      append_option_if_available("-Wno-uninitialized" cxx_compile_options)
    endif()

    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.5)
      # This isn't available until GCC 4.3, and misfires on TMP code until
      # GCC 4.5.
      append_option_if_available("-Wlogical-op" cxx_compile_options)
    endif()

    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.3)
      # GCC 7.3 complains about name mangling changes due to `noexcept`
      # becoming part of the type system; we don't care.
      append_option_if_available("-Wno-noexcept-type" cxx_compile_options)
    endif()
  endif()

  if (("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}") OR
      ("XL" STREQUAL "${CMAKE_CXX_COMPILER_ID}"))
    # xlC and Clang warn about unused parameters in uninstantiated templates.
    # This causes xlC to choke on the OMP backend, which is mostly #ifdef'd out
    # (and thus has unused parameters) when you aren't using it.
    append_option_if_available("-Wno-unused-parameters" cxx_compile_options)
  endif()

  if ("Clang" STREQUAL "${CMAKE_CXX_COMPILER_ID}")
    # -Wunneeded-internal-declaration misfires in the unit test framework
    # on older versions of Clang.
    append_option_if_available("-Wno-unneeded-internal-declaration" cxx_compile_options)
  endif()

  if ("Feta" STREQUAL "${CMAKE_CUDA_COMPILER_ID}")
    # Today:
    # * NVCC accepts CUDA C++ in .cu files but not .cpp files.
    # * Feta accepts CUDA C++ in .cpp files but not .cu files.
    # TODO: This won't be necessary in the future.
    list(APPEND cxx_compile_options -cppsuffix=cu)
  endif()

  add_library(thrust.compiler_interface INTERFACE)

  foreach (cxx_option IN LISTS cxx_compile_options)
    target_compile_options(thrust.compiler_interface INTERFACE
      $<$<COMPILE_LANGUAGE:CXX>:${cxx_option}>
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Feta>>:${cxx_option}>
      # Only use -Xcompiler with NVCC, not Feta.
      #
      # CMake can't split genexs, so this can't be formatted better :(
      # This is:
      # if (using CUDA and CUDA_COMPILER is NVCC) add -Xcompiler=opt:
      $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler=${cxx_option}>
    )
  endforeach()

  foreach (cxx_definition IN LISTS cxx_compile_definitions)
    # Add these for both CUDA and CXX targets:
    target_compile_definitions(thrust.compiler_interface INTERFACE
      ${cxx_definition}
    )
  endforeach()

  # Display warning numbers from nvcc cudafe errors:
  target_compile_options(thrust.compiler_interface INTERFACE
    # If using CUDA w/ NVCC...
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--display_error_number>
  )

  # This is kept separate for Github issue #1174.
  add_library(thrust.promote_cudafe_warnings INTERFACE)
  target_compile_options(thrust.promote_cudafe_warnings INTERFACE
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcudafe=--promote_warnings>
  )
endfunction()
