# Bring in CMAKE_INSTALL_LIBDIR
include(GNUInstallDirs)

# Thrust is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inl"
)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust/cmake/"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/thrust"
  PATTERN *.cmake.in EXCLUDE
)
# Need to configure a file to store the infix specified in
# CMAKE_INSTALL_INCLUDEDIR since it can be defined by the user
set(install_location "${CMAKE_INSTALL_LIBDIR}/cmake/thrust")
configure_file("${Thrust_SOURCE_DIR}/thrust/cmake/thrust-header-search.cmake.in"
  "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  @ONLY)
install(FILES "${Thrust_BINARY_DIR}/thrust/cmake/thrust-header-search.cmake"
  DESTINATION "${install_location}")

# Depending on how Thrust is configured, libcudacxx and CUB's CMake scripts may
# or may not be include()'d, so force include their install rules when requested.
# By default, these projects are installed alongside Thrust. This is controlled by
# THRUST_INSTALL_CUB_HEADERS and THRUST_INSTALL_LIBCUDACXX_HEADERS.
option(THRUST_INSTALL_CUB_HEADERS "Include CUB headers when installing." ON)
if (THRUST_INSTALL_CUB_HEADERS)
  # Use a function to limit scope of the CUB_*_DIR vars:
  function(_thrust_install_cub_headers)
    # Fake these for the logic in CUBInstallRules.cmake:
    set(CUB_SOURCE_DIR "${Thrust_SOURCE_DIR}/dependencies/cub/")
    set(CUB_BINARY_DIR "${Thrust_BINARY_DIR}/cub-config/")
    set(CUB_ENABLE_INSTALL_RULES ON)
    set(CUB_IN_THRUST OFF)
    include("${Thrust_SOURCE_DIR}/dependencies/cub/cmake/CubInstallRules.cmake")
  endfunction()

  _thrust_install_cub_headers()
endif()

option(THRUST_INSTALL_LIBCUDACXX_HEADERS "Include libcudacxx headers when installing." ON)
if (THRUST_INSTALL_LIBCUDACXX_HEADERS)
  # Use a function to limit scope of the libcudacxx_*_DIR vars:
  function(_thrust_install_libcudacxx_headers)
    # Fake these for the logic in libcudacxxInstallRules.cmake:
    set(libcudacxx_SOURCE_DIR "${Thrust_SOURCE_DIR}/dependencies/libcudacxx/")
    set(libcudacxx_BINARY_DIR "${Thrust_BINARY_DIR}/libcudacxx-config/")
    set(libcudacxx_ENABLE_INSTALL_RULES ON)
    include("${Thrust_SOURCE_DIR}/dependencies/libcudacxx/cmake/libcudacxxInstallRules.cmake")
  endfunction()

  _thrust_install_libcudacxx_headers()
endif()
