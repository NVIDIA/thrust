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
