# Thrust is a header library; no need to build anything before installing:
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

install(DIRECTORY "${Thrust_SOURCE_DIR}/thrust"
  TYPE INCLUDE
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.inl"
    PATTERN "*.cmake"
    PATTERN "*.md"
)

# Depending on how Thrust is configured, CUB's CMake scripts may or may not be
# included, so maintain a set of CUB install rules in both projects. By default
# CUB headers are installed alongside Thrust -- this may be disabled by turning
# off THRUST_INSTALL_CUB_HEADERS.
option(THRUST_INSTALL_CUB_HEADERS "Include cub headers when installing." ON)
if (THRUST_INSTALL_CUB_HEADERS)
  install(DIRECTORY "${Thrust_SOURCE_DIR}/dependencies/cub/cub"
    TYPE INCLUDE
    FILES_MATCHING
      PATTERN "*.cuh"
      PATTERN "*.cmake"
  )
endif()
