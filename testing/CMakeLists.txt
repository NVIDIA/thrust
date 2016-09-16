set(DRIVER "${CMAKE_CURRENT_SOURCE_DIR}/testframework.cpp")

FILE(GLOB SOURCES_CU  *.cu)
FILE(GLOB SOURCES_CPP *.cpp)
set(SOURCES ${SOURCES_CU} ${SOURCES_CPP})

list(FIND SOURCES ${DRIVER} index)
if (${index} EQUAL -1)
  MESSAGE(FATAL_ERROR "${DRIVER} was not found in source list. Something went wrong")
endif()

list(REMOVE_AT SOURCES ${index} SOURCES)

list(LENGTH SOURCES index)
message(STATUS "Found ${index} tests in testing")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
cuda_include_directories(${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(backend)

cuda_add_library(test_driver ${DRIVER} STATIC EXCLUDE_FROM_ALL)

set(targets "")
foreach(src ${SOURCES})
  get_filename_component(exec_name ${src} NAME_WE)
  set(target testing-${exec_name})
  thrust_add_executable(${target} ${src})
  target_link_libraries(${target} test_driver)
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_test(NAME ${target} COMMAND ${target})
  list(APPEND targets ${target})
endforeach()

string(TOLOWER ${DEVICE_BACKEND} backend)
set(targets-backend "")
foreach(src ${SOURCES_BACKEND})
  get_filename_component(exec_name ${src} NAME_WE)
  set(target testing-${backend}-${exec_name})
  thrust_add_executable(${target} ${src})
  target_link_libraries(${target} test_driver)
  set_target_properties(${target} PROPERTIES EXCLUDE_FROM_ALL TRUE)
  add_test(NAME ${target} COMMAND ${target})
  list(APPEND targets-backend ${target})
endforeach()

add_custom_target(testing DEPENDS ${targets} ${targets-backend})
add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
add_dependencies(check testing)

