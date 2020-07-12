cmake_minimum_required(VERSION 3.10)

project(Evolution)
option(Evolution_BUILD_TEST "Build evolution tests" ON)


find_package(Boost 1.72.0 COMPONENTS graph)
find_package(TBB REQUIRED tbb)
include_directories(${Boost_INCLUDE_DIRS})

add_library(Evolution INTERFACE)
target_include_directories(Evolution INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(Evolution INTERFACE ${Boost_LIBRARIES})
target_link_libraries(Evolution INTERFACE ${TBB_IMPORTED_TARGETS})
target_sources(Evolution
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/Environment.hpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/StateFlow.hpp
)

if (Evolution_BUILD_TEST)
  add_subdirectory(test)
endif ()
