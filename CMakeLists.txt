cmake_minimum_required(VERSION 3.10)

project(Evolution)

add_library(Evolution INTERFACE)

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/Utility)

target_link_libraries(Evolution INTERFACE ${Boost_LIBRARIES})
target_link_libraries(Evolution INTERFACE ${TBB_IMPORTED_TARGETS})
target_link_libraries(Evolution INTERFACE Utility)

target_include_directories(Evolution INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(Evolution INTERFACE ${Boost_INCLUDE_DIRS})

file(GLOB EvolutionSources
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/*.cpp
)

option(Evolution_BUILD_TEST "Build Evolution tests" OFF)
if (Evolution_BUILD_TEST)
  add_subdirectory(test)
endif ()
