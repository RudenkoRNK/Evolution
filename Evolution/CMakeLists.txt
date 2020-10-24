add_library(Evolution INTERFACE)

find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

target_link_libraries(Evolution INTERFACE ${Boost_LIBRARIES})
target_link_libraries(Evolution INTERFACE ${TBB_IMPORTED_TARGETS})
target_link_libraries(Evolution INTERFACE Utility)

target_include_directories(Evolution INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(Evolution INTERFACE ${Boost_INCLUDE_DIRS})

file(GLOB EvolutionSourcesLocal
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/*.hpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/Evolution/*.cpp
)
set(EvolutionSources ${EvolutionSourcesLocal} CACHE INTERNAL "")
