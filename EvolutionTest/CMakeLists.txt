add_executable(EvolutionTest)

find_package(Boost REQUIRED COMPONENTS unit_test_framework)

target_link_libraries(EvolutionTest PRIVATE Boost::unit_test_framework)
target_link_libraries(EvolutionTest PRIVATE Evolution)

target_include_directories(EvolutionTest PRIVATE ${Boost_INCLUDE_DIRS})

file(GLOB EvolutionTestSources
  CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.[hcm]pp
  ${CMAKE_CURRENT_SOURCE_DIR}/../Evolution/include/Evolution/*.[hcm]pp
)
target_sources(EvolutionTest PRIVATE ${EvolutionTestSources})

if (MSVC)
  target_compile_options(EvolutionTest PRIVATE "/bigobj")
endif()
