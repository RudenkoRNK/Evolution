add_executable(EvolutionTest Test.cpp)
target_link_libraries(EvolutionTest PRIVATE Evolution)

set_property(TARGET EvolutionTest PROPERTY CXX_STANDARD 20)
