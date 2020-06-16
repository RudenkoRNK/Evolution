add_executable(Test Main.cpp)
target_link_libraries(Test PRIVATE Evolution)

set_property(TARGET Test PROPERTY CXX_STANDARD 20)
