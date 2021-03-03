cmake_minimum_required(VERSION 3.10)

project(Evolution)

add_subdirectory(utility)
add_subdirectory(evolution)

option(EVOLUTION_TEST "Build Evolution tests" OFF)
if (EVOLUTION_TEST)
  add_subdirectory(evolution_test)
endif ()
