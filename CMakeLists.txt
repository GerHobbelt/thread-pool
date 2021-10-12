cmake_minimum_required(VERSION 3.1)
project(threadPool LANGUAGES CXX)

add_library(threadPool INTERFACE)
add_library(threadPool::threadPool ALIAS threadPool)
target_include_directories(threadPool
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if (${CMAKE_VERSION} VERSION_LESS "3.8.0")
  target_compile_features(threadPool INTERFACE cxx_range_for)
else()
  target_compile_features(threadPool INTERFACE cxx_std_17)
endif()
