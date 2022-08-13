cmake_minimum_required(VERSION 3.15)
project(BSTest)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

add_executable(BSTest
    BS_thread_pool_test.cpp
)

add_executable(MultiFutureTest
    multi_future_test.cpp
)

add_executable(ContinuationTest
    continuation_test.cpp
)

target_link_libraries(BSTest PRIVATE Threads::Threads)
target_link_libraries(MultiFutureTest PRIVATE Threads::Threads)
target_link_libraries(ContinuationTest PRIVATE Threads::Threads)