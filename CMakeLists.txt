cmake_minimum_required(VERSION 3.16)
project(Sutra VERSION 0.1.0 LANGUAGES CXX)

# This independent scheduler build never downloads dependencies or touches the
# llama.cpp submodule. Dhruv can add an optional backend without coupling policy
# tests to a GPU installation. Fail explicitly until that backend is wired up.
option(SUTRA_ENABLE_LLAMA "Build the llama.cpp worker backend (not integrated yet)" OFF)
if(SUTRA_ENABLE_LLAMA)
    message(FATAL_ERROR "llama.cpp worker integration is pending; build the scheduler with SUTRA_ENABLE_LLAMA=OFF")
endif()

find_package(Threads REQUIRED)
add_library(sutra_scheduler scheduler/scheduler.cpp)
target_include_directories(sutra_scheduler PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(sutra_scheduler PUBLIC cxx_std_17)
set_target_properties(sutra_scheduler PROPERTIES CXX_EXTENSIONS OFF)
target_link_libraries(sutra_scheduler PUBLIC Threads::Threads)
if(MSVC)
    target_compile_options(sutra_scheduler PRIVATE /W4)
else()
    target_compile_options(sutra_scheduler PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(sutra_scheduler_demo examples/scheduler_demo.cpp)
target_link_libraries(sutra_scheduler_demo PRIVATE sutra_scheduler)

include(CTest)
if(BUILD_TESTING)
    add_executable(sutra_scheduler_tests tests/test_scheduler.cpp)
    target_link_libraries(sutra_scheduler_tests PRIVATE sutra_scheduler)
    add_test(NAME scheduler COMMAND sutra_scheduler_tests)
    # Bound a deadlock regression rather than leave CI/ICE hanging forever.
    set_tests_properties(scheduler PROPERTIES TIMEOUT 30)
endif()
