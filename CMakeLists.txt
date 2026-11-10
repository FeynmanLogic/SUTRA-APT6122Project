cmake_minimum_required(VERSION 3.20)

project(Sutra LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(GGML_CUDA ON CACHE BOOL "" FORCE)
set(CMAKE_CUDA_ARCHITECTURES 70)

set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)

add_subdirectory(
    external/llama.cpp
    external/llama-build
)
find_package(Threads REQUIRED)

add_executable(
    sutra_benchmark
    benchmark/benchmark_client.cpp
    benchmark/benchmark_runner.cpp
    benchmark/workload_generator.cpp
    worker/worker.cpp
    inference/llama_engine.cpp
)

target_include_directories(
    sutra_benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
    sutra_benchmark
    PRIVATE
    llama
    Threads::Threads
)
