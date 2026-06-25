# ============================================================================
#  CMakeLists.txt  --  Portable, cross-platform build for the CUDA SHA project
# ============================================================================
#  CMake has first-class CUDA support, so this is the most portable way to
#  build: it works on Windows (generating a Visual Studio solution OR using
#  Ninja) and on Linux, and it auto-detects most of the toolchain for you.
#
#  TYPICAL USAGE
#  -------------
#      cmake -S . -B build_cmake                  # configure
#      cmake --build build_cmake --config Release # compile
#      ctest --test-dir build_cmake -C Release    # run the known-answer tests
#      build_cmake\Release\sha_demo.exe           # (Windows) run the demo
#      ./build_cmake/sha_demo                      # (Linux)   run the demo
#
#  On Windows this also emits SHA_CUDA.sln inside build_cmake/ that you can open
#  in Visual Studio, as an alternative to the hand-written solution in msvc/.
# ============================================================================

cmake_minimum_required(VERSION 3.18)   # 3.18+ has modern first-class CUDA support

# Declaring CUDA and CXX as project languages makes CMake locate nvcc and the
# host compiler and wire up the link rules automatically.
project(SHA_CUDA LANGUAGES CXX CUDA)

# --- Language standards for the host and device sources ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# --- Choose the GPU architecture(s) to generate code for --------------------
# If the caller did not pass -DCMAKE_CUDA_ARCHITECTURES=..., default to 75
# (Turing / RTX 20-series, the GPU this project was developed and verified on).
# Override for other GPUs, e.g. -DCMAKE_CUDA_ARCHITECTURES=86 for Ampere.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Silence MSVC's "unsafe function" warnings (the demo/tests use std::strlen etc.).
if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Every target needs the public headers in include/.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- The shared SHA sources, compiled once into a static library ------------
# Bundling the CPU reference + both GPU kernels into one library keeps the two
# executables (demo, tests) DRY: they just link this. The .cu files compile with
# nvcc; the .cpp compiles with the host compiler.
add_library(shacuda STATIC
    src/sha_cpu_reference.cpp
    src/sha256_gpu.cu
    src/sha3_gpu.cu
)

# --- The two programs -------------------------------------------------------
add_executable(sha_demo  demo/demo.cpp)
target_link_libraries(sha_demo  PRIVATE shacuda)

add_executable(sha_tests tests/test_vectors.cpp)
target_link_libraries(sha_tests PRIVATE shacuda)

# --- ctest integration so `ctest` runs the known-answer suite ---------------
enable_testing()
add_test(NAME known_answer_tests COMMAND sha_tests)

message(STATUS "SHA_CUDA configured for CUDA arch(s): ${CMAKE_CUDA_ARCHITECTURES}")
