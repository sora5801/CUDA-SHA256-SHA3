@echo off
REM ===========================================================================
REM  build.bat  --  One-shot nvcc build of the demo and the test suite
REM ===========================================================================
REM  The simplest possible build: no CMake, no Visual Studio solution, just
REM  nvcc compiling every source into two executables under build_nvcc\. Run it
REM  from the project ROOT (the folder that contains include\, src\, ...):
REM
REM      scripts\build.bat
REM      build_nvcc\sha_tests.exe     REM verify against the standard vectors
REM      build_nvcc\sha_demo.exe      REM run the guided tour
REM
REM  Requires the CUDA Toolkit (nvcc on PATH) and a host C++ compiler (MSVC).
REM  -arch=sm_75 targets Turing (RTX 20-series). Change it for your GPU, e.g.
REM  sm_86 (Ampere), sm_89 (Ada), sm_61 (Pascal).
REM ===========================================================================
setlocal
set ARCH=sm_75
set OUT=build_nvcc
if not exist %OUT% mkdir %OUT%

set SRC=src\sha256_gpu.cu src\sha3_gpu.cu src\sha_cpu_reference.cpp

echo [1/2] Building sha_tests.exe ...
nvcc -O2 -std=c++17 -arch=%ARCH% -Iinclude %SRC% tests\test_vectors.cpp -o %OUT%\sha_tests.exe
if errorlevel 1 goto :fail

echo [2/2] Building sha_demo.exe ...
nvcc -O2 -std=c++17 -arch=%ARCH% -Iinclude %SRC% demo\demo.cpp -o %OUT%\sha_demo.exe
if errorlevel 1 goto :fail

echo.
echo Build OK. Executables are in %OUT%\
echo   %OUT%\sha_tests.exe
echo   %OUT%\sha_demo.exe
exit /b 0

:fail
echo.
echo BUILD FAILED.
exit /b 1
