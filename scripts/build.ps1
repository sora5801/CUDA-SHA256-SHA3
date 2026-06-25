# ===========================================================================
#  build.ps1  --  One-shot nvcc build of the demo and the test suite
# ===========================================================================
#  PowerShell twin of build.bat. Run it from the project ROOT:
#
#      ./scripts/build.ps1
#      ./build_nvcc/sha_tests.exe     # verify against the standard vectors
#      ./build_nvcc/sha_demo.exe      # run the guided tour
#
#  Requires the CUDA Toolkit (nvcc on PATH) and a host C++ compiler (MSVC).
#  Change $Arch for your GPU: sm_86 (Ampere), sm_89 (Ada), sm_61 (Pascal).
# ===========================================================================
$ErrorActionPreference = "Stop"

$Arch = "sm_75"
$Out  = "build_nvcc"
if (-not (Test-Path $Out)) { New-Item -ItemType Directory -Path $Out | Out-Null }

# Shared sources (two CUDA kernels + the CPU reference). Splatted into nvcc.
$Src = @("src/sha256_gpu.cu", "src/sha3_gpu.cu", "src/sha_cpu_reference.cpp")

Write-Host "[1/2] Building sha_tests.exe ..."
nvcc -O2 -std=c++17 -arch=$Arch -Iinclude @Src tests/test_vectors.cpp -o "$Out/sha_tests.exe"
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED."; exit 1 }

Write-Host "[2/2] Building sha_demo.exe ..."
nvcc -O2 -std=c++17 -arch=$Arch -Iinclude @Src demo/demo.cpp -o "$Out/sha_demo.exe"
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED."; exit 1 }

Write-Host ""
Write-Host "Build OK. Executables are in $Out/"
Write-Host "  $Out/sha_tests.exe"
Write-Host "  $Out/sha_demo.exe"
