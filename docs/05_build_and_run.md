# 05 — Build and run

Three ways to build, all verified on the dev machine (Windows 11, Visual
Studio 18 with toolset v145, CUDA Toolkit 13.3, RTX 2080 SUPER / `sm_75`). Pick
whichever you like — they compile the same sources into the same two programs:

- **`sha_tests`** — the known-answer + GPU==CPU correctness suite. Prints
  `ALL TESTS PASSED` and returns exit code 0 on success (CI-friendly).
- **`sha_demo`** — the guided six-section tour.

---

## Prerequisites

1. **CUDA Toolkit** (this project: 13.3) with `nvcc` on your `PATH`. Check:
   ```
   nvcc --version
   ```
2. **A host C++ compiler.** On Windows, install **Visual Studio** with the
   *"Desktop development with C++"* workload (this also gives you MSBuild). On
   Linux, GCC/Clang.
3. **An NVIDIA GPU** and a driver new enough for your CUDA version. Check:
   ```
   nvidia-smi
   ```

---

## Option A — Visual Studio (double-click and go)

1. Open `msvc/SHA_CUDA.sln`.
2. Set the configuration to **Release / x64** (top toolbar).
3. Right-click **SHA_Tests → Set as Startup Project**, press **Ctrl+F5** — it
   builds and runs, printing `ALL TESTS PASSED`.
4. Switch the startup project to **SHA_Demo**, press **Ctrl+F5** for the tour.

The executables land in `build_vs\Release\`. The projects are wired for toolset
**v145** + **CUDA 13.3** + **sm_75**; if your setup differs, see *Changing the
toolchain* below.

---

## Option B — one-shot nvcc script

From the project root (the folder with `include/`, `src/`, …):

```bat
scripts\build.bat            REM  (PowerShell:  ./scripts/build.ps1)
build_nvcc\sha_tests.exe     REM  verify against the standard vectors
build_nvcc\sha_demo.exe      REM  run the guided demo
```

The scripts call `nvcc -O2 -std=c++17 -arch=sm_75`. Edit the `ARCH`/`$Arch`
variable at the top for a different GPU.

---

## Option C — CMake (also works on Linux)

```bat
cmake -S . -B build_cmake -G "Visual Studio 18 2026" -A x64
cmake --build build_cmake --config Release
ctest --test-dir build_cmake -C Release
build_cmake\Release\sha_demo.exe
```

On Linux (or with Ninja):
```bash
cmake -S . -B build_cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build_cmake -j
ctest --test-dir build_cmake
./build_cmake/sha_demo
```

CMake auto-detects most of the toolchain. Override the GPU arch with
`-DCMAKE_CUDA_ARCHITECTURES=86` (Ampere), `89` (Ada), `61` (Pascal), etc.

---

## Changing the toolchain for your machine

The defaults assume the dev machine. Two things commonly need changing:

### GPU architecture (compute capability)
Match your card so the GPU actually runs the code:

| GPU family | examples | arch flag |
|---|---|---|
| Pascal | GTX 10-series, Titan X | `sm_61` |
| Turing | RTX 20-series, GTX 16-series | `sm_75` *(default)* |
| Ampere | RTX 30-series, A100 | `sm_86` (A100: `sm_80`) |
| Ada | RTX 40-series | `sm_89` |
| Hopper | H100 | `sm_90` |

- CMake: `-DCMAKE_CUDA_ARCHITECTURES=86`
- nvcc scripts: edit `ARCH=sm_86`
- Visual Studio: in each `.vcxproj`, change `<CodeGeneration>compute_86,sm_86</CodeGeneration>`

### CUDA Toolkit / Visual Studio version (the hand-written `.sln` only)
The `msvc/*.vcxproj` files import `BuildCustomizations\CUDA 13.3.props` /
`.targets` and use `<PlatformToolset>v145</PlatformToolset>`. If you have a
different setup, edit those:

- **CUDA version**: change both `CUDA 13.3.props` and `CUDA 13.3.targets` to your
  installed version (the file must exist under
  `…\MSBuild\Microsoft\VC\<ver>\BuildCustomizations\`).
- **VS toolset**: `v145` = VS 18; use `v143` for VS 2022, `v142` for VS 2019.

If matching the hand-written project to your toolchain is fiddly, **use Option C
(CMake) instead** — it discovers the versions for you and emits a working solution
in `build_cmake/`.

---

## Troubleshooting

| symptom | cause / fix |
|---|---|
| `nvcc: command not found` | CUDA Toolkit not on `PATH`. Re-open the shell after install, or use the "x64 Native Tools Command Prompt". |
| `No CUDA-capable device found` | No GPU / driver too old. `nvidia-smi` should list your card and a CUDA version ≥ your toolkit's. |
| Runs but every digest is wrong / `cudaErrorNoKernelImageForDevice` | Built for the wrong `sm_*`. Set the arch to match your GPU (above). |
| VS: `CUDA 13.3.props not found` | You have a different CUDA version installed; edit the import to match, or use CMake. |
| `out of memory` on a huge batch | The batch buffers must fit in GPU memory. Reduce `n`/`msg_len`, or process in chunks. |

When in doubt, run `sha_tests` first: if it prints `ALL TESTS PASSED`, your build
and GPU are correct and the demo's numbers are trustworthy.
