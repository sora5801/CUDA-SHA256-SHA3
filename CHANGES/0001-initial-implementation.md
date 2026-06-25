# 0001 — Initial implementation

**Date:** 2026-06-24
**Summary:** First public release. A complete, heavily-commented, FIPS-verified
CUDA C++ implementation of **batched, high-throughput SHA-256 and SHA-3** (all
four variants), with a readable CPU reference, a known-answer + cross-check test
suite, a guided demo, three build systems (Visual Studio, nvcc scripts, CMake),
and a full didactic documentation set.

This is the convention for this repo: **every push adds a new `CHANGES/NNNN-*.md`
file** describing what changed and why, so the project's evolution is readable
start to finish.

---

## What was added

### The cryptographic core
- **`src/sha_cpu_reference.cpp` (+ `include/sha_cpu_reference.h`)** — readable,
  single-message CPU implementations of **SHA-256** (full `W[64]` schedule, exactly
  as FIPS 180-4 writes it) and **Keccak-f[1600] / SHA-3** (θ/ρ/π/χ/ι spelled out).
  This is the teaching reference *and* the trusted oracle the GPU is checked
  against.
- **`include/sha_common.h`** — the **batch layout** convention
  (`messages` / `offsets` / `lengths`) shared by both algorithms, plus the
  device-info helper. Read this first to understand the whole API.
- **`include/sha256.h` / `include/sha3.h`** — the public host APIs:
  variable-length batch, fixed-length (zero-divergence) batch, and a benchmark
  entry point. `sha3.h` derives rate and digest length from the variant enum so a
  single kernel serves SHA3-224/256/384/512.
- **`include/sha_cuda.cuh`** — CUDA-only shared helpers: the `CUDA_CHECK` macro,
  the 32-/64-bit rotate primitives (the hardware funnel-shift for SHA-256), and
  the big-endian load/store SHA-256 needs.

### The CUDA implementation
- **`src/sha256_gpu.cu`** — one GPU thread per message; the message schedule kept
  in a **16-word ring buffer** so the whole per-thread state lives in registers
  (64-register kernel, 64-byte stack, zero spills per `ptxas -v`). Round constants
  in **constant memory** (a perfect broadcast fit, unlike AES's data-dependent
  S-box). Also defines the shared `sha_cuda_device_info()`.
- **`src/sha3_gpu.cu`** — one Keccak-f[1600] per thread, all four variants from one
  kernel (rate/output derived from the variant). The readable array-based
  permutation keeps the 200-byte state in local memory — a deliberate
  clarity-over-speed choice, measured and explained in the docs.
- Both expose variable-length, fixed-length, and benchmark host entry points; each
  is self-contained (allocate → copy → launch → copy back → free).

### Verification (`tests/test_vectors.cpp`)
- **SHA-256** known-answer vectors incl. padding edges (55 B single block, 56 B
  two-block, 64 B data+pad).
- **SHA-3** known-answer vectors for **all four variants**, incl. the SHA3-256
  sponge-rate boundaries (135 = rate−1, 136 = full rate, 137 = rate+1).
- **GPU == CPU** on 1000 pseudo-random, **variable-length** messages per algorithm
  in a single batch (exercises the real batch path + warp divergence).
- Non-zero exit code on any failure (CI gate).

### Demo (`demo/demo.cpp`)
A runnable six-part tour: device info; `"abc"` matched to FIPS; batched hashing;
the SHA-3 family (rate/capacity); the avalanche effect; and a two-workload
throughput benchmark (short-record vs multi-block) that teaches the
register-vs-local, sponge-rate, and PCIe-tax lessons honestly.

### Build systems (all three verified on the dev machine)
- **`msvc/SHA_CUDA.sln`** with `SHA_Demo` and `SHA_Tests` projects (PlatformToolset
  `v145`, CUDA 13.3 build customization, `sm_75`).
- **`scripts/build.bat`** and **`scripts/build.ps1`** — one-shot nvcc builds.
- **`CMakeLists.txt`** — portable CMake build with `ctest` integration (also emits
  a VS solution and works on Linux).

### Documentation (`docs/`)
Seven documents: an overview, SHA-256 internals, SHA-3/Keccak internals, the
batched GPU design, the **CUDA performance deep-dive** (with measured numbers and
`ptxas` figures), a build guide, and a study guide with graded exercises. Plus
this `README.md`, `LICENSE` (MIT), `.gitignore`, and `.gitattributes`.

---

## Verified results (RTX 2080 SUPER, CUDA 13.3)

- `sha_tests` → **ALL TESTS PASSED** (SHA-256 + SHA3-224/256/384/512 known-answer
  vectors, padding/rate edge cases, and GPU==CPU on 1000 random variable-length
  messages per algorithm).
- Built and run via all three systems (nvcc script, CMake+ctest, hand-written VS
  solution); all green.
- Throughput (kernel-only):

  | workload | SHA-256 | SHA3-256 | SHA3-512 |
  |----------|---------|----------|----------|
  | 1 M × 64 B   | 33.7 GB/s (526 M msg/s) | 1.18 GB/s | — |
  | 64 K × 8 KiB | 34.2 GB/s | 2.55 GB/s | 1.41 GB/s |

  End-to-end (1 M × 64 B, incl. PCIe): SHA-256 3.5 GB/s, SHA3-256 0.87 GB/s.

---

## Notable engineering decisions / lessons captured
- **One thread per message** is the whole design: a single hash is sequential, but
  independent messages are embarrassingly parallel.
- **Constant memory is the right home for the round constants/tables**, because
  every thread reads the same index each round (broadcast) — the contrast with
  AES's data-dependent S-box (which serialized) is documented.
- **SHA-256 stays in registers; the readable Keccak spills to local memory.** This
  is the dominant performance fork and is measured with `ptxas -v` (64-byte vs
  200-byte stack frame). The register-resident Keccak optimization is written up
  and set as an exercise rather than hidden.
- **The sponge rate/capacity trade-off** is demonstrated empirically (SHA3-256
  ≈ 1.8× SHA3-512 on multi-block messages, matching the permutation-count ratio).

## Known limitations (intentional, for a study project)
- The readable Keccak is not the fastest possible (see docs/04 and the study guide).
- The simple host API copies data in/out every call; high-throughput use should
  keep data resident on the GPU and overlap copies (study guide B4/C1).
- Plain hashes only: no HMAC/KMAC, no side-channel hardening. Not for production.
