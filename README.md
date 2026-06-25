# SHA-256 & SHA-3 on CUDA — Batched, High-Throughput Hashing

A heavily-commented, **didactic** implementation of two cryptographic hash
families — **SHA-256** (FIPS 180-4) and **SHA-3 / Keccak** (FIPS 202, all four
variants: SHA3-224/256/384/512) — for NVIDIA GPUs in CUDA C++. It hashes
**thousands of independent messages in a single launch** (one GPU thread per
message), and is built as **study material**: every file is documented line by
line, the algorithms are mirrored between a readable CPU reference and the GPU
kernels, and the GPU output is verified against the official **NIST/FIPS test
vectors**.

> One of a series of CUDA C++ cryptography study projects (see also `CUDA-AES`).
> The goal is not just "SHA on a GPU" but to *teach* both the hashes **and** the
> GPU programming lessons they illustrate: data parallelism, register vs local
> memory, constant-memory broadcast, warp divergence, and the PCIe transfer tax.

---

## Why batched hashing is a great GPU exercise

A *single* hash is **sequential** — each block (SHA-256) or permutation (Keccak)
depends on the previous one, so you cannot cheaply split one hash across threads.
But hashing `N` **independent** messages is *embarrassingly parallel*: give each
message its own thread and run thousands at once.

```
thread i  ──►  hash(message i)  ──►  digest i        (no thread talks to another)
```

"Independent" is the magic word for a GPU. This models real workloads: hashing a
table of passwords, deduplicating millions of files, verifying a batch of records,
proof-of-work search.

---

## What's inside

```
SHA_256/
├── include/                    Public headers
│   ├── sha_common.h            The batch layout (messages/offsets/lengths) + device info
│   ├── sha256.h                SHA-256 host API
│   ├── sha3.h                  SHA-3 host API (+ rate/digest helpers, the variant enum)
│   ├── sha_cpu_reference.h     CPU oracle declarations
│   └── sha_cuda.cuh            CUDA-only: CUDA_CHECK + rotate / big-endian helpers
├── src/
│   ├── sha_cpu_reference.cpp   ★ READ FIRST: readable CPU SHA-256 + Keccak (the oracle)
│   ├── sha256_gpu.cu           ★ CUDA SHA-256 (16-word ring schedule, one thread/message)
│   └── sha3_gpu.cu             ★ CUDA Keccak-f[1600] (θ ρ π χ ι), all four variants
├── demo/demo.cpp               A guided, runnable six-section tour (start here)
├── tests/test_vectors.cpp      FIPS known-answer + GPU==CPU verification
├── msvc/                       Visual Studio solution (SHA_CUDA.sln + 2 projects)
├── scripts/                    One-shot nvcc build (build.bat / build.ps1)
├── CMakeLists.txt              Portable CMake build (Windows + Linux)
├── docs/                       In-depth, didactic write-ups (read these!)
└── CHANGES/                    One markdown file per push, explaining what changed
```

The single most important file to *read* is
[`src/sha_cpu_reference.cpp`](src/sha_cpu_reference.cpp). The most important GPU
files are [`src/sha256_gpu.cu`](src/sha256_gpu.cu) and
[`src/sha3_gpu.cu`](src/sha3_gpu.cu), and the key perf doc is
[`docs/04_cuda_performance.md`](docs/04_cuda_performance.md).

---

## Quick start (Windows, the way it was developed)

You need the **CUDA Toolkit** (13.x) and **Visual Studio** (with the "Desktop
development with C++" workload). Three ways to build — pick any one:

### Option A — Visual Studio (double-click and go)
1. Open `msvc/SHA_CUDA.sln`.
2. Set configuration to **Release / x64**.
3. Right-click **SHA_Tests → Set as Startup Project**, press **Ctrl+F5** to verify.
4. Switch the startup project to **SHA_Demo**, press **Ctrl+F5** for the tour.

### Option B — one-shot nvcc script
```bat
scripts\build.bat            REM (or:  ./scripts/build.ps1 )
build_nvcc\sha_tests.exe     REM verify against the FIPS vectors
build_nvcc\sha_demo.exe      REM run the guided demo
```

### Option C — CMake (also works on Linux)
```bat
cmake -S . -B build_cmake -G "Visual Studio 18 2026" -A x64
cmake --build build_cmake --config Release
ctest --test-dir build_cmake -C Release
```

> **GPU architecture.** Everything defaults to `sm_75` (Turing, e.g. RTX 20-series).
> For a different GPU change the arch: `sm_61` Pascal, `sm_86` Ampere, `sm_89` Ada.
> See [`docs/05_build_and_run.md`](docs/05_build_and_run.md).

---

## What the demo shows

Running `sha_demo` walks through six sections:

1. **GPU device** — which card you're on (name, SM count, memory).
2. **`"abc"` matches** — SHA-256 and SHA3-256 of `"abc"`, matched byte-for-byte to
   the published FIPS digests, so you trust the rest.
3. **Batched hashing** — many strings hashed in a single launch, one thread each
   (and two identical inputs producing identical digests — a hash is deterministic).
4. **The SHA-3 family** — the same input under all four variants, illustrating the
   rate / capacity / output-length relationship.
5. **Avalanche** — flip one input bit and watch ≈ half the output bits flip
   (measured ≈ 51.6 %, right on the ideal).
6. **Throughput** — two batched benchmarks teaching the register-vs-local-memory,
   sponge-rate, and PCIe-tax lessons with real numbers.

---

## The headline GPU lessons (measured on an RTX 2080 SUPER, CUDA 13.3)

| workload | SHA-256 | SHA3-256 | SHA3-512 |
|---|---:|---:|---:|
| 1 M × 64 B, kernel-only | **33.7 GB/s** (526 M msg/s) | 1.18 GB/s | — |
| 65,536 × 8 KiB, kernel-only | 34.2 GB/s | **2.55 GB/s** | 1.41 GB/s |
| 1 M × 64 B, end-to-end (w/ PCIe) | 3.5 GB/s | 0.87 GB/s | — |

1. **Registers vs local memory.** SHA-256 keeps its state in registers (64-reg
   kernel, 64-byte stack) and flies. Our *readable* Keccak keeps its 200-byte
   state in local memory (200-byte stack frame) and pays ~28× for it — the
   opposite emphasis from the AES project, and the optimization to chase.
2. **The sponge rate sets SHA-3's speed.** SHA3-256 (rate 136 B) beats SHA3-512
   (rate 72 B) ≈ 1.8× per byte on long messages — but the gap is *invisible* for
   single-block messages (both do one permutation).
3. **The PCIe tax.** For short messages the host↔device copy dominates end-to-end
   throughput; real pipelines keep data resident on the GPU and overlap copies.

The full investigation, with `ptxas -v` register/stack figures and the reasoning,
is in [`docs/04_cuda_performance.md`](docs/04_cuda_performance.md).

---

## Correctness

`sha_tests` checks, and must print `ALL TESTS PASSED`:

- **SHA-256** known-answer vectors (FIPS 180-4 examples + computed), including the
  padding edge cases (55-byte single block, 56-byte two-block, 64-byte data+pad).
- **SHA-3 (all four variants)** known-answer vectors (FIPS 202), including the
  sponge-rate boundaries for SHA3-256 (135 = rate−1, 136 = full rate, 137 = rate+1).
- **GPU == CPU** on 1000 pseudo-random, **variable-length** messages per algorithm,
  hashed in a single batch (this exercises the real batch path and warp divergence).
- Returns a non-zero exit code on any failure (CI-friendly).

Because the GPU agrees with the readable CPU reference, and the CPU reference
agrees with the published FIPS digests, all three can be trusted.

---

## API in 30 seconds

```c
#include "sha256.h"
#include "sha3.h"

// Batch layout: messages concatenated; offsets[i]/lengths[i] describe message i.
const uint8_t* messages = ...;          // e.g. "abchello"
size_t offsets[] = {0, 3};
size_t lengths[] = {3, 5};
uint8_t digests[2 * SHA256_DIGEST_SIZE];

sha256_gpu_batch(messages, offsets, lengths, 2, digests);   // 2 SHA-256 digests

uint8_t d3[2 * 32];
sha3_gpu_batch(SHA3_256, messages, offsets, lengths, 2, d3); // 2 SHA3-256 digests

// Equal-length convenience (zero warp divergence):
sha256_gpu_batch_fixed(records, 64, n, out);                 // n records of 64 B each
```

See [`include/sha256.h`](include/sha256.h) and [`include/sha3.h`](include/sha3.h)
for the full, commented API (including the `*_benchmark` functions).

---

## ⚠️ Security note

This project is for **learning**, not for protecting real secrets:

- It is a plain hash — **no key, no authentication**. For message authentication
  use **HMAC** (SHA-256) or **KMAC** (SHA-3); for passwords use a slow,
  memory-hard KDF (Argon2, scrypt, bcrypt) — *never* a raw fast hash.
- SHA-256 is subject to **length-extension**; SHA-3 is not (see
  [`docs/06_study_guide.md`](docs/06_study_guide.md) A2). Use HMAC to avoid it.
- The implementation is **not hardened against side channels** and the readable
  Keccak is **not** the fastest possible (see docs/04).

For production, use a vetted library (libsodium, OpenSSL, BoringSSL) or your GPU
vendor's crypto offerings.

---

## Documentation map

| Doc | What it covers |
|-----|----------------|
| [00_overview.md](docs/00_overview.md) | The 10-minute mental model of both hashes + the codebase |
| [01_sha256_internals.md](docs/01_sha256_internals.md) | SHA-256 from first principles: padding, schedule, compression, the ring buffer |
| [02_sha3_keccak_internals.md](docs/02_sha3_keccak_internals.md) | The sponge, Keccak-f's θ ρ π χ ι, rate/capacity, the 0x06 domain-separation trap |
| [03_batched_gpu_design.md](docs/03_batched_gpu_design.md) | One-thread-per-message, the batch layout, warp divergence, memory placement |
| [04_cuda_performance.md](docs/04_cuda_performance.md) | ★ Measured numbers; register-vs-local, the sponge rate, the PCIe tax |
| [05_build_and_run.md](docs/05_build_and_run.md) | Toolchain, GPU arch, all three build systems, troubleshooting |
| [06_study_guide.md](docs/06_study_guide.md) | Graded exercises (SHAKE, HMAC, length-extension, register-resident Keccak, streams) |

---

## A note on the changelog convention

Every time something new is pushed to this repository, a new file is added under
[`CHANGES/`](CHANGES/) (e.g. `0001-initial-implementation.md`) explaining what was
added and why, in the same didactic spirit as the rest of the project. Read them
in order to follow the project's evolution.

## License

MIT — see [LICENSE](LICENSE). SHA-256 (FIPS 180-4) and SHA-3/Keccak (FIPS 202) are
public standards; the test vectors are from / computed per those NIST publications.
