# 00 — Overview: the 10-minute mental model

This project hashes **many independent messages at once** on an NVIDIA GPU, with
two algorithms — **SHA-256** (the SHA-2 family) and **SHA-3** (Keccak) — sharing
one batching design. It is built as **study material**: read the code top to
bottom, compare the readable CPU reference against the GPU kernels, and run the
demo and tests to see everything work and verify.

This doc gives you the whole picture in one sitting. The other docs drill in.

---

## 1. What a cryptographic hash is

A hash takes a message of *any* length and returns a fixed-size **digest**
(SHA-256 → 32 bytes; SHA3-256 → 32 bytes; SHA3-512 → 64 bytes). A good hash is:

- **Deterministic** — same input, same digest, always.
- **One-way** — given a digest, you cannot recover the message.
- **Collision-resistant** — you cannot find two messages with the same digest.
- **Avalanche** — flip one input bit and ≈ half the output bits flip (the demo
  measures ≈ 51.6 % for SHA-256, right on target).

There is **no key** — this is not encryption. (For the keyed/encryption story,
see the sibling `CUDA-AES` project.)

---

## 2. The two algorithms, side by side

| | **SHA-256** (FIPS 180-4) | **SHA-3 / Keccak** (FIPS 202) |
|---|---|---|
| Construction | Merkle–Damgård | Sponge |
| Internal state | 256 bits (8 × 32-bit words) | 1600 bits (25 × 64-bit lanes) |
| Block / "rate" | 512-bit block | rate = 1088 bits (SHA3-256), down to 576 (SHA3-512) |
| Core operation | 64-round compression function | 24-round Keccak-f[1600] permutation |
| Math primitives | 32-bit rotate, shift, AND/XOR, **add** | 64-bit rotate, AND/XOR, **NOT** (no add) |
| Year standardized | 2001 | 2015 |
| Why it exists | the workhorse | a structurally different backup to SHA-2 |

They share **almost nothing** internally, which is exactly why having both in one
study repo is useful — you see two completely different ways to build the same
kind of object. The deep dives are [`01_sha256_internals.md`](01_sha256_internals.md)
and [`02_sha3_keccak_internals.md`](02_sha3_keccak_internals.md).

---

## 3. Why this is a GPU project: parallelize over MESSAGES

A *single* hash is **sequential**: each block/permutation depends on the previous
one, so you cannot cheaply split one long hash across threads. But hashing `N`
**independent** messages is *embarrassingly parallel*. So the unit of work on the
GPU is **one message per thread**:

```
thread 0  ->  hash message 0  ->  digest 0
thread 1  ->  hash message 1  ->  digest 1
...                ...                ...
thread N-1 -> hash message N-1 -> digest N-1
```

No thread ever talks to another. Thousands run at once. This models real
workloads: hashing a table of passwords, deduplicating millions of files,
verifying a batch of records, proof-of-work search. The design is in
[`03_batched_gpu_design.md`](03_batched_gpu_design.md).

---

## 4. The batch layout (the one data structure to learn)

A batch of `n` messages is three host arrays (defined in `include/sha_common.h`):

```c
const uint8_t* messages;   // every message's bytes, concatenated
const size_t*  offsets;    // offsets[i] = where message i starts in `messages`
const size_t*  lengths;    // lengths[i] = byte length of message i
```

Message `i` is the slice `messages[offsets[i] .. offsets[i]+lengths[i])`. Output
is `n` digests packed back to back. This is one big PCIe copy instead of `n`
tiny ones, and it maps perfectly onto one-thread-per-message. The convenience
wrappers `*_batch_fixed()` handle the common equal-length case with no
offsets/lengths bookkeeping (and zero warp divergence).

---

## 5. Codebase map

```
SHA_256/
├── include/
│   ├── sha_common.h          The batch layout + device-info; shared by both
│   ├── sha256.h              SHA-256 public API
│   ├── sha3.h                SHA-3 public API (+ rate/digest helpers)
│   ├── sha_cpu_reference.h   CPU oracle declarations
│   └── sha_cuda.cuh          CUDA-only: CUDA_CHECK + rotate / big-endian helpers
├── src/
│   ├── sha_cpu_reference.cpp  ★ READ FIRST: readable CPU SHA-256 + Keccak
│   ├── sha256_gpu.cu          ★ CUDA SHA-256 (16-word ring schedule)
│   └── sha3_gpu.cu            ★ CUDA Keccak (theta/rho/pi/chi/iota)
├── demo/demo.cpp             A guided, runnable tour (start here when running)
├── tests/test_vectors.cpp    NIST/FIPS known-answer + GPU==CPU verification
├── msvc/                     Hand-written Visual Studio solution (2 projects)
├── scripts/                  One-shot nvcc build (build.bat / build.ps1)
├── CMakeLists.txt            Portable CMake build (Windows + Linux)
├── docs/                     These write-ups
└── CHANGES/                  One markdown file per push, explaining what changed
```

The single most important file to *read* is
[`src/sha_cpu_reference.cpp`](../src/sha_cpu_reference.cpp); the most important to
*understand for GPU lessons* is [`src/sha256_gpu.cu`](../src/sha256_gpu.cu) and
[`docs/04_cuda_performance.md`](04_cuda_performance.md).

---

## 6. How to trust it

`tests/test_vectors.cpp` establishes a three-way agreement:

1. The GPU output matches **published NIST/FIPS digests** (known-answer tests),
   including the awkward padding boundaries.
2. The GPU output matches the **readable CPU reference** on 1000 random,
   variable-length messages per algorithm.

Since NIST == CPU and CPU == GPU, all three agree. On the dev machine the suite
prints **`ALL TESTS PASSED`** for SHA-256 and all four SHA-3 variants. Build and
run instructions: [`05_build_and_run.md`](05_build_and_run.md).

---

## 7. The headline GPU lessons (measured, RTX 2080 SUPER, CUDA 13.3)

| workload | SHA-256 | SHA3-256 | SHA3-512 |
|---|---|---|---|
| 1 M × 64 B, kernel-only | **33.7 GB/s** (526 M msg/s) | 1.18 GB/s | — |
| 65,536 × 8 KiB, kernel-only | 34.2 GB/s | **2.55 GB/s** | 1.41 GB/s |
| 1 M × 64 B, end-to-end (with PCIe) | 3.5 GB/s | 0.87 GB/s | — |

Three lessons fall out, each explored in
[`04_cuda_performance.md`](04_cuda_performance.md):

1. **Registers vs local memory.** SHA-256 keeps its tiny state in registers and
   flies; our *readable* Keccak keeps its 200-byte state in local memory and
   pays for it. This is the opposite emphasis from the AES project (where the
   bottleneck was a data-dependent table lookup).
2. **The sponge rate sets SHA-3's speed.** SHA3-256 (rate 136 B) absorbs nearly
   twice the bytes per Keccak-f permutation that SHA3-512 (rate 72 B) does, so it
   is ≈ 1.8× faster per byte — but only for multi-block messages (workload 2).
3. **The PCIe tax.** For short messages the host↔device copy dominates the
   end-to-end number. Real pipelines keep data resident on the GPU.
