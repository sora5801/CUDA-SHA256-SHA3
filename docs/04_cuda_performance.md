# 04 — CUDA performance: measured numbers and the lessons

All numbers below are measured on the development machine: **NVIDIA GeForce
RTX 2080 SUPER** (Turing, `sm_75`, 48 SMs, 8 GiB), **CUDA 13.3**, MSVC v145,
Release build. Reproduce them with `sha_demo` (section 6) or the
`*_gpu_benchmark()` API. Your numbers will differ by GPU, but the *ratios* and
the *lessons* hold.

---

## 1. The headline table

| workload | algorithm | kernel-only | end-to-end | throughput |
|---|---|---:|---:|---:|
| **A:** 1,048,576 × 64 B | SHA-256 | **33.7 GB/s** | 3.5 GB/s | 526 M msg/s |
| **A:** 1,048,576 × 64 B | SHA3-256 | 1.18 GB/s | 0.87 GB/s | 18.4 M msg/s |
| **B:** 65,536 × 8 KiB | SHA-256 | 34.2 GB/s | 4.45 GB/s | — |
| **B:** 65,536 × 8 KiB | SHA3-256 | **2.55 GB/s** | 1.76 GB/s | — |
| **B:** 65,536 × 8 KiB | SHA3-512 | 1.41 GB/s | 1.10 GB/s | — |

Three independent lessons come out of this table.

---

## 2. Lesson 1 — registers vs local memory (SHA-256 ≫ Keccak)

SHA-256 runs **~28×** faster than our Keccak on the short workload. The reason is
*where each algorithm's working state lives*, which `ptxas -v` tells us exactly:

```
kernel_sha256_batch_fixed : 64 registers,  64 bytes stack frame, 0 spills
kernel_sha3_batch_fixed   : 128 registers, 200 bytes stack frame, 0 spills
```

- **SHA-256** keeps its eight state words *and* the 16-word message-schedule ring
  entirely in **registers**. Its only stack frame is the 64-byte temporary used to
  build the final padding block. Register access is effectively free, so the
  kernel runs at the speed of the arithmetic.
- **Keccak** has a **200-byte stack frame** — that is the entire `uint64_t st[25]`
  state living in per-thread **local memory** (which physically sits in the L1/L2
  cache hierarchy and global memory, far slower than registers). It *has* to: the
  θ/ρ/π/χ steps index the state with loop-variable indices (`st[j+i]`,
  `st[piln[i]]`), and an array indexed by non-constant values cannot stay in the
  register file. Every one of the 24 rounds reads and writes those 25 lanes
  through local memory, and that traffic dominates the runtime.

This is the **opposite** emphasis from the sibling `CUDA-AES` project, where the
bottleneck was a *data-dependent table lookup* serializing constant memory. Here
the table accesses are uniform (perfect for constant memory); the bottleneck is
the *state* being too big and too dynamically-indexed to keep in registers.

### The fix (left as an exercise)
A fully **unrolled** Keccak-f that holds the 25 lanes in 25 *named* local
variables (`a00, a01, … a44`) — with θ/ρ/π/χ written out as straight-line code
using only compile-time-constant lane references — lets the compiler keep the
whole state in registers. Production GPU Keccak implementations do exactly this
and run many times faster. We deliberately keep the **readable, array-based**
version so it mirrors the CPU reference line for line; the unrolled version is
[`06_study_guide.md`](06_study_guide.md) exercise 4.

A cheaper partial win, also an exercise: absorb full blocks **8 bytes (one lane)
at a time** instead of byte-by-byte, cutting the absorb's local-memory
transactions ~8×.

---

## 3. Lesson 2 — the sponge rate sets SHA-3's speed

Compare SHA3-256 vs SHA3-512 on **workload B** (8 KiB messages, multi-block):

```
SHA3-256 (rate 136 B): 2.55 GB/s
SHA3-512 (rate  72 B): 1.41 GB/s          ratio ≈ 1.81×
```

An 8 KiB message needs `⌈8192/136⌉+1 = 61` Keccak-f permutations under SHA3-256
but `⌈8192/72⌉+1 = 114` under SHA3-512. The permutation is the cost, so
`114/61 ≈ 1.87` predicts the speed ratio — and we measure 1.81×. The small gap is
fixed per-message overhead. **Bigger capacity buys security but costs throughput,
proportional to the rate.**

Crucially this gap is **invisible on workload A** (64-byte messages): 64 < 72 <
136, so *every* variant does exactly **one** permutation per message and they all
run at the same speed. The rate trade-off only appears once messages span multiple
rate blocks. (The demo's section 6 was written in two workloads precisely so this
is shown honestly, not asserted.)

---

## 4. Lesson 3 — the PCIe transfer tax

End-to-end (including host↔device copies) is far below kernel-only for short
messages:

```
SHA-256, workload A:  kernel 33.7 GB/s  →  end-to-end 3.5 GB/s   (~10× drop)
SHA-256, workload B:  kernel 34.2 GB/s  →  end-to-end 4.45 GB/s  (~8× drop)
```

For a fast kernel on small data, **moving the bytes across PCIe costs more than
hashing them**. The fixed cost of launching transfers and the ~12–16 GB/s
practical PCIe bandwidth become the ceiling. This is the same lesson as in the AES
project, and the same fix: **keep data resident on the GPU** across many
operations, and **overlap** copies with compute using CUDA streams
(`cudaMemcpyAsync` + multiple streams) so the next batch copies while the current
one hashes. That overlap is exercise 5 in [`06_study_guide.md`](06_study_guide.md).

Note SHA-3's end-to-end penalty is smaller *in relative terms* only because its
kernel is already slow — the copy is a smaller fraction of a bigger total. That is
not a virtue; speed up the kernel (Lesson 1) and the PCIe tax reappears.

---

## 5. Why constant memory is the right call here

Both algorithms read their constants (`K[t]`; `RC[round]`, `rotc[i]`, `piln[i]`)
at an index that is the **same for every thread in the warp at every step** —
because all threads are on the same round at the same time. Constant memory
**broadcasts** a single value to the whole warp in one transaction in exactly this
case. So constant memory is free here, and no shared-memory staging (the central
trick in the AES project) is needed. `ptxas` confirms only a few hundred bytes of
`cmem` are used. Recognizing *when* constant memory broadcasts vs serializes —
uniform index vs data-dependent index — is one of the most useful CUDA instincts,
and these two projects are the contrast that teaches it.

---

## 6. How to measure it yourself

- **Quick:** run `sha_demo`, section 6.
- **API:** call `sha256_gpu_benchmark()` / `sha3_gpu_benchmark()` with your own
  `msg_len`, `n`, `iters`; they time kernel-only and end-to-end with CUDA events.
- **Registers/local memory:** add `--ptxas-options=-v` to an `nvcc -c` of either
  `.cu` file and read the "registers / stack frame / spills" line.
- **Deep profile:** `nsys profile build_nvcc\sha_demo.exe` (timeline) or
  `ncu --set full build_nvcc\sha_tests.exe` (per-kernel counters: occupancy,
  memory throughput, local-memory traffic — you will see Keccak's local-memory
  load/store dominate). See [`06_study_guide.md`](06_study_guide.md).
