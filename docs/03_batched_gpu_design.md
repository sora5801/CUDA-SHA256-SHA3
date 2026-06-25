# 03 — The batched GPU design: one thread per message

This doc is about the *parallelization*, independent of which hash you run. It
explains the data layout, the thread mapping, warp divergence, and the memory
choices. Code: the kernels and host launchers in
[`src/sha256_gpu.cu`](../src/sha256_gpu.cu) and
[`src/sha3_gpu.cu`](../src/sha3_gpu.cu).

---

## 1. Why "one thread per message" (and not per block)

A hash is internally sequential, so you cannot split *one* hash across threads
without expensive coordination. But messages are independent of each other, so the
clean decomposition is:

> **thread `i` computes the entire hash of message `i`, alone, start to finish.**

No shared state, no synchronization, no atomics. This is the simplest possible
GPU program shape — a pure `map`. It scales to as many messages as you have, and
the GPU hides the per-thread latency by keeping thousands of threads in flight.

The alternative — cooperative hashing of a single huge message across a warp — is
a real technique (tree hashing, or a warp computing one Keccak-f with one lane per
thread) but it is far more complex and only wins when you have *few, enormous*
messages. For "many independent messages", one-thread-per-message is both simplest
and fastest. See [`06_study_guide.md`](06_study_guide.md) for the cooperative
variant as an exercise.

---

## 2. The batch layout, and why it is shaped this way

```c
const uint8_t* messages;   // all message bytes, concatenated
const size_t*  offsets;    // offsets[i] = start of message i
const size_t*  lengths;    // lengths[i] = length of message i
uint8_t*       digests;    // n * DIGEST_SIZE, packed
```

Three design reasons (all in `include/sha_common.h`):

1. **One big PCIe copy, not `n` tiny ones.** The host→device transfer of
   `messages` is a single `cudaMemcpy`. PCIe has high per-transfer overhead;
   batching the bytes is the difference between good and terrible throughput.
2. **Offsets survive the trip; pointers would not.** A host array of `char*`
   pointers is meaningless on the device. Integer offsets are device-independent
   by construction, so the same arrays work on both sides.
3. **It maps directly to the thread index.** Thread `i` reads `offsets[i]` and
   `lengths[i]`, finds its slice, and writes digest `i`. The kernel body is two
   lines (see `kernel_sha256_batch`).

The kernel launch rounds the grid up to cover `n`:
```c
dim3 threads(256);
dim3 grid((n + 255) / 256);
kernel<<<grid, threads>>>(...);   // last block has a tail guard: if (idx >= n) return;
```

---

## 3. Threads, warps, blocks, occupancy

- A **thread** hashes one message.
- 32 threads form a **warp**, the unit that executes in lockstep (SIMT).
- We pick **256 threads per block** (`*_THREADS_PER_BLOCK`): a multiple of 32,
  enough resident warps to hide memory latency, small enough to keep many blocks
  resident per SM (good occupancy). It is a solid default; tuning it is an
  exercise in [`06_study_guide.md`](06_study_guide.md).
- The RTX 2080 SUPER has 48 SMs; with thousands of blocks the scheduler always has
  work to overlap, which is how the GPU stays busy.

---

## 4. Warp divergence: the cost of unequal lengths

Threads in a warp execute in lockstep. If message lengths differ, two threads in
the same warp run the per-message block loop a **different number of times**. The
warp cannot retire until its **slowest** lane finishes, so the faster lanes idle.

```
warp of 4 (illustrative):  msg lens 64, 64, 64, 4096
                           ├─ 3 threads finish after ~2 compressions
                           └─ 1 thread runs ~64 compressions; the warp waits for it
```

Implications:

- **Equal-length batches have zero divergence** — every lane does the same work.
  That is exactly what `*_batch_fixed()` targets, and why the benchmark uses it.
- **Wildly uneven batches waste lanes.** Real-world mitigation: **sort messages by
  length** before batching, so each warp's 32 messages are similar in size. This
  is a classic, high-impact GPU optimization and a great exercise.
- The padding branch (`if (rem >= 56)` in SHA-256, the final-block handling in
  Keccak) is also a small divergence, but it is one extra iteration at most, so it
  is negligible next to length-driven divergence.

The variable-length kernel is correct for *any* lengths — divergence is a
*performance* effect, not a correctness one. The test suite hashes 1000 random
lengths in one batch precisely to exercise this path.

---

## 5. Memory: where everything lives

| data | lives in | why |
|---|---|---|
| round constants `K` / `RC,rotc,piln` | **constant memory** | every thread reads the same index each round → broadcast for free |
| message bytes, offsets, lengths, digests | **global memory** | the bulk data; accessed once-ish per byte |
| SHA-256 state + 16-word ring | **registers** | small and indexed by compile-time constants → stays in registers → fast |
| Keccak 200-byte state | **local memory** | indexed by loop-variable indices → must be addressable → spills (see docs/02 §7, docs/04) |
| SHA-256 final padding block (64 B) | registers/local | tiny, short-lived |

Two takeaways: **constant memory is a perfect fit here** (unlike the AES S-box,
whose *data-dependent* index made constant memory serialize), and **register vs
local memory is the dominant performance fork** between SHA-256 and our readable
Keccak.

---

## 6. Coalescing

When a warp's 32 threads hash 32 *consecutive* equal-length messages, their byte
reads at the same loop step cover neighboring addresses, so the hardware
**coalesces** them into a few wide memory transactions. This is automatic given
the contiguous batch layout and equal lengths — another reason the fixed-length
path is the fast path. For variable lengths the accesses are less regular, but the
kernel is still correct and the per-message compute usually dominates anyway
(hashing is compute-bound, not bandwidth-bound, especially SHA-256).
