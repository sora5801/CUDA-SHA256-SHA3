# 06 — Study guide: exercises to cement the ideas

These build directly on the code. They are ordered roughly easy → hard, and each
names the files to touch and how to *verify* you got it right (almost always:
"`sha_tests` still prints `ALL TESTS PASSED`"). The point is to learn by changing
working code, not by reading alone.

---

## A. Cryptography exercises

### A1 — Add SHAKE128 / SHAKE256 (extendable-output functions)
SHAKE is Keccak with a **different domain suffix (`0x1F` instead of `0x06`)** and a
**caller-chosen output length** (it keeps squeezing, applying Keccak-f between
rate-sized reads). Add `shake128(msg, len, out, outlen)` to the CPU reference and
a GPU kernel.
- *Hint:* rate for SHAKE128 is 168 B, SHAKE256 is 136 B; capacity = 256/512 bits.
- *Verify:* `python -c "import hashlib; print(hashlib.shake_128(b'abc').hexdigest(64))"`.
- *Lesson:* domain separation, and the squeeze loop you didn't need for fixed SHA-3.

### A2 — Length-extension: SHA-256 is vulnerable, SHA-3 is not
Given `SHA-256(secret || msg)` and `len(secret)`, you can compute
`SHA-256(secret || msg || pad || extra)` **without knowing the secret** — because
the digest *is* the full internal state. Demonstrate it. Then show the same attack
fails against SHA3-256 (the capacity bits are never exposed).
- *Lesson:* why Merkle–Damgård needs HMAC and the sponge does not; a concrete
  security difference between the two families in this repo.

### A3 — Implement HMAC-SHA256 on the GPU
`HMAC(K,m) = H((K⊕opad) || H((K⊕ipad) || m))`. Batch it: one thread per message,
same key. Verify against `hashlib.new` / RFC 4231 test vectors.

---

## B. GPU performance exercises (the meaty ones)

### B1 — Sort the batch by length to kill divergence
Generate a batch with a wide length distribution. Measure throughput, then sort
messages by length before batching so each warp's 32 messages are similar, and
measure again.
- *Files:* host side only; build offsets/lengths from a sorted order.
- *Verify:* digests unchanged (just reordered); time improved on skewed inputs.
- *Lesson:* warp divergence (docs/03 §4) made concrete and then defeated.

### B2 — Absorb Keccak a lane at a time
The readable Keccak absorbs the message **byte by byte** into local-memory state.
Rewrite the *full-block* absorb to XOR **8 bytes (one 64-bit lane) at a time**
using a little-endian 64-bit load. Keep the partial final block byte-wise.
- *Verify:* `sha_tests` still passes; `ncu` shows fewer local-memory stores.
- *Lesson:* the cost of byte-granular local memory access.

### B3 — ★ Fully unroll Keccak-f into registers (the big one)
Replace `uint64_t st[25]` with 25 named variables `a00..a44` and write θ/ρ/π/χ/ι
as straight-line code with only compile-time-constant lane references, so the
state stays in **registers** instead of the 200-byte stack frame.
- *Verify:* `ptxas -v` shows a much smaller (ideally zero) stack frame and no
  spills; `sha_tests` passes; benchmark jumps several×.
- *Lesson:* the single most important Keccak-on-GPU optimization, and a direct
  payoff of docs/04 Lesson 1. This is what production GPU SHA-3 does.

### B4 — Overlap PCIe copies with compute (CUDA streams)
The simple API copies in, launches, copies out, serially. Split the batch into
chunks and pipeline them across ≥ 2 CUDA streams with `cudaMemcpyAsync`, so chunk
`i+1` copies while chunk `i` hashes.
- *Verify:* end-to-end throughput rises toward the kernel-only number.
- *Lesson:* the PCIe tax (docs/04 Lesson 3) and how real pipelines hide it.

### B5 — Tune threads-per-block and use the occupancy calculator
Sweep `*_THREADS_PER_BLOCK` over {64,128,256,512,1024} and plot throughput. Cross-
check against `cudaOccupancyMaxPotentialBlockSize` and `ncu`'s occupancy.
- *Lesson:* occupancy is not monotonic; registers/local memory cap it.

### B6 — `uint4` vectorized loads for SHA-256
Load each 64-byte block as four 16-byte `uint4` vectors (then byte-swap) instead
of 16 scalar `load_be32`s, to widen the memory transactions.
- *Verify:* digests unchanged; check whether it helps (SHA-256 is compute-bound,
  so it may not — measuring that *is* the lesson).

---

## C. Engineering exercises

### C1 — A resident-data API
Add `*_upload()` / `*_hash_resident()` / `*_download()` so a caller can hash many
batches without re-copying data that's already on the GPU. This is the realistic
high-throughput interface; the current self-contained API trades that for
simplicity.

### C2 — Multi-GPU
Split a giant batch across all visible GPUs (`cudaGetDeviceCount`, a thread or
stream per device) and merge the digests.

### C3 — Wire up `nsys` / `ncu` and write up what you see
Profile `sha_tests`/`sha_demo`. For Keccak, confirm in `ncu` that local-memory
traffic dominates; for SHA-256, confirm it is compute-bound with high occupancy.
Add a short markdown with screenshots/numbers, in the spirit of these docs.

---

## D. Conceptual checks (no code)

1. Why can't you parallelize a *single* SHA-256 across threads, but you *can*
   parallelize a single Keccak-f across 25 threads (one lane each)? What
   synchronization would the latter need?
2. SHA3-512 has rate 72 B and SHA3-256 has 136 B. For a 100-byte message, how many
   Keccak-f permutations does each perform? (Answer: SHA3-256 does **1** — 100 <
   136, so just the final padding block; SHA3-512 does **2** — one full 72-byte
   absorb, then the 28-byte final padding block. Work through why.)
3. Constant memory *broadcasts* for the round constants here but *serialized* for
   the AES S-box in the sibling project. State the one-sentence rule that explains
   both.
4. The avalanche demo reports ≈ 51.6 % of output bits flip for a 1-bit input
   change. Why is ~50 % (not ~100 %) the *ideal*?

---

When you finish an exercise, the gold standard is unchanged: rebuild, run
`sha_tests`, and confirm **`ALL TESTS PASSED`**. Correctness first, speed second.
