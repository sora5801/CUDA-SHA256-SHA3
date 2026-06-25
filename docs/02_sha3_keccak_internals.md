# 02 — SHA-3 / Keccak internals, from first principles

SHA-3 (FIPS 202) is the **Keccak sponge**. It is structurally unlike SHA-256: no
compression function, no Merkle–Damgård, no addition at all. This doc builds it up
piece by piece. Follow [`src/sha_cpu_reference.cpp`](../src/sha_cpu_reference.cpp)
(Part B) and [`src/sha3_gpu.cu`](../src/sha3_gpu.cu).

---

## 1. The sponge

```
            ┌────────── absorb ──────────┐   ┌──── squeeze ────┐
 0 ─┐
    │  XOR    f    XOR    f    XOR    f          read    f   read
 IV ┼─ m0 ─► [P] ─ m1 ─► [P] ─ m2 ─► [P] ─...─► out0 ─► [P] ─► out1 ...
    │   (only the top `rate` bits are touched; the bottom `capacity` bits never are)
```

A sponge has a big fixed state (here **1600 bits**) split each step into:

- **rate `r`** — the top bits we XOR message into (absorb) and read output from
  (squeeze);
- **capacity `c`** — the bottom bits we **never touch directly**. They are the
  security reserve; their secrecy is what resists attacks.

Between steps we apply a fixed **permutation `f`** (Keccak-f[1600]) that stirs the
whole state. Absorb the message `r` bits at a time, then squeeze the digest `r`
bits at a time. For the four fixed SHA-3 sizes the digest is ≤ `r`, so a single
squeeze read suffices — no extra permutation needed.

---

## 2. Rate vs capacity: the one trade-off

FIPS 202 fixes `c = 2 × (digest length)`:

| variant | digest | capacity `c` | rate `r = 1600 − c` | rate bytes |
|---|---|---|---|---|
| SHA3-224 | 224 b | 448 b | 1152 b | **144** |
| SHA3-256 | 256 b | 512 b | 1088 b | **136** |
| SHA3-384 | 384 b | 768 b |  832 b | **104** |
| SHA3-512 | 512 b | 1024 b | 576 b |  **72** |

Bigger capacity → more security → **smaller rate** → fewer message bytes absorbed
per permutation → **slower**. This is why SHA3-512 is roughly half the throughput
of SHA3-256 on long messages (measured in [`04_cuda_performance.md`](04_cuda_performance.md)).
All four use the **same** Keccak-f[1600]; only `r` and the output length change —
which is exactly why one kernel serves all four (`sha3_gpu_batch(variant, ...)`).

`include/sha3.h` computes both from the variant:
```c
digest_bytes = variant / 8;                  // SHA3_256 -> 32
rate_bytes   = 200 - variant / 4;            // SHA3_256 -> 136
```

---

## 3. The state: a 5 × 5 grid of 64-bit lanes

1600 bits = 25 **lanes** of 64 bits, arranged on a 5×5 grid. We store it flat as
`uint64_t st[25]`, lane `(x, y)` at `st[x + 5*y]`. Bytes map into lanes
**little-endian**, which is why on a little-endian CPU/GPU we can alias the state
as a byte array for absorb/squeeze (NVIDIA GPUs are little-endian, so this is
always valid here).

Vocabulary you'll see in the spec: a **row** is the 5 lanes with fixed `y`; a
**column** is the 5 lanes with fixed `x`; a **lane** is one 64-bit word.

---

## 4. Keccak-f[1600]: 24 rounds of θ ρ π χ ι

Each round applies five steps. The CPU reference spells them out; here is what
each one is *for*:

### θ (theta) — column parity mixing
For each column compute its 5-bit-wise parity, then XOR into every lane a
combination of two neighboring columns' parities (one rotated by 1):
```
C[x]      = st[x,0] ^ st[x,1] ^ st[x,2] ^ st[x,3] ^ st[x,4]   (column parity)
D[x]      = C[x-1] ^ ROTL64(C[x+1], 1)
st[x,y]  ^= D[x]   for all y
```
θ is the main source of **long-range diffusion** — it couples all five columns so
a change anywhere spreads everywhere within a couple of rounds.

### ρ (rho) — intra-lane rotation
Rotate each lane left by a fixed, lane-specific offset (the triangular numbers
`t(t+1)/2 mod 64`). This moves bits **within** each 64-bit lane so that θ's column
mixing doesn't keep bits stuck in the same bit-position.

### π (pi) — lane permutation
Move whole lanes around the grid: `st[y, 2x+3y] ← st[x, y]`. This provides
**inter-lane** diffusion (mixing across the grid), complementing ρ's intra-lane
mixing.

In the code ρ and π are **fused** into one loop over a 24-lane chain, using two
small constant tables: `rotc[i]` (the ρ offset) and `piln[i]` (the π destination).
The chain carries one lane in a temporary `t` while each is rotated and dropped
into its destination. This is the compact tiny_sha3 formulation.

### χ (chi) — the nonlinear step
For each row, the only **non-linear** operation in all of Keccak:
```
st[x] ^= (NOT st[x+1]) AND st[x+2]      (indices mod 5 within the row)
```
Without χ the whole permutation would be linear (just XORs and rotations) and
trivially invertible/breakable. χ is what gives Keccak its cryptographic strength.
Note it uses **AND/NOT**, never addition — Keccak has no adders at all.

### ι (iota) — break symmetry
XOR a per-round constant into lane `(0,0)`:
```
st[0] ^= RC[round]
```
θ ρ π χ are all symmetric under certain rotations of the state; the 24 distinct
round constants (a maximal-length LFSR sequence) break that symmetry so the rounds
aren't all "the same", preventing slide/symmetry attacks.

24 rounds of these five steps = Keccak-f[1600]. The round count gives a large
security margin over the best known attacks.

---

## 5. Domain separation and padding (the #1 bug source)

Raw Keccak and SHA-3 differ by a **domain suffix** appended before padding. SHA-3
appends the two bits `01`, *then* the sponge's `pad10*1` padding. Packed into
bytes (little-endian bit order) this means:

- the first padding byte is **`0x06`** (binary `00000110`: bits `0,1,1` →
  the suffix `01` followed by the first pad `1`);
- the **last byte of the rate block** is OR'd with **`0x80`** (the final pad `1`).

Contrast: the original Keccak submission used `0x01`, and the SHAKE XOFs use
`0x1F`. Using the wrong suffix gives digests that look plausible but match nothing
— a classic silent bug. In the code:

```c
sb[rem]      ^= 0x06;     // domain suffix + first pad bit, right after the message
sb[rate - 1] ^= 0x80;     // final pad bit at the end of the rate block
keccakf(st);
```

Two edge cases fall out automatically because we **XOR**:

- **Message length ≡ rate − 1**: `0x06` and `0x80` land in the *same* byte and
  combine to `0x86`. (Tested: 135-byte input for SHA3-256, rate 136.)
- **Message length is an exact multiple of rate**: `rem == 0`, so this is a full
  *extra* padding-only block. (Tested: 136-byte input for SHA3-256.)

Both 135-, 136-, and 137-byte cases are in `tests/test_vectors.cpp` and pass.

---

## 6. Output

Read the first `digest_bytes` of the state, little-endian (just the byte alias).
For example:

```
SHA3-256("abc") = 3a985da7 4fe225b2 045c172d 6bd390bd
                  855f086e 3e9d525b 46bfe245 11431532
```

matching FIPS 202, as the demo and tests verify for all four variants.

---

## 7. The GPU formulation and its honest cost

The GPU kernel uses the **same** array-based Keccak-f as the CPU reference,
because θ ρ π χ index the state with loop-dependent indices (`st[j+i]`,
`st[piln[i]]`). That addressable indexing forces the 200-byte state into per-thread
**local memory** — it cannot stay purely in registers with this readable
formulation. The constants (`RC`, `rotc`, `piln`) live in **constant memory**,
where they broadcast for free because every thread reads the same index each round.

The consequence (Keccak running several× slower than SHA-256) is real, measured,
and explained in [`04_cuda_performance.md`](04_cuda_performance.md). The faster
alternative — fully unrolling Keccak-f into 25 *named* registers so the state never
leaves the register file — is described there and set as an exercise in
[`06_study_guide.md`](06_study_guide.md). We keep the readable version here on
purpose: clarity first, and the slowdown is itself a GPU lesson.
