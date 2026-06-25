# 01 — SHA-256 internals, from first principles

SHA-256 (FIPS 180-4) maps a message of any length to a 32-byte digest. This doc
walks every piece, in the order the code performs it. Follow along in
[`src/sha_cpu_reference.cpp`](../src/sha_cpu_reference.cpp) (Part A) for the
readable version and [`src/sha256_gpu.cu`](../src/sha256_gpu.cu) for the GPU one.

---

## 1. The big shape: Merkle–Damgård

```
        message ──pad──► [block 0][block 1]...[block k]
                              │       │           │
  H = IV ──► compress ──► H ──► compress ──► H ──► compress ──► H ──► digest
```

A fixed 256-bit **state** `H` (eight 32-bit words `H0..H7`) starts at a constant
initial value, then each 512-bit (64-byte) block is folded in by the
**compression function**. After the last block, `H` *is* the digest (written
big-endian). This is the Merkle–Damgård construction: build a hash of arbitrary
input out of a fixed-size compression function plus padding.

The state being only 256 bits is why SHA-256 is **sequential**: block `i+1`'s
compression needs the `H` produced by block `i`. You cannot parallelize one hash
across blocks — which is precisely why we parallelize across *messages* instead
(see [`03_batched_gpu_design.md`](03_batched_gpu_design.md)).

---

## 2. Words, endianness, and rotation

SHA-256 works on **32-bit words**, big-endian: the first byte of a block is the
most significant byte of word 0. Our helpers `be32()` (CPU) and `load_be32()`
(GPU, in `sha_cuda.cuh`) assemble a word from four bytes explicitly, so the code
is correct regardless of the machine's endianness and works on **unaligned**
pointers (batch messages start at arbitrary offsets).

The workhorse operation is **rotate-right** `ROTR(x,n)` — a circular shift, so no
bits are lost. On the GPU it is a single hardware funnel-shift instruction
(`__funnelshift_r`). SHA-256 is built almost entirely from rotates, shifts, XOR,
AND, and 32-bit **addition** (addition is what makes it nonlinear over GF(2) and
is the one place carries propagate).

---

## 3. The six functions

Named exactly as in FIPS 180-4:

```
Ch (x,y,z) = (x AND y) XOR ((NOT x) AND z)      "choose": x picks y or z, bit by bit
Maj(x,y,z) = (x AND y) XOR (x AND z) XOR (y AND z)  "majority" of the three
Σ0(x) = ROTR(x,2)  XOR ROTR(x,13) XOR ROTR(x,22)    big sigma 0  (round update)
Σ1(x) = ROTR(x,6)  XOR ROTR(x,11) XOR ROTR(x,25)    big sigma 1  (round update)
σ0(x) = ROTR(x,7)  XOR ROTR(x,18) XOR  SHR(x,3)     small sigma 0 (schedule)
σ1(x) = ROTR(x,17) XOR ROTR(x,19) XOR  SHR(x,10)    small sigma 1 (schedule)
```

The specific rotate/shift amounts are part of the standard, chosen so that, after
64 rounds, every input bit influences every output bit (good diffusion / the
avalanche property). `Ch` and `Maj` are the bitwise "logic"; the `Σ`/`σ` mixers
spread bits across the word.

---

## 4. The constants (nothing-up-my-sleeve numbers)

- **Initial state `H0..H7`**: the first 32 bits of the fractional parts of the
  **square roots** of the first 8 primes (2, 3, 5, 7, 11, 13, 17, 19).
- **Round constants `K[0..63]`**: the first 32 bits of the fractional parts of
  the **cube roots** of the first 64 primes.

Why derive them from primes? So everyone can recompute them and see the designers
did **not** secretly pick values hiding a backdoor. These are called
"nothing-up-my-sleeve" numbers. You can verify `K[0] = 0x428a2f98` yourself:
`frac(cbrt(2)) × 2^32 = 0x428a2f98…`.

---

## 5. The message schedule W

Each block expands its 16 input words into **64** schedule words:

```
W[t] = the t-th big-endian word of the block,                  for t = 0..15
W[t] = σ1(W[t-2]) + W[t-7] + σ0(W[t-15]) + W[t-16],            for t = 16..63
```

This recurrence stirs four earlier words into each new one, spreading every input
byte across the whole schedule. The CPU reference stores the full `W[64]` exactly
as written — clearest to read.

### The GPU's 16-word ring buffer (the one real optimization)

`W[64]` is 256 bytes *per thread*; on the GPU that would spill out of registers
into slow "local" memory. But the recurrence only ever looks back **16** words, so
we keep a 16-slot ring `w[16]` and overwrite each slot once we are past it. Taking
indices mod 16 (note `t-16 ≡ t`, `t-15 ≡ t+1`, `t-7 ≡ t+9`, `t-2 ≡ t+14`):

```c
uint32_t s0 = σ0(w[(t+1)  & 15]);
uint32_t s1 = σ1(w[(t+14) & 15]);
wt = w[t & 15] + s0 + w[(t+9) & 15] + s1;   // == W[t]
w[t & 15] = wt;                              // reuse the W[t-16] slot
```

Same numbers as `W[64]`, but 16 registers instead of 64 words of local memory.
This is the main reason the GPU file looks different from the CPU reference while
computing identical results.

---

## 6. The compression function (64 rounds)

Copy the state into eight working registers `a..h`, then for `t = 0..63`:

```
T1 = h + Σ1(e) + Ch(e,f,g) + K[t] + W[t]
T2 = Σ0(a) + Maj(a,b,c)
h=g; g=f; f=e; e=d+T1;
d=c; c=b; b=a; a=T1+T2;
```

Think of it as a shift register: every round, all eight registers shift down by
one, and the two "fresh" values `e` and `a` are produced by mixing in the schedule
word `W[t]`, the round constant `K[t]`, and the logic functions. After 64 rounds,
**feed-forward** the result into the state:

```
H0 += a;  H1 += b;  ...  H7 += h;
```

That add-back (the Davies–Meyer feed-forward) is what makes the compression
function hard to invert: you would have to "subtract" the rounds *and* know the
pre-round state.

---

## 7. Padding (where off-by-one bugs live)

The message must be padded to a whole number of 64-byte blocks. FIPS 180-4 §5.1.1:

1. Append a single `1` bit → in bytes, append `0x80`.
2. Append `0` bits until the length is ≡ 56 mod 64 bytes.
3. Append the original message length **in bits** as a 64-bit big-endian integer.

So the last 8 bytes of the final block always hold the bit-length. The tricky
case: if the `0x80` lands at byte offset ≥ 56, there is no room for the 8-byte
length, so you emit that block and add **one more** all-zero block carrying just
the length.

The test suite deliberately exercises this:

| input | bytes | blocks | why |
|---|---|---|---|
| `""` | 0 | 1 | pure padding block |
| 55 × `'a'` | 55 | 1 | `0x80` at offset 55, length fits (just barely) |
| 56-byte string | 56 | **2** | `0x80` at offset 56 → no room → second block |
| 64 × `'a'` | 64 | **2** | full data block + a pure padding block |

All pass — see `tests/test_vectors.cpp` and run `sha_tests`.

---

## 8. Output

Serialize `H0..H7` big-endian (most significant byte first) into 32 bytes. Done.

```
SHA-256("abc") = ba7816bf 8f01cfea 414140de 5dae2223
                 b00361a3 96177a9c b410ff61 f20015ad
```

which is exactly what the demo and tests print and check against FIPS 180-4.
