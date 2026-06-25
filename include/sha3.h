/* ============================================================================
 *  sha3.h  --  Public API for the CUDA batched SHA-3 (Keccak) hasher
 * ============================================================================
 *
 *  WHAT SHA-3 IS, AND HOW IT DIFFERS FROM SHA-256
 *  ----------------------------------------------
 *  SHA-3 (FIPS 202) is the Keccak sponge construction. It was standardized in
 *  2015 as a structurally DIFFERENT backup to the SHA-2 family (of which SHA-256
 *  is a member) -- if a weakness were ever found in SHA-2's design, SHA-3 is not
 *  built the same way and would likely be unaffected. The two families share
 *  almost nothing internally:
 *
 *    SHA-256 (Merkle-Damgard):  state = 256 bits; absorb 512-bit blocks through
 *                               a compression function; output the whole state.
 *    SHA-3   (sponge):          state = 1600 bits (a 5x5 grid of 64-bit lanes);
 *                               XOR in `rate` bits, stir with the Keccak-f[1600]
 *                               permutation, repeat (ABSORB); then read `rate`
 *                               bits at a time out of the state (SQUEEZE).
 *
 *  THE SPONGE'S "RATE" AND "CAPACITY" -- the one idea to internalize
 *  ----------------------------------------------------------------
 *  The 1600-bit state is split into two parts every round:
 *      rate (r)     : the bits we XOR message into / read output from.
 *      capacity (c) : the bits we NEVER touch directly; they are the security
 *                     reserve. Bigger c = more security but slower (less message
 *                     absorbed per permutation).
 *  FIPS 202 fixes c = 2 * (digest length) for the four SHA-3 variants, so:
 *
 *      variant     digest      capacity c     rate r = 1600 - c     rate bytes
 *      SHA3-224    224 bits    448 bits       1152 bits             144
 *      SHA3-256    256 bits    512 bits       1088 bits             136
 *      SHA3-384    384 bits    768 bits        832 bits             104
 *      SHA3-512    512 bits   1024 bits        576 bits              72
 *
 *  All four use the SAME Keccak-f[1600] permutation (24 rounds of theta, rho, pi,
 *  chi, iota) -- only the rate and the output length change. That is why this one
 *  API serves all four variants from a single kernel: see src/sha3_gpu.cu and
 *  docs/02_sha3_keccak_internals.md.
 *
 *  DOMAIN SEPARATION (why SHA-3 is not raw Keccak)
 *  -----------------------------------------------
 *  Before the sponge's "pad10*1" padding, SHA-3 appends the two bits "01" to the
 *  message. In bytes this means the first padding byte is 0x06 (not 0x01 as in
 *  the original Keccak submission, nor 0x1F as in the SHAKE XOFs). This is what
 *  makes a SHA-3 digest differ from a raw-Keccak digest of the same input, and it
 *  is the single most common place implementations go wrong -- so it is heavily
 *  commented at the point of use.
 *
 *  Batch layout (messages / offsets / lengths) is shared with SHA-256; see
 *  sha_common.h. As with SHA-256, the design is one GPU thread per message.
 * ========================================================================== */

#ifndef SHA3_H
#define SHA3_H

#include "sha_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The Keccak-f[1600] state is 1600 bits = 200 bytes = 25 lanes of 64 bits. */
#define SHA3_STATE_BYTES 200

/* ----------------------------------------------------------------------------
 *  Sha3Variant  --  selects digest length AND (implicitly) the sponge rate
 * ----------------------------------------------------------------------------
 *  The enum value is the digest length in BITS on purpose, so that:
 *      digest bytes = variant / 8
 *      capacity bits = 2 * variant
 *      rate bytes   = (1600 - 2*variant) / 8  =  SHA3_STATE_BYTES - variant/4
 *  The kernel derives both rate and output length from this single value, which
 *  is the cleanest possible expression of "the variant IS the security level".
 * -------------------------------------------------------------------------- */
typedef enum Sha3Variant {
    SHA3_224 = 224,
    SHA3_256 = 256,
    SHA3_384 = 384,
    SHA3_512 = 512
} Sha3Variant;

/* Digest length in bytes for a given variant (e.g. SHA3_256 -> 32). */
static inline int sha3_digest_bytes(Sha3Variant v) { return (int)v / 8; }

/* Sponge rate in bytes for a given variant (e.g. SHA3_256 -> 136).
 * Derivation: rate_bits = 1600 - capacity = 1600 - 2*v; divide by 8 for bytes. */
static inline int sha3_rate_bytes(Sha3Variant v)  { return SHA3_STATE_BYTES - (int)v / 4; }

/* ----------------------------------------------------------------------------
 *  sha3_gpu_batch  --  hash a batch of variable-length messages on the GPU
 * ----------------------------------------------------------------------------
 *  variant  : which SHA-3 (224/256/384/512); sets both rate and output length.
 *  messages : all message bytes, concatenated (see sha_common.h).
 *  offsets  : offsets[i] = start of message i within `messages`.
 *  lengths  : lengths[i] = byte length of message i.
 *  n        : number of messages in the batch.
 *  digests  : OUTPUT, at least n * sha3_digest_bytes(variant) bytes. Digest i is
 *             written to digests + i*sha3_digest_bytes(variant).
 *
 *  Self-contained: allocates device memory, copies in, launches one thread per
 *  message, copies out, frees. All pointers are host pointers.
 * -------------------------------------------------------------------------- */
void sha3_gpu_batch(Sha3Variant    variant,
                    const uint8_t* messages,
                    const size_t*  offsets,
                    const size_t*  lengths,
                    int            n,
                    uint8_t*       digests);

/* ----------------------------------------------------------------------------
 *  sha3_gpu_batch_fixed  --  convenience wrapper for EQUAL-length messages
 * ----------------------------------------------------------------------------
 *  Treats `messages` as `n` records of exactly `msg_len` bytes each (message i
 *  is messages + i*msg_len). The zero-divergence fast path. See the SHA-256
 *  twin in sha256.h for the rationale.
 * -------------------------------------------------------------------------- */
void sha3_gpu_batch_fixed(Sha3Variant    variant,
                          const uint8_t* messages,
                          size_t         msg_len,
                          int            n,
                          uint8_t*       digests);

/* ----------------------------------------------------------------------------
 *  sha3_gpu_benchmark  --  measure batched throughput two ways
 * ----------------------------------------------------------------------------
 *  Same shape as sha256_gpu_benchmark (see sha256.h). The interesting twist for
 *  SHA-3 is that throughput depends on the variant's RATE: SHA3-512 absorbs only
 *  72 bytes per Keccak-f permutation while SHA3-256 absorbs 136, so for long
 *  messages SHA3-256 is nearly twice as fast per byte. The benchmark lets you
 *  watch that rate/speed trade-off directly. Throughput in GB/s (1 GB = 1e9 B).
 * -------------------------------------------------------------------------- */
void sha3_gpu_benchmark(Sha3Variant variant,
                        size_t      msg_len,
                        int         n,
                        int         iters,
                        double*     kernel_gbps,
                        double*     e2e_gbps,
                        double*     mhs);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHA3_H */
