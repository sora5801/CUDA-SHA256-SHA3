/* ============================================================================
 *  sha_cpu_reference.h  --  Readable, single-message CPU implementations of
 *                           SHA-256 and SHA-3, used as the TRUSTED ORACLE.
 * ============================================================================
 *
 *  WHY A CPU REFERENCE EXISTS IN A "GPU" PROJECT
 *  ---------------------------------------------
 *  Two reasons, both central to how this study project earns your trust:
 *
 *    1. TEACHING. The CPU code in src/sha_cpu_reference.cpp is written for
 *       maximum READABILITY, not speed: the SHA-256 message schedule uses the
 *       full 64-word array W[64] exactly as FIPS 180-4 writes it, and the Keccak
 *       permutation spells out theta/rho/pi/chi/iota as separate, commented
 *       steps. Read it first; the GPU kernels are the same math, reorganized for
 *       throughput (e.g. a 16-word ring buffer instead of W[64]).
 *
 *    2. VERIFICATION. The test suite (tests/test_vectors.cpp) checks the GPU
 *       output against (a) published NIST/FIPS known-answer vectors AND (b) this
 *       CPU reference on large pseudo-random batches. Because the CPU reference
 *       itself is checked against the published vectors, a three-way agreement
 *       (NIST == CPU == GPU) means all three can be trusted. The CPU code is the
 *       bridge that lets us verify the GPU on inputs NIST never published.
 *
 *  Each function hashes ONE message (host memory in, digest out). They have no
 *  CUDA dependency and compile as plain C++.
 * ========================================================================== */

#ifndef SHA_CPU_REFERENCE_H
#define SHA_CPU_REFERENCE_H

#include <cstdint>
#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------
 *  sha256_cpu  --  reference SHA-256 of one message
 * ----------------------------------------------------------------------------
 *  msg / len : the message bytes and its length.
 *  out       : 32-byte output buffer (the digest), big-endian as FIPS specifies.
 * -------------------------------------------------------------------------- */
void sha256_cpu(const uint8_t* msg, size_t len, uint8_t out[32]);

/* ----------------------------------------------------------------------------
 *  sha3_cpu  --  reference SHA-3 of one message, for any of the four variants
 * ----------------------------------------------------------------------------
 *  rate_bytes : the sponge rate in bytes (136 for SHA3-256, 72 for SHA3-512,
 *               etc.). Passed explicitly so this one function serves all
 *               variants; sha3.h's sha3_rate_bytes() computes it from a variant.
 *  out_bytes  : digest length in bytes (32 for SHA3-256, 64 for SHA3-512, ...).
 *  msg / len  : the message bytes and its length.
 *  out        : output buffer of `out_bytes` bytes.
 *
 *  Note: this is the FIPS-202 fixed-length SHA-3, so it uses the 0x06 domain
 *  suffix. It is NOT the SHAKE extendable-output function (which would use 0x1F
 *  and a caller-chosen output length).
 * -------------------------------------------------------------------------- */
void sha3_cpu(int rate_bytes, int out_bytes,
              const uint8_t* msg, size_t len, uint8_t* out);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHA_CPU_REFERENCE_H */
