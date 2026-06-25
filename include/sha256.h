/* ============================================================================
 *  sha256.h  --  Public API for the CUDA batched SHA-256 hasher
 * ============================================================================
 *
 *  WHAT SHA-256 IS, IN ONE PARAGRAPH
 *  ---------------------------------
 *  SHA-256 (FIPS 180-4) reads a message of any length and produces a fixed
 *  32-byte (256-bit) "digest". It works in three steps: PAD the message to a
 *  multiple of 64 bytes (appending a 1 bit, zeros, and the original bit-length),
 *  split it into 64-byte BLOCKS, and run each block through a COMPRESSION
 *  function that stirs the block into a running 256-bit state (eight 32-bit
 *  words H0..H7). The final state, written big-endian, is the digest. There is
 *  no key -- SHA-256 is a one-way hash, not a cipher. The math lives in
 *  src/sha_cpu_reference.cpp (readable) and src/sha256_gpu.cu (the GPU kernel);
 *  docs/01_sha256_internals.md walks through every constant.
 *
 *  WHY IT IS A GREAT *BATCHED* GPU EXERCISE
 *  ----------------------------------------
 *  A single SHA-256 is INHERENTLY SEQUENTIAL: block i+1's compression needs the
 *  state produced by block i, so you cannot parallelize ONE long hash across
 *  threads (cheaply). But hashing N *independent* messages is "embarrassingly
 *  parallel": give each message its own thread and run thousands at once. That
 *  is the design here -- one thread per message -- and it is exactly how real
 *  workloads use the GPU (think: hashing a database of passwords, deduplicating
 *  millions of files, or a proof-of-work search). Contrast this with AES, which
 *  is memory-bound; SHA-256 is COMPUTE-bound (lots of 32-bit integer ops per
 *  byte), which changes which GPU lessons matter -- see docs/04.
 *
 *  See sha_common.h for the batch layout (messages / offsets / lengths) that
 *  every function below takes.
 * ========================================================================== */

#ifndef SHA256_H
#define SHA256_H

#include "sha_common.h"

#ifdef __cplusplus
extern "C" {
#endif

/* SHA-256 always emits 32 bytes (256 bits), and processes the message in
 * 64-byte (512-bit) blocks, regardless of the message length. */
#define SHA256_DIGEST_SIZE 32
#define SHA256_BLOCK_SIZE  64

/* ----------------------------------------------------------------------------
 *  sha256_gpu_batch  --  hash a batch of variable-length messages on the GPU
 * ----------------------------------------------------------------------------
 *  messages : all message bytes, concatenated (see sha_common.h).
 *  offsets  : offsets[i] = start of message i within `messages`.
 *  lengths  : lengths[i] = byte length of message i.
 *  n        : number of messages in the batch.
 *  digests  : OUTPUT buffer, must be at least n * SHA256_DIGEST_SIZE bytes.
 *             Digest i is written to digests + i*SHA256_DIGEST_SIZE.
 *
 *  All pointers are ordinary HOST pointers. This function allocates the device
 *  buffers, copies the inputs across PCIe, launches one thread per message,
 *  copies the digests back, and frees the device memory. It is fully
 *  self-contained: no setup or teardown call is required.
 * -------------------------------------------------------------------------- */
void sha256_gpu_batch(const uint8_t* messages,
                      const size_t*  offsets,
                      const size_t*  lengths,
                      int            n,
                      uint8_t*       digests);

/* ----------------------------------------------------------------------------
 *  sha256_gpu_batch_fixed  --  convenience wrapper for EQUAL-length messages
 * ----------------------------------------------------------------------------
 *  When every message has the same length (a very common micro-benchmark and
 *  real-world case, e.g. fixed-size records), you do not need to build offsets
 *  and lengths arrays. This wrapper treats `messages` as `n` records of exactly
 *  `msg_len` bytes each (so message i is messages + i*msg_len) and writes n
 *  digests. Internally it still uses the one-thread-per-message kernel; it just
 *  spares you the bookkeeping. This is also the ZERO-DIVERGENCE path: every
 *  thread runs the block loop the same number of times.
 * -------------------------------------------------------------------------- */
void sha256_gpu_batch_fixed(const uint8_t* messages,
                            size_t         msg_len,
                            int            n,
                            uint8_t*       digests);

/* ----------------------------------------------------------------------------
 *  sha256_gpu_benchmark  --  measure batched throughput two ways
 * ----------------------------------------------------------------------------
 *  Hashes `n` messages of `msg_len` bytes each, `iters` times, and reports:
 *
 *    *kernel_gbps : throughput of the KERNEL ALONE (inputs already resident on
 *                   the GPU, digests left on the GPU). This is the raw hashing
 *                   speed of the device.
 *    *e2e_gbps    : END-TO-END throughput INCLUDING the host<->device PCIe
 *                   copies -- what a caller of sha256_gpu_batch() actually sees.
 *    *mhs         : millions of MESSAGES hashed per second (kernel-only). For
 *                   short messages this "hashes/sec" number is often the metric
 *                   you care about (e.g. password cracking), more than GB/s.
 *
 *  The gap between kernel_gbps and e2e_gbps is the "PCIe tax" lesson (docs/04):
 *  for short messages the transfer can dominate, which is why real pipelines
 *  keep data resident on the GPU. Throughput is in GB/s with 1 GB = 1e9 bytes.
 * -------------------------------------------------------------------------- */
void sha256_gpu_benchmark(size_t  msg_len,
                          int     n,
                          int     iters,
                          double* kernel_gbps,
                          double* e2e_gbps,
                          double* mhs);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHA256_H */
