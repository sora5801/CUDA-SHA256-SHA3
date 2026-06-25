/* ============================================================================
 *  sha_common.h  --  Shared types, the "batch layout" convention, and the
 *                    device-info helper used by BOTH the SHA-256 and the
 *                    SHA-3 halves of this library.
 * ============================================================================
 *
 *  WHY THIS FILE EXISTS
 *  --------------------
 *  This project hashes MANY independent messages in a single GPU launch -- that
 *  is what "batched, high-throughput" means here. SHA-256 and SHA-3 are two very
 *  different algorithms, but they share the exact same *batching* problem: how do
 *  you hand a GPU thousands of variable-length byte strings at once, and get back
 *  one digest per string? This header defines the one data layout both halves use
 *  for that, so the SHA-256 API (sha256.h) and the SHA-3 API (sha3.h) look and
 *  feel identical.
 *
 *  THE BATCH LAYOUT  (read this once and the whole API falls into place)
 *  ---------------------------------------------------------------------
 *  A "batch" of `n` messages is described by THREE host arrays:
 *
 *      const uint8_t* messages;   // every message's bytes, concatenated end to end
 *      const size_t*  offsets;    // offsets[i] = byte index in `messages` where msg i starts
 *      const size_t*  lengths;    // lengths[i] = length in bytes of message i
 *
 *  So message i is exactly the slice  messages[offsets[i] .. offsets[i]+lengths[i]).
 *  The messages do not need to be the same length, and they do not need to be
 *  aligned or padded by the caller -- the GPU kernels do all of the SHA padding
 *  themselves, per message.
 *
 *  Example: to hash {"abc", "hello"} you would build
 *      messages = "abchello"             (8 bytes, no separators needed)
 *      offsets  = { 0, 3 }
 *      lengths  = { 3, 5 }
 *      n        = 2
 *
 *  The output is one contiguous buffer of `n` digests back to back:
 *      uint8_t* digests;   // n * DIGEST_SIZE bytes; digest i is at digests + i*DIGEST_SIZE
 *
 *  WHY THIS LAYOUT (instead of, say, an array of pointers)?
 *  --------------------------------------------------------
 *    * One big `messages` buffer is a SINGLE host->device copy, instead of n tiny
 *      ones. PCIe hates many small transfers; it loves one big one.
 *    * Plain integer offsets/lengths survive the trip to the GPU unchanged. Host
 *      pointers would be meaningless on the device, so a pointer array could not
 *      be copied verbatim -- offsets are device-independent by construction.
 *    * It maps cleanly to "one GPU thread per message": thread i reads offsets[i]
 *      and lengths[i], finds its slice, hashes it, writes digest i. No thread ever
 *      touches another thread's data, so there is zero synchronization. That
 *      independence is the entire reason this problem is a good GPU workload.
 *
 *  A NOTE ON DIVERGENCE (a GPU lesson this layout sets up)
 *  -------------------------------------------------------
 *  Because messages may differ in length, two threads in the same 32-thread warp
 *  may run the per-message block loop a different number of times. The warp can
 *  only retire when its slowest thread finishes, so a batch of wildly uneven
 *  lengths wastes some lanes. For the common case of equal-length messages there
 *  is no divergence at all. docs/03_batched_gpu_design.md explores this in depth,
 *  and sha*_gpu_batch_fixed() below is the zero-divergence fast path.
 * ========================================================================== */

#ifndef SHA_COMMON_H
#define SHA_COMMON_H

#include <cstdint>   /* uint8_t, uint32_t, uint64_t */
#include <cstddef>   /* size_t                       */

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------
 *  sha_cuda_device_info  --  print the active CUDA device (name, SMs, memory)
 * ----------------------------------------------------------------------------
 *  Defined once in src/sha256_gpu.cu (it needs the CUDA runtime, so it lives in
 *  a .cu translation unit). Both the demo and the test harness call it so the
 *  reader can see WHICH GPU produced the throughput numbers below the results.
 *  Returns 1 if a CUDA device was found and selected, 0 otherwise.
 * -------------------------------------------------------------------------- */
int sha_cuda_device_info(void);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* SHA_COMMON_H */
