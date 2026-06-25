/* ============================================================================
 *  sha_cuda.cuh  --  CUDA-only shared helpers: the error-checking macro and the
 *                    small bit-twiddling primitives both kernels lean on.
 * ============================================================================
 *
 *  This header is included ONLY by the .cu files (sha256_gpu.cu, sha3_gpu.cu).
 *  It is not part of the public API -- application code includes sha256.h /
 *  sha3.h, which expose plain host functions and hide everything in here.
 *
 *  It collects three things that would otherwise be duplicated between the two
 *  kernels:
 *    1. CUDA_CHECK  -- crash loudly with file/line on any CUDA runtime error.
 *    2. rotate primitives -- the rotations are the literal heart of both hashes
 *       (SHA-256 rotates 32-bit words; Keccak rotates 64-bit lanes).
 *    3. big-endian load/store -- SHA-256 reads its message words and writes its
 *       digest in big-endian order, independent of the machine's endianness.
 * ========================================================================== */

#ifndef SHA_CUDA_CUH
#define SHA_CUDA_CUH

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

/* ----------------------------------------------------------------------------
 *  CUDA_CHECK  --  fail loudly and immediately on any CUDA error
 * ----------------------------------------------------------------------------
 *  Silent CUDA failures are the #1 time sink for newcomers: a kernel launch or a
 *  cudaMemcpy fails, nothing is printed, and you stare at wrong output for an
 *  hour. This macro wraps a runtime call; if it did not return cudaSuccess it
 *  prints the file, line, the call text, and the human-readable error, then
 *  aborts. For a study tool, crashing with a clear message beats limping along
 *  with corrupt results. Wrapped in do/while(0) so it behaves like a statement.
 * -------------------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _err = (call);                                              \
        if (_err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s -> %s\n",              \
                         __FILE__, __LINE__, #call,                             \
                         cudaGetErrorString(_err));                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

/* ----------------------------------------------------------------------------
 *  Rotations  --  the core mixing operation of both hash families
 * ----------------------------------------------------------------------------
 *  A "rotate" (a.k.a. circular shift) moves bits that fall off one end back in
 *  the other end, so no information is lost -- unlike a plain shift. SHA-256 and
 *  Keccak are built almost entirely out of rotations + XOR + add, because those
 *  ops are cheap in hardware yet, composed enough times, diffuse every input bit
 *  across the whole state (the "avalanche" property).
 *
 *  CUDA exposes a hardware funnel-shift intrinsic, __funnelshift_r, that does a
 *  32-bit rotate in a single instruction; we use it for the 32-bit SHA-256
 *  rotate. The 64-bit Keccak rotate has no single intrinsic, so we write it as
 *  two shifts + an OR (the compiler maps it to an efficient sequence).
 *
 *  IMPORTANT precondition: these assume 1 <= n <= width-1. A rotate by 0 or by
 *  the full width is undefined via the shift route (shifting a 32/64-bit value
 *  by 32/64 is UB in C++), and neither hash ever needs a zero/full rotate, so we
 *  keep the helpers simple rather than guarding a case that cannot occur.
 * -------------------------------------------------------------------------- */

/* 32-bit rotate-right, used pervasively by SHA-256 (Sigma/sigma functions). */
__device__ __forceinline__ uint32_t rotr32(uint32_t x, uint32_t n) {
    /* __funnelshift_r(lo, hi, n) concatenates hi:lo into 64 bits, shifts right
     * by n, and returns the low 32 bits. With hi == lo == x that is precisely a
     * 32-bit rotate-right by n -- one instruction on the GPU. */
    return __funnelshift_r(x, x, n);
}

/* 64-bit rotate-left, used pervasively by Keccak (the rho step + theta). */
__device__ __forceinline__ uint64_t rotl64(uint64_t x, uint32_t n) {
    return (x << n) | (x >> (64u - n));
}

/* ----------------------------------------------------------------------------
 *  Big-endian load/store of 32-bit words  --  SHA-256's byte order
 * ----------------------------------------------------------------------------
 *  SHA-256 defines its message words and digest as BIG-ENDIAN: the first byte is
 *  the most significant. NVIDIA GPUs are little-endian, so we must assemble the
 *  word explicitly from bytes rather than reinterpreting memory -- doing it
 *  byte-by-byte is also why the message pointer can be UNALIGNED (messages in a
 *  batch start at arbitrary offsets), which a raw 4-byte load could not handle.
 * -------------------------------------------------------------------------- */
__device__ __forceinline__ uint32_t load_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
}

__device__ __forceinline__ void store_be32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v >> 24);
    p[1] = (uint8_t)(v >> 16);
    p[2] = (uint8_t)(v >>  8);
    p[3] = (uint8_t)(v);
}

#endif /* SHA_CUDA_CUH */
