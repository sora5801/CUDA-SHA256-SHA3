/* ============================================================================
 *  sha256_gpu.cu  --  The CUDA side: BATCHED, high-throughput SHA-256
 * ============================================================================
 *
 *  THE BIG IDEA
 *  ------------
 *  One SHA-256 hash is sequential: block i+1 needs the state from block i, so a
 *  single long hash cannot be split across threads cheaply. But hashing N
 *  INDEPENDENT messages is embarrassingly parallel. So the unit of parallelism
 *  here is the MESSAGE, not the block: thread t hashes message t, start to
 *  finish, all by itself. Thousands of threads -> thousands of hashes in flight.
 *
 *  WHERE THE SHARED DATA LIVES: CONSTANT MEMORY (and why it is a *perfect* fit)
 *  ---------------------------------------------------------------------------
 *  Every thread uses the same 64 round constants K[0..63]. We put them in CUDA
 *  __constant__ memory. Constant memory BROADCASTS at full speed when every
 *  thread in a warp reads the SAME address -- and in SHA-256's round loop, every
 *  thread is on the same round t at the same time, so they all read K[t]
 *  together. This is the ideal case for constant memory. (Contrast the sibling
 *  CUDA-AES project, where the S-box index depends on DATA, so threads read
 *  DIFFERENT addresses and constant memory serializes -- forcing a shared-memory
 *  workaround. SHA-256 needs no such trick: see docs/04.)
 *
 *  THE MESSAGE SCHEDULE: A 16-WORD RING BUFFER (the one real optimization)
 *  ----------------------------------------------------------------------
 *  The readable CPU reference expands each block into a full W[64] array. On the
 *  GPU that would be 256 bytes of per-thread state -- it would spill out of
 *  registers into slow "local" memory. Instead we keep only the last 16 words in
 *  a ring buffer w[16], because the recurrence
 *        W[t] = SSIG1(W[t-2]) + W[t-7] + SSIG0(W[t-15]) + W[t-16]
 *  only ever looks back 16 words. Reusing the slot w[t & 15] (note t-16 == t mod
 *  16) keeps the whole schedule in 16 registers. This is the standard trick and
 *  the main reason the GPU file looks different from the CPU reference while
 *  computing exactly the same numbers.
 * ========================================================================== */

#include "sha256.h"
#include "sha_cuda.cuh"

/* ============================================================================
 *  CONSTANT-MEMORY ROUND CONSTANTS  (the data shared by every thread)
 * ============================================================================
 *  Filled once per launch by cudaMemcpyToSymbol in the host launchers below.
 *  Same numbers as K256 in the CPU reference (cube roots of the first 64 primes).
 * ========================================================================== */
__constant__ uint32_t c_K256[64];

/* Host-side copy of the constants, uploaded to c_K256. Kept here (rather than
 * #included from the CPU file) so this translation unit is self-contained. */
static const uint32_t h_K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* ============================================================================
 *  DEVICE HELPERS  --  the six SHA-256 mixing functions, on the GPU
 * ============================================================================
 *  __forceinline__ so the compiler folds them into the kernel: readable source,
 *  zero call overhead. These mirror the CPU reference exactly. rotr32 is the
 *  hardware funnel-shift rotate from sha_cuda.cuh.
 * ========================================================================== */
__device__ __forceinline__ uint32_t Ch (uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
__device__ __forceinline__ uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
__device__ __forceinline__ uint32_t BSIG0(uint32_t x) { return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22); }
__device__ __forceinline__ uint32_t BSIG1(uint32_t x) { return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25); }
__device__ __forceinline__ uint32_t SSIG0(uint32_t x) { return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3); }
__device__ __forceinline__ uint32_t SSIG1(uint32_t x) { return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10); }

/* ----------------------------------------------------------------------------
 *  dev_sha256_compress  --  fold ONE 64-byte block into the state H[8]
 * ----------------------------------------------------------------------------
 *  `block` points at 64 bytes (anywhere -- possibly unaligned, since batch
 *  messages start at arbitrary offsets; load_be32 reads byte-by-byte so that is
 *  fine). Uses the 16-word ring buffer `w` described in the file header.
 * -------------------------------------------------------------------------- */
__device__ __forceinline__ void dev_sha256_compress(uint32_t H[8], const uint8_t* block) {
    /* Load the 16 input words big-endian into the ring buffer. */
    uint32_t w[16];
    #pragma unroll
    for (int t = 0; t < 16; ++t)
        w[t] = load_be32(block + 4 * t);

    /* Eight working variables, copied from the running state. */
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    /* 64 rounds. For t < 16 the schedule word is just w[t]. For t >= 16 we
     * compute the next schedule word IN PLACE, overwriting the slot we will not
     * need again: with indices taken mod 16, t-16 == t, t-15 == t+1, t-7 == t+9,
     * t-2 == t+14. So w[t&15] (the t-16 term) is updated to become W[t]. */
    #pragma unroll
    for (int t = 0; t < 64; ++t) {
        uint32_t wt;
        if (t < 16) {
            wt = w[t];
        } else {
            uint32_t s0 = SSIG0(w[(t +  1) & 15]);   /* sigma0 of W[t-15] */
            uint32_t s1 = SSIG1(w[(t + 14) & 15]);   /* sigma1 of W[t-2]  */
            wt = w[t & 15] + s0 + w[(t + 9) & 15] + s1;  /* + W[t-16] + W[t-7] */
            w[t & 15] = wt;                          /* store back into the ring */
        }

        /* The standard round update (identical to the CPU reference). */
        uint32_t T1 = h + BSIG1(e) + Ch(e, f, g) + c_K256[t] + wt;
        uint32_t T2 = BSIG0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    /* Feed-forward: add the working variables back into the running state. */
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

/* ----------------------------------------------------------------------------
 *  dev_sha256  --  full SHA-256 of ONE message, run entirely by one thread
 * ----------------------------------------------------------------------------
 *  msg/len : this thread's message slice.   out : its 32-byte digest.
 *  Mirrors sha256_cpu(): compress the complete blocks, then build the padded
 *  final one-or-two blocks in a local buffer. The padding logic is identical to
 *  the CPU reference; see there for the line-by-line reasoning.
 * -------------------------------------------------------------------------- */
__device__ void dev_sha256(const uint8_t* msg, size_t len, uint8_t* out) {
    /* Initial state H0..H7 (square roots of the first 8 primes). Inlined as
     * literals so no constant-memory round-trip is needed to start. */
    uint32_t H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    /* Absorb complete 64-byte blocks directly from global memory. */
    size_t full = len / 64;
    for (size_t i = 0; i < full; ++i)
        dev_sha256_compress(H, msg + i * 64);

    /* Build the final padded block(s) in a local buffer. */
    uint8_t block[64];
    size_t rem = len - full * 64;                  /* 0..63 */
    for (size_t j = 0; j < rem; ++j) block[j] = msg[full * 64 + j];
    block[rem] = 0x80;                             /* append the '1' bit */
    for (size_t j = rem + 1; j < 64; ++j) block[j] = 0;

    unsigned long long bitlen = (unsigned long long)len * 8ull;

    /* No room for the 8-byte length? Finish this block and start a fresh one. */
    if (rem >= 56) {
        dev_sha256_compress(H, block);
        #pragma unroll
        for (int j = 0; j < 64; ++j) block[j] = 0;
    }

    /* 64-bit big-endian length in the last 8 bytes, then the final compression. */
    #pragma unroll
    for (int j = 0; j < 8; ++j)
        block[56 + j] = (uint8_t)(bitlen >> (56 - 8 * j));
    dev_sha256_compress(H, block);

    /* Write the digest big-endian. */
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        store_be32(out + 4 * i, H[i]);
}

/* ============================================================================
 *  KERNELS  --  one thread per message
 * ============================================================================ */

/* VARIABLE-LENGTH batch: thread `idx` hashes the slice
 * messages[offsets[idx] .. offsets[idx]+lengths[idx]) into digests[idx]. */
__global__ void kernel_sha256_batch(const uint8_t* __restrict__ messages,
                                    const size_t*  __restrict__ offsets,
                                    const size_t*  __restrict__ lengths,
                                    int            n,
                                    uint8_t*       __restrict__ digests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;                          /* guard the grid tail */
    dev_sha256(messages + offsets[idx], lengths[idx],
               digests + (size_t)idx * SHA256_DIGEST_SIZE);
}

/* FIXED-LENGTH batch: every message is exactly `msg_len` bytes, so message idx
 * starts at idx*msg_len. No offsets/lengths arrays needed, and every thread runs
 * the block loop the same number of times -> zero warp divergence. */
__global__ void kernel_sha256_batch_fixed(const uint8_t* __restrict__ messages,
                                          size_t         msg_len,
                                          int            n,
                                          uint8_t*       __restrict__ digests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dev_sha256(messages + (size_t)idx * msg_len, msg_len,
               digests + (size_t)idx * SHA256_DIGEST_SIZE);
}

/* ============================================================================
 *  HOST SIDE  --  upload constants, allocate, copy, launch, copy back, free
 * ============================================================================ */

/* 256 threads/block: a multiple of the 32-lane warp, enough resident warps to
 * hide memory latency, small enough to keep occupancy high. docs/04 tunes this. */
#define SHA256_THREADS_PER_BLOCK 256

/* Push the round constants into constant memory. Tiny (256 bytes) and done once
 * per call to keep the API stateless. */
static void upload_constants() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_K256, h_K256, sizeof(h_K256)));
}

extern "C" void sha256_gpu_batch(const uint8_t* messages,
                                 const size_t*  offsets,
                                 const size_t*  lengths,
                                 int            n,
                                 uint8_t*       digests) {
    if (n <= 0) return;
    upload_constants();

    /* Total bytes of message data = the end of the last message. We do not
     * require the messages to be tightly packed, but we must copy enough of the
     * `messages` buffer to cover every slice; the farthest byte any thread reads
     * is max(offsets[i] + lengths[i]). Compute that span. */
    size_t span = 0;
    for (int i = 0; i < n; ++i) {
        size_t end = offsets[i] + lengths[i];
        if (end > span) span = end;
    }

    /* Allocate device buffers: the message bytes, the two index arrays, and the
     * output digests (n * 32 bytes). */
    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    size_t  *d_offsets = nullptr, *d_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, span ? span : 1));
    CUDA_CHECK(cudaMalloc(&d_offsets,  (size_t)n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_lengths,  (size_t)n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * SHA256_DIGEST_SIZE));

    /* One big host->device copy for the messages (PCIe loves big transfers). */
    if (span) CUDA_CHECK(cudaMemcpy(d_messages, messages, span, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets, (size_t)n * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, lengths, (size_t)n * sizeof(size_t), cudaMemcpyHostToDevice));

    /* One thread per message; round the grid up to cover all n. */
    dim3 threads(SHA256_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));
    kernel_sha256_batch<<<grid, threads>>>(d_messages, d_offsets, d_lengths, n, d_digests);
    CUDA_CHECK(cudaGetLastError());                /* catch launch errors */
    CUDA_CHECK(cudaDeviceSynchronize());           /* wait + catch run errors */

    CUDA_CHECK(cudaMemcpy(digests, d_digests, (size_t)n * SHA256_DIGEST_SIZE,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_digests));
}

extern "C" void sha256_gpu_batch_fixed(const uint8_t* messages,
                                       size_t         msg_len,
                                       int            n,
                                       uint8_t*       digests) {
    if (n <= 0) return;
    upload_constants();

    size_t total = (size_t)n * msg_len;
    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, total ? total : 1));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * SHA256_DIGEST_SIZE));
    if (total) CUDA_CHECK(cudaMemcpy(d_messages, messages, total, cudaMemcpyHostToDevice));

    dim3 threads(SHA256_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));
    kernel_sha256_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, d_digests);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(digests, d_digests, (size_t)n * SHA256_DIGEST_SIZE,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_digests));
}

extern "C" void sha256_gpu_benchmark(size_t  msg_len,
                                     int     n,
                                     int     iters,
                                     double* kernel_gbps,
                                     double* e2e_gbps,
                                     double* mhs) {
    if (iters < 1) iters = 1;
    if (n <= 0)    { if (kernel_gbps) *kernel_gbps = 0; if (e2e_gbps) *e2e_gbps = 0; if (mhs) *mhs = 0; return; }
    upload_constants();

    size_t total = (size_t)n * msg_len;

    /* Stage a pseudo-random host buffer (varying bytes, like real input). */
    uint8_t* h_buf = (uint8_t*)std::malloc(total ? total : 1);
    if (h_buf) for (size_t i = 0; i < total; ++i) h_buf[i] = (uint8_t)(i * 167 + 13);

    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, total ? total : 1));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * SHA256_DIGEST_SIZE));
    CUDA_CHECK(cudaMemcpy(d_messages, h_buf, total, cudaMemcpyHostToDevice));

    dim3 threads(SHA256_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* (1) KERNEL-ONLY throughput: data already resident, no copies timed. */
    kernel_sha256_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, d_digests); /* warm-up */
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
        kernel_sha256_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, d_digests);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_k = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms_k, start, stop));
    if (kernel_gbps) *kernel_gbps = ((double)total * iters) / ((double)ms_k / 1000.0) / 1e9;
    if (mhs)         *mhs         = ((double)n     * iters) / ((double)ms_k / 1000.0) / 1e6;

    /* (2) END-TO-END throughput of the same kernel, INCLUDING the PCIe copies in
     * and out -- what a caller of sha256_gpu_batch_fixed() actually experiences. */
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemcpy(d_messages, h_buf, total, cudaMemcpyHostToDevice));
        kernel_sha256_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, d_digests);
        CUDA_CHECK(cudaMemcpy(h_buf, d_digests, (size_t)n * SHA256_DIGEST_SIZE, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_e = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms_e, start, stop));
    if (e2e_gbps) *e2e_gbps = ((double)total * iters) / ((double)ms_e / 1000.0) / 1e9;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::free(h_buf);
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_digests));
}

/* ============================================================================
 *  sha_cuda_device_info  --  shared device-info helper (declared in sha_common.h)
 * ============================================================================
 *  Defined here (rather than duplicated in sha3_gpu.cu) so there is exactly one
 *  definition when both .cu files are linked into the library/executable.
 * ========================================================================== */
extern "C" int sha_cuda_device_info(void) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::fprintf(stderr, "No CUDA-capable device found.\n");
        return 0;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
    std::printf("CUDA device %d: %s (compute capability %d.%d)\n",
                dev, p.name, p.major, p.minor);
    std::printf("  Multiprocessors (SMs): %d\n", p.multiProcessorCount);
    std::printf("  Global memory        : %.2f GiB\n",
                (double)p.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    std::printf("  Max threads / block  : %d\n", p.maxThreadsPerBlock);
    return 1;
}
