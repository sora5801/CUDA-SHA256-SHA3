/* ============================================================================
 *  sha3_gpu.cu  --  The CUDA side: BATCHED, high-throughput SHA-3 (Keccak)
 * ============================================================================
 *
 *  THE BIG IDEA (same as SHA-256: parallelize over MESSAGES)
 *  ---------------------------------------------------------
 *  Like SHA-256, a single SHA-3 hash is sequential (each Keccak-f permutation
 *  depends on the previous), but hashing N independent messages is fully
 *  parallel. So again: one GPU thread per message, thousands at a time. One
 *  kernel serves all four SHA-3 variants because they differ only in the sponge
 *  RATE and the output length -- the Keccak-f[1600] permutation is identical.
 *
 *  STATE LAYOUT
 *  ------------
 *  The 1600-bit state is 25 lanes of 64 bits: uint64_t st[25], lane (x,y) at
 *  st[x + 5*y]. The sponge XORs message bytes into the first `rate` bytes of the
 *  state (viewed as a byte array, little-endian per lane), runs Keccak-f, and
 *  repeats; the digest is read back from the front of the state. NVIDIA GPUs are
 *  little-endian, so aliasing st as bytes gives the exact FIPS-202 byte order.
 *
 *  A HONEST PERFORMANCE NOTE (documented, not hidden)
 *  --------------------------------------------------
 *  Keccak-f indexes the state with data-/loop-dependent indices (st[j+i],
 *  st[piln[i]]), so the 200-byte state cannot live purely in registers with this
 *  compact, READABLE formulation -- it sits in per-thread "local" memory. That is
 *  the deliberate trade-off here: clarity that mirrors the CPU reference line for
 *  line. A faster variant fully unrolls Keccak-f into 25 named registers; see
 *  docs/02 and docs/06 for that optimization as an exercise.
 *
 *  CONSTANT MEMORY IS A PERFECT FIT (again)
 *  ----------------------------------------
 *  The round constants, rho offsets, and pi indices are indexed by the round/loop
 *  counter, which is identical across every thread in a warp -> constant memory
 *  broadcasts them for free. No shared-memory staging is needed.
 * ========================================================================== */

#include "sha3.h"
#include "sha_cuda.cuh"

/* ============================================================================
 *  CONSTANT-MEMORY KECCAK TABLES  (shared by every thread; see CPU reference)
 * ============================================================================ */
__constant__ uint64_t c_rc[24];    /* iota round constants                    */
__constant__ unsigned c_rotc[24];  /* rho rotation offsets (fused loop order) */
__constant__ unsigned c_piln[24];  /* pi destination indices (fused loop)     */

/* Host-side copies, uploaded once per launch. Identical to the CPU reference. */
static const uint64_t h_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};
static const unsigned h_rotc[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};
static const unsigned h_piln[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

/* ----------------------------------------------------------------------------
 *  dev_keccakf  --  the Keccak-f[1600] permutation, 24 rounds, on the GPU
 * ----------------------------------------------------------------------------
 *  Identical math to keccakf() in the CPU reference; read that first for the
 *  step-by-step theta/rho/pi/chi/iota commentary. rotl64 is from sha_cuda.cuh.
 * -------------------------------------------------------------------------- */
__device__ void dev_keccakf(uint64_t st[25]) {
    #pragma unroll 1                                   /* keep code size sane */
    for (int round = 0; round < 24; ++round) {
        uint64_t bc[5], t;

        /* THETA -- couple the 5 columns via their parities. */
        #pragma unroll
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        #pragma unroll
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
            #pragma unroll
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        /* RHO + PI -- rotate each lane and relocate it around the grid. */
        t = st[1];
        #pragma unroll
        for (int i = 0; i < 24; ++i) {
            unsigned j = c_piln[i];
            bc[0] = st[j];
            st[j] = rotl64(t, c_rotc[i]);
            t = bc[0];
        }

        /* CHI -- the nonlinear row mix. */
        #pragma unroll
        for (int j = 0; j < 25; j += 5) {
            #pragma unroll
            for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
            #pragma unroll
            for (int i = 0; i < 5; ++i)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        /* IOTA -- break inter-round symmetry. */
        st[0] ^= c_rc[round];
    }
}

/* ----------------------------------------------------------------------------
 *  dev_sha3  --  full SHA-3 of ONE message, run by one thread
 * ----------------------------------------------------------------------------
 *  rate / outlen : sponge rate and digest length in bytes (from the variant).
 *  Mirrors sha3_cpu(): absorb full rate blocks, pad the final block with the
 *  0x06 domain suffix and the 0x80 terminator, then squeeze `outlen` bytes.
 * -------------------------------------------------------------------------- */
__device__ void dev_sha3(const uint8_t* msg, size_t len,
                         int rate, int outlen, uint8_t* out) {
    uint64_t st[25];
    #pragma unroll
    for (int i = 0; i < 25; ++i) st[i] = 0;
    uint8_t* sb = (uint8_t*)st;                 /* byte view of the state */

    /* ABSORB complete rate-sized blocks. */
    size_t i = 0;
    while (len - i >= (size_t)rate) {
        for (int k = 0; k < rate; ++k) sb[k] ^= msg[i + k];
        dev_keccakf(st);
        i += rate;
    }

    /* FINAL block + pad10*1 with the SHA-3 domain suffix (see CPU reference for
     * the bit-level explanation of 0x06 and 0x80; XOR makes the rate-1 and
     * exact-multiple edge cases fall out automatically). */
    size_t rem = len - i;                       /* 0 .. rate-1 */
    for (size_t k = 0; k < rem; ++k) sb[k] ^= msg[i + k];
    sb[rem]      ^= 0x06;
    sb[rate - 1] ^= 0x80;
    dev_keccakf(st);

    /* SQUEEZE: one read suffices (outlen <= rate for all fixed SHA-3 variants). */
    for (int k = 0; k < outlen; ++k) out[k] = sb[k];
}

/* ============================================================================
 *  KERNELS  --  one thread per message
 * ============================================================================ */

/* VARIABLE-LENGTH batch. digests are packed `outlen` bytes apart. */
__global__ void kernel_sha3_batch(const uint8_t* __restrict__ messages,
                                  const size_t*  __restrict__ offsets,
                                  const size_t*  __restrict__ lengths,
                                  int            n,
                                  int            rate,
                                  int            outlen,
                                  uint8_t*       __restrict__ digests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dev_sha3(messages + offsets[idx], lengths[idx], rate, outlen,
             digests + (size_t)idx * outlen);
}

/* FIXED-LENGTH batch (zero divergence): message idx starts at idx*msg_len. */
__global__ void kernel_sha3_batch_fixed(const uint8_t* __restrict__ messages,
                                        size_t         msg_len,
                                        int            n,
                                        int            rate,
                                        int            outlen,
                                        uint8_t*       __restrict__ digests) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dev_sha3(messages + (size_t)idx * msg_len, msg_len, rate, outlen,
             digests + (size_t)idx * outlen);
}

/* ============================================================================
 *  HOST SIDE
 * ============================================================================ */
#define SHA3_THREADS_PER_BLOCK 256

/* Upload the three Keccak constant tables (a few hundred bytes total). */
static void upload_constants() {
    CUDA_CHECK(cudaMemcpyToSymbol(c_rc,   h_rc,   sizeof(h_rc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_rotc, h_rotc, sizeof(h_rotc)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_piln, h_piln, sizeof(h_piln)));
}

extern "C" void sha3_gpu_batch(Sha3Variant    variant,
                               const uint8_t* messages,
                               const size_t*  offsets,
                               const size_t*  lengths,
                               int            n,
                               uint8_t*       digests) {
    if (n <= 0) return;
    upload_constants();
    int rate   = sha3_rate_bytes(variant);
    int outlen = sha3_digest_bytes(variant);

    /* Farthest byte any thread reads (messages need not be tightly packed). */
    size_t span = 0;
    for (int i = 0; i < n; ++i) {
        size_t end = offsets[i] + lengths[i];
        if (end > span) span = end;
    }

    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    size_t  *d_offsets = nullptr, *d_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, span ? span : 1));
    CUDA_CHECK(cudaMalloc(&d_offsets,  (size_t)n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_lengths,  (size_t)n * sizeof(size_t)));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * outlen));

    if (span) CUDA_CHECK(cudaMemcpy(d_messages, messages, span, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, offsets, (size_t)n * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lengths, lengths, (size_t)n * sizeof(size_t), cudaMemcpyHostToDevice));

    dim3 threads(SHA3_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));
    kernel_sha3_batch<<<grid, threads>>>(d_messages, d_offsets, d_lengths, n,
                                         rate, outlen, d_digests);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(digests, d_digests, (size_t)n * outlen, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_digests));
}

extern "C" void sha3_gpu_batch_fixed(Sha3Variant    variant,
                                     const uint8_t* messages,
                                     size_t         msg_len,
                                     int            n,
                                     uint8_t*       digests) {
    if (n <= 0) return;
    upload_constants();
    int rate   = sha3_rate_bytes(variant);
    int outlen = sha3_digest_bytes(variant);

    size_t total = (size_t)n * msg_len;
    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, total ? total : 1));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * outlen));
    if (total) CUDA_CHECK(cudaMemcpy(d_messages, messages, total, cudaMemcpyHostToDevice));

    dim3 threads(SHA3_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));
    kernel_sha3_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, rate, outlen, d_digests);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(digests, d_digests, (size_t)n * outlen, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_messages));
    CUDA_CHECK(cudaFree(d_digests));
}

extern "C" void sha3_gpu_benchmark(Sha3Variant variant,
                                   size_t      msg_len,
                                   int         n,
                                   int         iters,
                                   double*     kernel_gbps,
                                   double*     e2e_gbps,
                                   double*     mhs) {
    if (iters < 1) iters = 1;
    if (n <= 0) { if (kernel_gbps) *kernel_gbps = 0; if (e2e_gbps) *e2e_gbps = 0; if (mhs) *mhs = 0; return; }
    upload_constants();
    int rate   = sha3_rate_bytes(variant);
    int outlen = sha3_digest_bytes(variant);

    size_t total = (size_t)n * msg_len;
    uint8_t* h_buf = (uint8_t*)std::malloc(total ? total : 1);
    if (h_buf) for (size_t i = 0; i < total; ++i) h_buf[i] = (uint8_t)(i * 167 + 13);

    uint8_t *d_messages = nullptr, *d_digests = nullptr;
    CUDA_CHECK(cudaMalloc(&d_messages, total ? total : 1));
    CUDA_CHECK(cudaMalloc(&d_digests,  (size_t)n * outlen));
    CUDA_CHECK(cudaMemcpy(d_messages, h_buf, total, cudaMemcpyHostToDevice));

    dim3 threads(SHA3_THREADS_PER_BLOCK);
    dim3 grid((unsigned)(((size_t)n + threads.x - 1) / threads.x));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* (1) kernel-only. */
    kernel_sha3_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, rate, outlen, d_digests);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it)
        kernel_sha3_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, rate, outlen, d_digests);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_k = 0.0f; CUDA_CHECK(cudaEventElapsedTime(&ms_k, start, stop));
    if (kernel_gbps) *kernel_gbps = ((double)total * iters) / ((double)ms_k / 1000.0) / 1e9;
    if (mhs)         *mhs         = ((double)n     * iters) / ((double)ms_k / 1000.0) / 1e6;

    /* (2) end-to-end with PCIe copies. */
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemcpy(d_messages, h_buf, total, cudaMemcpyHostToDevice));
        kernel_sha3_batch_fixed<<<grid, threads>>>(d_messages, msg_len, n, rate, outlen, d_digests);
        CUDA_CHECK(cudaMemcpy(h_buf, d_digests, (size_t)n * outlen, cudaMemcpyDeviceToHost));
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
