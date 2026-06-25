/* ============================================================================
 *  sha_cpu_reference.cpp  --  Readable CPU SHA-256 and SHA-3, the TRUSTED ORACLE
 * ============================================================================
 *
 *  This file is written to be READ. Every constant is named, every step matches
 *  the published standard's notation, and speed is deliberately sacrificed for
 *  clarity (e.g. SHA-256 uses the full 64-word schedule W[64] verbatim from FIPS
 *  180-4). The GPU kernels in src/*_gpu.cu perform the IDENTICAL math, just
 *  reorganized for throughput; comparing the two side by side is the whole point.
 *
 *  Layout of this file:
 *    PART A -- SHA-256  (FIPS 180-4)
 *    PART B -- SHA-3 / Keccak-f[1600]  (FIPS 202)
 *
 *  Nothing here touches CUDA; it is plain, portable C++.
 * ========================================================================== */

#include "sha_cpu_reference.h"
#include <cstring>   /* std::memcpy, std::memset */

/* ============================================================================
 *  PART A -- SHA-256   (FIPS 180-4, section 6.2)
 * ============================================================================ */

/* --- Small helpers. SHA-256 works on 32-bit words and big-endian bytes. ---- */

/* Rotate-right of a 32-bit word: the bits that fall off the bottom reappear at
 * the top. (1 <= n <= 31 in every use here, so no zero/full-width edge case.) */
static inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

/* Read 4 bytes as a big-endian (most-significant-byte-first) 32-bit word. */
static inline uint32_t be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] <<  8) | ((uint32_t)p[3]);
}

/* The six bit-mixing functions of SHA-256, named exactly as in FIPS 180-4.
 *  Ch  ("choose")     : for each bit, pick y's bit where x is 1, else z's bit.
 *  Maj ("majority")   : for each bit, the majority value among x, y, z.
 *  BSIG0/BSIG1 (big sigma, the Sigma used in the compression round update).
 *  SSIG0/SSIG1 (small sigma, the sigma used in the message schedule).
 * The specific rotation/shift amounts are part of the standard and chosen so
 * that, composed over 64 rounds, every input bit influences every output bit. */
static inline uint32_t Ch (uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
static inline uint32_t BSIG0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
static inline uint32_t BSIG1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
static inline uint32_t SSIG0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
static inline uint32_t SSIG1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

/* The 64 round constants K[0..63].
 * These are the first 32 bits of the FRACTIONAL parts of the cube roots of the
 * first 64 prime numbers (2, 3, 5, 7, ...). "Nothing-up-my-sleeve" numbers like
 * these prove the designers did not secretly choose constants to plant a
 * backdoor -- anyone can recompute them from the primes. */
static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* The eight initial hash values H0..H7.
 * These are the first 32 bits of the fractional parts of the SQUARE roots of the
 * first 8 primes -- another nothing-up-my-sleeve choice. The running state
 * starts here and is stirred by every block. */
static const uint32_t H0_256[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* Compress ONE 64-byte block into the running state H[8].
 * This is the heart of SHA-256: it expands the 16 input words into a 64-word
 * "message schedule" W, then runs 64 rounds that fold W and the constants K into
 * eight working registers a..h, and finally adds the result back into H. */
static void sha256_compress(uint32_t H[8], const uint8_t block[64]) {
    uint32_t W[64];

    /* (1) The first 16 schedule words are just the block, read big-endian. */
    for (int t = 0; t < 16; ++t)
        W[t] = be32(block + 4 * t);

    /* (2) Extend to 64 words. Each new word mixes four earlier ones; this is
     * what spreads the influence of every input byte throughout the schedule. */
    for (int t = 16; t < 64; ++t)
        W[t] = SSIG1(W[t - 2]) + W[t - 7] + SSIG0(W[t - 15]) + W[t - 16];

    /* (3) Initialize the eight working variables from the current state. */
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
    uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

    /* (4) 64 rounds. T1 folds in the schedule word and round constant; T2 is the
     * data-independent diffusion term. The registers shift down by one (h<-g,
     * g<-f, ...) with the two new values e and a coming from T1/T2. */
    for (int t = 0; t < 64; ++t) {
        uint32_t T1 = h + BSIG1(e) + Ch(e, f, g) + K256[t] + W[t];
        uint32_t T2 = BSIG0(a) + Maj(a, b, c);
        h = g; g = f; f = e; e = d + T1;
        d = c; c = b; b = a; a = T1 + T2;
    }

    /* (5) Add the working variables back into the state (the Davies-Meyer
     * feed-forward that makes the compression function one-way). */
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;
}

extern "C" void sha256_cpu(const uint8_t* msg, size_t len, uint8_t out[32]) {
    /* Start from the standard initial state. */
    uint32_t H[8];
    std::memcpy(H, H0_256, sizeof(H));

    /* Absorb every COMPLETE 64-byte block straight from the message. */
    size_t full = len / 64;
    for (size_t i = 0; i < full; ++i)
        sha256_compress(H, msg + i * 64);

    /* PADDING (FIPS 180-4 section 5.1.1). We must append a single '1' bit, then
     * enough '0' bits, then the 64-bit big-endian message length in bits, so the
     * total is a multiple of 64 bytes. In bytes: append 0x80, then zeros, then
     * the 8-byte length. Build the final one or two blocks by hand. */
    uint8_t block[64];
    size_t rem = len - full * 64;          /* leftover bytes, 0..63           */
    std::memcpy(block, msg + full * 64, rem);
    block[rem] = 0x80;                      /* the '1' bit, byte-aligned       */
    std::memset(block + rem + 1, 0, 64 - (rem + 1));

    uint64_t bitlen = (uint64_t)len * 8;    /* message length in BITS          */

    /* If the 0x80 byte landed at offset >= 56 there is no room for the 8-byte
     * length in this block, so we finish this block now and start a fresh,
     * all-zero block to carry just the length. (The 56-byte test vector exists
     * precisely to exercise this two-block path.) */
    if (rem >= 56) {
        sha256_compress(H, block);
        std::memset(block, 0, 64);
    }

    /* Write the 64-bit length, big-endian, into the last 8 bytes. */
    for (int i = 0; i < 8; ++i)
        block[56 + i] = (uint8_t)(bitlen >> (56 - 8 * i));
    sha256_compress(H, block);

    /* Serialize the final state big-endian -> the 32-byte digest. */
    for (int i = 0; i < 8; ++i) {
        out[4 * i + 0] = (uint8_t)(H[i] >> 24);
        out[4 * i + 1] = (uint8_t)(H[i] >> 16);
        out[4 * i + 2] = (uint8_t)(H[i] >>  8);
        out[4 * i + 3] = (uint8_t)(H[i]);
    }
}

/* ============================================================================
 *  PART B -- SHA-3 / Keccak-f[1600]   (FIPS 202)
 * ============================================================================
 *
 *  The state is 1600 bits laid out as a 5 x 5 grid of 64-bit "lanes". We store
 *  it as a flat array st[25], indexing lane (x, y) at st[x + 5*y]. The sponge
 *  XORs message bytes into the first `rate` bytes of this state, then applies the
 *  Keccak-f permutation (24 rounds), and repeats; output is read back out of the
 *  same first `rate` bytes. The byte<->lane mapping is LITTLE-ENDIAN, which is
 *  why we can alias the state as a byte array on little-endian CPUs and GPUs.
 *
 *  This permutation is the well-known compact form (after M-J. Saarinen's
 *  tiny_sha3): the rho+pi steps are fused using two small constant tables. The
 *  alternative is to write theta/rho/pi/chi/iota as five fully separate loops;
 *  docs/02 shows that longhand version. The math is identical either way. */

/* Rotate-left of a 64-bit lane (1 <= n <= 63 for every offset used below). */
static inline uint64_t rotl64_cpu(uint64_t x, unsigned n) {
    return (x << n) | (x >> (64u - n));
}

/* The 24 ROUND CONSTANTS for iota, one XORed into lane (0,0) each round.
 * They encode a maximal-length LFSR sequence; their job is to break the symmetry
 * that theta/rho/pi/chi would otherwise leave between the 24 rounds. */
static const uint64_t keccak_rc[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
    0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
    0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
};

/* The 24 RHO rotation offsets, in the order the fused rho+pi loop visits lanes.
 * rho rotates each lane by a fixed amount; these are the triangular-number
 * offsets t(t+1)/2 mod 64 from the Keccak spec, pre-sequenced for the loop. */
static const unsigned keccak_rotc[24] = {
     1,  3,  6, 10, 15, 21, 28, 36, 45, 55,  2, 14,
    27, 41, 56,  8, 25, 43, 62, 18, 39, 61, 20, 44
};

/* The PI lane permutation, as destination indices for the same fused loop:
 * the lane currently in hand is written to st[keccak_piln[i]] next. pi is the
 * step that moves lanes around the 5x5 grid (it transposes-and-shuffles). */
static const unsigned keccak_piln[24] = {
    10,  7, 11, 17, 18,  3,  5, 16,  8, 21, 24,  4,
    15, 23, 19, 13, 12,  2, 20, 14, 22,  9,  6,  1
};

/* The Keccak-f[1600] permutation: 24 rounds of theta, rho+pi, chi, iota. */
static void keccakf(uint64_t st[25]) {
    for (int round = 0; round < 24; ++round) {
        uint64_t bc[5], t;

        /* THETA: compute the parity of each of the 5 columns, then XOR into every
         * lane a combination of its neighboring columns' parities (one of them
         * rotated by 1). This is the step that couples all 5 columns together --
         * the main source of long-range diffusion. */
        for (int i = 0; i < 5; ++i)
            bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rotl64_cpu(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5)
                st[j + i] ^= t;
        }

        /* RHO + PI (fused): walk a chain of 24 lanes; each is rotated left by its
         * rho offset (keccak_rotc) and deposited at its pi destination
         * (keccak_piln). `t` carries the lane being relocated. rho provides
         * intra-lane diffusion (bits move within a 64-bit word); pi provides
         * inter-lane diffusion (lanes move around the grid). */
        t = st[1];
        for (int i = 0; i < 24; ++i) {
            int j = (int)keccak_piln[i];
            bc[0] = st[j];
            st[j] = rotl64_cpu(t, keccak_rotc[i]);
            t = bc[0];
        }

        /* CHI: the only NON-LINEAR step. For each row, replace each lane with
         * lane XOR ((NOT next-lane) AND lane-after-that). Without chi the whole
         * permutation would be linear (and trivially breakable). */
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
            for (int i = 0; i < 5; ++i)
                st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }

        /* IOTA: XOR this round's constant into lane (0,0) to break symmetry. */
        st[0] ^= keccak_rc[round];
    }
}

extern "C" void sha3_cpu(int rate_bytes, int out_bytes,
                         const uint8_t* msg, size_t len, uint8_t* out) {
    /* The sponge state starts empty (all zero). We alias it as bytes so that
     * absorbing and squeezing are simple byte XOR / byte copy. This relies on
     * the host being little-endian (true on x86/ARM and on every CUDA GPU). */
    uint64_t st[25];
    std::memset(st, 0, sizeof(st));
    uint8_t* sb = reinterpret_cast<uint8_t*>(st);

    /* ABSORB every COMPLETE rate-sized block: XOR `rate_bytes` of message into
     * the front of the state, then stir with the permutation. */
    size_t i = 0;
    while (len - i >= (size_t)rate_bytes) {
        for (int k = 0; k < rate_bytes; ++k) sb[k] ^= msg[i + k];
        keccakf(st);
        i += rate_bytes;
    }

    /* FINAL (partial) block + PADDING. First XOR in the leftover message bytes.
     * Then apply SHA-3's "pad10*1 with domain suffix":
     *   - sb[rem] ^= 0x06 : the domain bits "01" of SHA-3 followed by the first
     *                       padding '1' bit (0x06 = binary 0000_0110, read LSB
     *                       first: bits 0,1,1 -> the "01" suffix then the pad 1).
     *   - sb[rate-1] ^= 0x80 : the final padding '1' bit at the very end of the
     *                       rate block.
     * If the message exactly fills rate-1 bytes, these two land in the same byte
     * and combine to 0x86 -- handled automatically because we XOR. If the message
     * exactly fills a whole rate block, rem == 0 and this is a full extra
     * padding-only block (also handled automatically). */
    size_t rem = len - i;                 /* 0 .. rate_bytes-1 */
    for (size_t k = 0; k < rem; ++k) sb[k] ^= msg[i + k];
    sb[rem]            ^= 0x06;
    sb[rate_bytes - 1] ^= 0x80;
    keccakf(st);

    /* SQUEEZE. For all four fixed SHA-3 variants the digest fits within a single
     * rate block (out_bytes <= rate_bytes always holds), so one read suffices --
     * no further permutation is needed. Copy the first out_bytes of the state. */
    for (int k = 0; k < out_bytes; ++k) out[k] = sb[k];
}
