/* ============================================================================
 *  demo.cpp  --  A guided, runnable tour of the batched CUDA SHA hasher
 * ============================================================================
 *
 *  This is the "start here" program. It walks through six sections, each
 *  printing what it is doing and why, so you can read the output top to bottom
 *  and understand both the cryptography and the GPU lessons:
 *
 *    1. Which GPU you are on.
 *    2. SHA-256 and SHA3-256 of "abc" -- matched byte-for-byte to the published
 *       digests, so you trust the rest.
 *    3. BATCHED hashing: hash many strings in one launch, one thread per message.
 *    4. The SHA-3 family: the same input under all four variants, illustrating
 *       the rate/capacity/output-length relationship.
 *    5. The AVALANCHE effect: flip ONE input bit and watch ~half the output bits
 *       change -- the defining property of a good hash.
 *    6. THROUGHPUT: batched hashing benchmarks (kernel-only vs end-to-end), and
 *       the SHA3-256-vs-SHA3-512 rate trade-off.
 *
 *  Everything here calls the public API (sha256.h / sha3.h) and the CPU
 *  reference (for verification), nothing CUDA-specific -- the GPU details are
 *  hidden behind the library, which is the whole point of the API design.
 * ========================================================================== */

#include "sha256.h"
#include "sha3.h"
#include "sha_cpu_reference.h"
#include "sha_common.h"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

/* Print `n` bytes as hex (for showing digests). */
static void print_hex(const uint8_t* p, int n) {
    for (int i = 0; i < n; ++i) std::printf("%02x", p[i]);
}

/* Count how many of the bits differ between two equal-length byte buffers.
 * Used by the avalanche demo. popcount of the XOR per byte, summed. */
static int bit_diff(const uint8_t* a, const uint8_t* b, int n) {
    int d = 0;
    for (int i = 0; i < n; ++i) {
        uint8_t x = a[i] ^ b[i];
        while (x) { d += (x & 1); x >>= 1; }
    }
    return d;
}

static void section(const char* title) {
    std::printf("\n========================================================\n");
    std::printf("  %s\n", title);
    std::printf("========================================================\n");
}

int main() {
    std::printf("CUDA batched SHA-256 / SHA-3 -- guided demo\n");

    /* ---- 1. Device ---- */
    section("1. The GPU doing the work");
    if (!sha_cuda_device_info()) {
        std::printf("No CUDA device; cannot run the demo.\n");
        return 1;
    }

    /* ---- 2. Single-hash sanity check against the standard ---- */
    section("2. \"abc\" matches the published digests");
    {
        const char* msg = "abc";
        size_t      len = 3;
        size_t      off = 0;

        uint8_t d256[32];
        sha256_gpu_batch((const uint8_t*)msg, &off, &len, 1, d256);
        std::printf("  SHA-256(\"abc\")   = "); print_hex(d256, 32); std::printf("\n");
        std::printf("  expected (FIPS)  = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n");

        uint8_t d3[32];
        sha3_gpu_batch(SHA3_256, (const uint8_t*)msg, &off, &len, 1, d3);
        std::printf("  SHA3-256(\"abc\")  = "); print_hex(d3, 32); std::printf("\n");
        std::printf("  expected (FIPS)  = 3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532\n");
    }

    /* ---- 3. Batched hashing: many messages, one launch ---- */
    section("3. Batched hashing -- one GPU thread per message");
    {
        /* A little batch of words. We build the three-array batch layout
         * (messages / offsets / lengths) described in sha_common.h. */
        const char* words[] = { "the", "quick", "brown", "fox", "jumps",
                                "over", "the", "lazy", "dog" };
        int n = (int)(sizeof(words) / sizeof(words[0]));

        std::vector<uint8_t> messages;
        std::vector<size_t>  offsets(n), lengths(n);
        for (int i = 0; i < n; ++i) {
            offsets[i] = messages.size();
            lengths[i] = std::strlen(words[i]);
            messages.insert(messages.end(), words[i], words[i] + lengths[i]);
        }

        std::vector<uint8_t> digests((size_t)n * SHA256_DIGEST_SIZE);
        sha256_gpu_batch(messages.data(), offsets.data(), lengths.data(), n, digests.data());

        std::printf("  Hashed %d messages in a SINGLE kernel launch:\n", n);
        for (int i = 0; i < n; ++i) {
            std::printf("    SHA-256(%-6s) = ", ("\"" + std::string(words[i]) + "\"").c_str());
            print_hex(digests.data() + (size_t)i * 32, 8);     /* first 8 bytes for brevity */
            std::printf("...\n");
        }
        /* Note the two "the" entries produce identical digests -- a hash is a
         * deterministic function of its input. */
        std::printf("  (Both \"the\" rows match: a hash is deterministic.)\n");
    }

    /* ---- 4. The SHA-3 family: rate, capacity, output length ---- */
    section("4. The SHA-3 family on one input (rate vs capacity)");
    {
        const char* msg = "The quick brown fox jumps over the lazy dog";
        size_t len = std::strlen(msg), off = 0;
        Sha3Variant vs[4]   = { SHA3_224, SHA3_256, SHA3_384, SHA3_512 };
        const char* names[4] = { "SHA3-224", "SHA3-256", "SHA3-384", "SHA3-512" };
        std::printf("  variant    out  rate  capacity   digest\n");
        for (int t = 0; t < 4; ++t) {
            int outlen = sha3_digest_bytes(vs[t]);
            int rate   = sha3_rate_bytes(vs[t]);
            std::vector<uint8_t> d(outlen);
            sha3_gpu_batch(vs[t], (const uint8_t*)msg, &off, &len, 1, d.data());
            std::printf("  %-8s  %3dB %4dB   %4d bit  ", names[t], outlen, rate, 2 * (int)vs[t]);
            print_hex(d.data(), outlen > 16 ? 16 : outlen);
            std::printf("%s\n", outlen > 16 ? "..." : "");
        }
        std::printf("  Bigger capacity (more security) -> smaller rate -> slower (sec. 6).\n");
    }

    /* ---- 5. Avalanche effect ---- */
    section("5. Avalanche -- flip one input bit, ~half the output flips");
    {
        const char* base = "avalanche test vector zero";
        std::string a(base), b(base);
        b[0] ^= 0x01;                       /* flip the lowest bit of the first byte */

        size_t la = a.size(), lb = b.size(), off = 0;
        uint8_t da[32], db[32];
        sha256_gpu_batch((const uint8_t*)a.data(), &off, &la, 1, da);
        sha256_gpu_batch((const uint8_t*)b.data(), &off, &lb, 1, db);

        int diff = bit_diff(da, db, 32);
        std::printf("  input A: \"%s\"\n", a.c_str());
        std::printf("  input B: same, but 1 bit flipped in the first byte\n");
        std::printf("  SHA-256(A) = "); print_hex(da, 32); std::printf("\n");
        std::printf("  SHA-256(B) = "); print_hex(db, 32); std::printf("\n");
        std::printf("  bits changed: %d of 256 (%.1f%%) -- ideal is ~50%%\n",
                    diff, 100.0 * diff / 256.0);
    }

    /* ---- 6. Throughput ---- */
    section("6. Throughput -- batched, high-throughput hashing");
    {
        double k_gbps, e_gbps, mhs;

        /* WORKLOAD A -- many SHORT messages (64 bytes each). This is the
         * "hashes per second" regime: keys, password candidates, dedup
         * fingerprints. Each 64-byte message is a single rate block for every
         * algorithm here, so the metric that matters is M msg/s, and the PCIe
         * tax (kernel-only vs end-to-end) is at its most brutal. */
        {
            size_t msg_len = 64;
            int    n       = 1 << 20;     /* ~1 million messages */
            int    iters   = 50;
            std::printf("  WORKLOAD A: %d messages x %zu bytes (short-record regime)\n\n", n, msg_len);

            sha256_gpu_benchmark(msg_len, n, iters, &k_gbps, &e_gbps, &mhs);
            std::printf("    SHA-256   kernel-only: %7.2f GB/s   end-to-end: %6.2f GB/s   %8.1f M msg/s\n",
                        k_gbps, e_gbps, mhs);
            sha3_gpu_benchmark(SHA3_256, msg_len, n, iters, &k_gbps, &e_gbps, &mhs);
            std::printf("    SHA3-256  kernel-only: %7.2f GB/s   end-to-end: %6.2f GB/s   %8.1f M msg/s\n",
                        k_gbps, e_gbps, mhs);
        }

        /* WORKLOAD B -- fewer LONG messages (8 KiB each). Now each message spans
         * MANY rate blocks, so the sponge rate decides the speed: SHA3-256
         * (rate 136 B) needs 61 Keccak-f permutations for 8 KiB, while SHA3-512
         * (rate 72 B) needs 114 -> SHA3-256 should be ~1.8x faster PER BYTE.
         * This is the workload where the rate/capacity trade-off is visible
         * (it is invisible for single-block messages like Workload A). */
        {
            size_t msg_len = 8192;
            int    n       = 1 << 16;     /* 65536 messages */
            int    iters   = 50;
            std::printf("\n  WORKLOAD B: %d messages x %zu bytes (multi-block, rate matters)\n\n", n, msg_len);

            sha256_gpu_benchmark(msg_len, n, iters, &k_gbps, &e_gbps, &mhs);
            std::printf("    SHA-256   kernel-only: %7.2f GB/s   end-to-end: %6.2f GB/s\n", k_gbps, e_gbps);
            sha3_gpu_benchmark(SHA3_256, msg_len, n, iters, &k_gbps, &e_gbps, &mhs);
            std::printf("    SHA3-256  kernel-only: %7.2f GB/s   end-to-end: %6.2f GB/s\n", k_gbps, e_gbps);
            sha3_gpu_benchmark(SHA3_512, msg_len, n, iters, &k_gbps, &e_gbps, &mhs);
            std::printf("    SHA3-512  kernel-only: %7.2f GB/s   end-to-end: %6.2f GB/s\n", k_gbps, e_gbps);
        }

        std::printf("\n  Lessons:\n");
        std::printf("   * kernel-only >> end-to-end: for short messages the PCIe copy\n");
        std::printf("     dominates. Real pipelines keep data resident on the GPU.\n");
        std::printf("   * In Workload B, SHA3-256 (rate 136B) beats SHA3-512 (rate 72B)\n");
        std::printf("     per byte, because it absorbs more data per Keccak-f permutation.\n");
        std::printf("   * SHA-256 >> SHA-3 here: SHA-256 keeps its state in registers,\n");
        std::printf("     while our readable Keccak keeps its 200-byte state in local\n");
        std::printf("     memory. That is the register-vs-local lesson -- see docs/04.\n");
    }

    std::printf("\nDemo complete. See docs/ for the deep dives, and run the test\n");
    std::printf("suite (sha_tests) to verify every digest against the standards.\n");
    return 0;
}
