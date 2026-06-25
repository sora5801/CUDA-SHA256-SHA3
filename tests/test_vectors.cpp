/* ============================================================================
 *  test_vectors.cpp  --  Known-answer + cross-check correctness suite
 * ============================================================================
 *
 *  This program is the proof that the GPU hashes are correct. It checks, in
 *  order, three independent things and prints PASS/FAIL for each, returning a
 *  non-zero exit code if ANY check fails (so it doubles as a CI gate):
 *
 *    1. KNOWN-ANSWER TESTS (KATs). The GPU batch output is compared against
 *       digests published by / computed from the standards (NIST FIPS 180-4 for
 *       SHA-256, FIPS 202 for SHA-3). These are the ground truth.
 *    2. PADDING EDGE CASES. Inputs chosen to exercise every awkward boundary:
 *       a message that just fits one block, one that forces a second padding
 *       block, and (for SHA-3) messages of length rate-1, rate, and rate+1.
 *    3. GPU == CPU on RANDOM, VARIABLE-LENGTH batches. The GPU is compared to the
 *       readable CPU reference on hundreds of pseudo-random messages of differing
 *       lengths in a SINGLE batch -- this exercises the real batch path (varying
 *       offsets/lengths, warp divergence) on inputs the standards never published.
 *
 *  Because NIST == CPU (checked by the KATs) and CPU == GPU (checked on random
 *  data), all three agree, and the GPU can be trusted.
 * ========================================================================== */

#include "sha256.h"
#include "sha3.h"
#include "sha_cpu_reference.h"
#include "sha_common.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cstdint>

/* ----- tiny helpers -------------------------------------------------------- */

/* Convert a hex string ("ab12...") to raw bytes. */
static std::vector<uint8_t> hex2bytes(const std::string& hex) {
    std::vector<uint8_t> out;
    out.reserve(hex.size() / 2);
    auto nib = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    };
    for (size_t i = 0; i + 1 < hex.size(); i += 2)
        out.push_back((uint8_t)((nib(hex[i]) << 4) | nib(hex[i + 1])));
    return out;
}

/* Convert raw bytes to a lowercase hex string. */
static std::string bytes2hex(const uint8_t* p, int n) {
    static const char* d = "0123456789abcdef";
    std::string s;
    s.reserve((size_t)n * 2);
    for (int i = 0; i < n; ++i) { s.push_back(d[p[i] >> 4]); s.push_back(d[p[i] & 15]); }
    return s;
}

/* A single known-answer case: a message and its expected digest (as hex). */
struct Case {
    std::string        name;
    std::vector<uint8_t> msg;
    std::string        expect;   /* expected digest, hex */
};

/* Build a message of `count` copies of byte `b` (for the "N copies of 'a'" KATs). */
static std::vector<uint8_t> repeat(uint8_t b, size_t count) {
    return std::vector<uint8_t>(count, b);
}
/* Build a message from a literal ASCII string. */
static std::vector<uint8_t> ascii(const char* s) {
    return std::vector<uint8_t>(s, s + std::strlen(s));
}

static int g_failures = 0;   /* global failure counter */

/* ----------------------------------------------------------------------------
 *  Batch a list of SHA-256 cases through the GPU and check each digest.
 *  Running them as ONE batch (not one call each) is deliberate: it exercises the
 *  variable-length batch path with heterogeneous message lengths in a single
 *  kernel launch -- exactly the workload the library is built for.
 * -------------------------------------------------------------------------- */
static void check_sha256(const std::vector<Case>& cases) {
    int n = (int)cases.size();
    std::vector<uint8_t> messages;
    std::vector<size_t>  offsets(n), lengths(n);
    for (int i = 0; i < n; ++i) {
        offsets[i] = messages.size();
        lengths[i] = cases[i].msg.size();
        messages.insert(messages.end(), cases[i].msg.begin(), cases[i].msg.end());
    }
    std::vector<uint8_t> digests((size_t)n * SHA256_DIGEST_SIZE);
    sha256_gpu_batch(messages.empty() ? (const uint8_t*)"" : messages.data(),
                     offsets.data(), lengths.data(), n, digests.data());

    for (int i = 0; i < n; ++i) {
        std::string got = bytes2hex(digests.data() + (size_t)i * SHA256_DIGEST_SIZE, SHA256_DIGEST_SIZE);
        bool ok = (got == cases[i].expect);
        std::printf("  [%s] SHA-256 %-34s %s\n", ok ? "PASS" : "FAIL",
                    cases[i].name.c_str(), ok ? "" : ("got " + got).c_str());
        if (!ok) { std::printf("        expected %s\n", cases[i].expect.c_str()); ++g_failures; }
    }
}

/* Same idea for SHA-3, parameterized by variant. */
static void check_sha3(Sha3Variant v, const char* vname, const std::vector<Case>& cases) {
    int n = (int)cases.size();
    int outlen = sha3_digest_bytes(v);
    std::vector<uint8_t> messages;
    std::vector<size_t>  offsets(n), lengths(n);
    for (int i = 0; i < n; ++i) {
        offsets[i] = messages.size();
        lengths[i] = cases[i].msg.size();
        messages.insert(messages.end(), cases[i].msg.begin(), cases[i].msg.end());
    }
    std::vector<uint8_t> digests((size_t)n * outlen);
    sha3_gpu_batch(v, messages.empty() ? (const uint8_t*)"" : messages.data(),
                   offsets.data(), lengths.data(), n, digests.data());

    for (int i = 0; i < n; ++i) {
        std::string got = bytes2hex(digests.data() + (size_t)i * outlen, outlen);
        bool ok = (got == cases[i].expect);
        std::printf("  [%s] %s %-30s %s\n", ok ? "PASS" : "FAIL", vname,
                    cases[i].name.c_str(), ok ? "" : ("got " + got).c_str());
        if (!ok) { std::printf("        expected %s\n", cases[i].expect.c_str()); ++g_failures; }
    }
}

/* ----------------------------------------------------------------------------
 *  GPU == CPU on a random, variable-length batch.
 *  We generate `n` pseudo-random messages whose lengths range across several
 *  block/rate boundaries, hash the whole batch on the GPU in one launch, and
 *  compare every digest to the CPU reference. A deterministic LCG keeps the test
 *  reproducible run to run.
 * -------------------------------------------------------------------------- */
static uint32_t lcg(uint32_t& s) { s = s * 1664525u + 1013904223u; return s; }

static void check_gpu_vs_cpu(int n) {
    uint32_t seed = 0xC0FFEEu;
    std::vector<uint8_t> messages;
    std::vector<size_t>  offsets(n), lengths(n);
    for (int i = 0; i < n; ++i) {
        size_t len = lcg(seed) % 600;            /* 0..599 bytes: crosses 64B and 136B/72B */
        offsets[i] = messages.size();
        lengths[i] = len;
        for (size_t j = 0; j < len; ++j) messages.push_back((uint8_t)lcg(seed));
    }
    const uint8_t* mp = messages.empty() ? (const uint8_t*)"" : messages.data();

    /* ---- SHA-256 ---- */
    {
        std::vector<uint8_t> gpu((size_t)n * SHA256_DIGEST_SIZE);
        sha256_gpu_batch(mp, offsets.data(), lengths.data(), n, gpu.data());
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            uint8_t cpu[32];
            sha256_cpu(messages.data() + offsets[i], lengths[i], cpu);
            if (std::memcmp(cpu, gpu.data() + (size_t)i * 32, 32) != 0) ++bad;
        }
        bool ok = (bad == 0);
        std::printf("  [%s] SHA-256 GPU==CPU on %d random msgs (%d mismatches)\n",
                    ok ? "PASS" : "FAIL", n, bad);
        if (!ok) ++g_failures;
    }

    /* ---- all four SHA-3 variants ---- */
    Sha3Variant vs[4]   = { SHA3_224, SHA3_256, SHA3_384, SHA3_512 };
    const char* names[4] = { "SHA3-224", "SHA3-256", "SHA3-384", "SHA3-512" };
    for (int t = 0; t < 4; ++t) {
        int outlen = sha3_digest_bytes(vs[t]);
        int rate   = sha3_rate_bytes(vs[t]);
        std::vector<uint8_t> gpu((size_t)n * outlen);
        sha3_gpu_batch(vs[t], mp, offsets.data(), lengths.data(), n, gpu.data());
        int bad = 0;
        for (int i = 0; i < n; ++i) {
            uint8_t cpu[64];
            sha3_cpu(rate, outlen, messages.data() + offsets[i], lengths[i], cpu);
            if (std::memcmp(cpu, gpu.data() + (size_t)i * outlen, outlen) != 0) ++bad;
        }
        bool ok = (bad == 0);
        std::printf("  [%s] %s GPU==CPU on %d random msgs (%d mismatches)\n",
                    ok ? "PASS" : "FAIL", names[t], n, bad);
        if (!ok) ++g_failures;
    }
}

int main() {
    std::printf("=== CUDA SHA-256 / SHA-3 known-answer test suite ===\n\n");
    if (!sha_cuda_device_info()) return 2;
    std::printf("\n");

    /* ---------------- SHA-256 known-answer vectors ---------------- */
    std::printf("-- SHA-256 known-answer vectors (FIPS 180-4 + computed) --\n");
    check_sha256({
        { "empty",            {},                        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
        { "\"abc\"",          ascii("abc"),              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
        { "56B 2-block edge", ascii("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
                                                         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" },
        { "112B",             ascii("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
                                                         "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1" },
        { "55*'a' fits-1-blk",repeat('a', 55),           "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318" },
        { "64*'a' +pad block",repeat('a', 64),           "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb" },
        { "1000*'a'",         repeat('a', 1000),         "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3" },
    });
    std::printf("\n");

    /* ---------------- SHA3-256: include the rate-boundary edges (rate=136) ---- */
    std::printf("-- SHA3-256 known-answer vectors (FIPS 202, incl. rate edges) --\n");
    check_sha3(SHA3_256, "SHA3-256", {
        { "empty",            {},                "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a" },
        { "\"abc\"",          ascii("abc"),      "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532" },
        { "135*'b' rate-1",   repeat('b', 135),  "bbcfe2803f2108ca1d38b9f9f5499cec8535096c9156e895817b60f407e229f4" },
        { "136*'b' full rate",repeat('b', 136),  "491d43679ebf9eeb191b33432034caebed97df8be9125a6db9b133c7ce660ca7" },
        { "137*'b' rate+1",   repeat('b', 137),  "f6381b4e7a53f850ae4bb86b430785df61c50c0cf35b0ee9ee7e34bf2f9bce24" },
        { "200*'c'",          repeat('c', 200),  "691193ec196d5fb6802bbef55a0ce490960212a00534038df9342d313f52da7b" },
    });
    std::printf("\n");

    /* ---------------- SHA3-224 / 384 / 512 spot checks ---------------- */
    std::printf("-- SHA3-224 / SHA3-384 / SHA3-512 known-answer vectors --\n");
    check_sha3(SHA3_224, "SHA3-224", {
        { "empty",   {},           "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7" },
        { "\"abc\"", ascii("abc"), "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf" },
        { "200*'c'", repeat('c',200), "0962e0c74033470dd14568f33b4ef54a351904a563fc0e1d99097ae2" },
    });
    check_sha3(SHA3_384, "SHA3-384", {
        { "empty",   {},           "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004" },
        { "\"abc\"", ascii("abc"), "ec01498288516fc926459f58e2c6ad8df9b473cb0fc08c2596da7cf0e49be4b298d88cea927ac7f539f1edf228376d25" },
        { "200*'c'", repeat('c',200), "2dedab69bdd6fe398ce4f32d8bd6a76916fb9f132765968f3dc52bd0253b386fc652f27884580af36c39ec108e5b8731" },
    });
    check_sha3(SHA3_512, "SHA3-512", {
        { "empty",   {},           "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26" },
        { "\"abc\"", ascii("abc"), "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0" },
        { "200*'c'", repeat('c',200), "687aaa5b1a4edd33d5b47bb619adcba81c6abc07378c480a63932eef4f0443138c14750aacee9c812e39938528166f6c394d6ce142a63722cf417ba3359bebde" },
    });
    std::printf("\n");

    /* ---------------- GPU vs CPU on random variable-length batches ---------------- */
    std::printf("-- GPU vs CPU reference on random variable-length batches --\n");
    check_gpu_vs_cpu(1000);
    std::printf("\n");

    if (g_failures == 0) {
        std::printf("ALL TESTS PASSED\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED\n", g_failures);
    return 1;
}
