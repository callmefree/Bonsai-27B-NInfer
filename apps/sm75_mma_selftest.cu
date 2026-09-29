// Standalone sm_75 MMA bridge self-test.
//
// Turing has no tf32 MMA and no m16n8 int8 MMA. mma.cuh bridges what it can:
//   mma_bf16()      -> fp16 MMA      (the author's bridge, used by every bf16 kernel)
//   mma_tf32_bits() -> fp16 MMA      (added for the sm_75 port: GDN chunked prefill reaches it)
//   mma_s8()        -> 4x m8n8k16    (added for the sm_75 port: Turing's IMMA is real, it is
//                                     just shaped 8x8x16 instead of 16x8x32)
//
// A bridge that packs the fragment registers the wrong way compiles, launches, and returns
// plausible numbers -- and silently corrupts every prefill. SASS cannot catch it. So this probe
// re-derives the same products on the CUDA cores in fp32 and demands BIT-EXACT equality.
//
// The operands are built so that nothing can round: every A/B value is a small multiple of
// 1/8 (exact in fp16, bf16 and tf32 alike), every product is exact in fp32, and every partial
// sum is exact in fp32. That removes the tolerance argument entirely -- a wrong packing shows
// up as "value is not a reference entry", not as "error 3e-2, is that OK?".
//
// Five mma checks, each scored three ways:
//   bad        produced values that match NO reference entry  -> the decisive one. A row/k
//              misassignment lands on a mixed product that is not any true entry.
//   unmatched  reference entries no thread produced           -> informational; duplicates in
//              the reference matrix can make this nonzero without anything being wrong.
//   sum        sum of all produced values vs sum of the reference -> catches a k-pairing error
//              that happens to keep values inside the reference set.
//
// Check 6 is not an mma check: it proves the PRMT byte-permute V-dequant magic in
// causal_prompt_i8_dequant_f16x8() (the function the sm_75 int8-KV prefill kernel calls)
// is bit-identical to the plain I2F path for all 256 signed codes x 4 scales.
//
// Run it on the target GPU (gate 0) before trusting prefill PPL.

#include "ops/common/mma.cuh"
#include "ops/softmax_attention/dense/causal_cache/prompt_i8.cuh"

#include <cuda_runtime.h>
#include <cstdio>

namespace {

using namespace ninfer::ops;

constexpr int kThreads = 32;
constexpr int kChecks  = 5;

// Reference matrices: 16x8 (k=8 checks) and 16x8 via k=16 (bf16 check).
__device__ __forceinline__ float gen_a(int flat) {
    return 0.25f * static_cast<float>(flat % 13) + 0.5f;
}
__device__ __forceinline__ float gen_b(int flat) {
    return 0.125f * static_cast<float>(flat % 7) + 0.25f;
}

// int8 operands for the mma_s8 check. Small signed values so every product and every partial
// sum is exact in int32 and in fp32: |a| <= 6, |b| <= 3, k = 32 -> |sum| <= 576.
__device__ __forceinline__ int gen_a8(int flat) { return (flat % 13) - 6; }
__device__ __forceinline__ int gen_b8(int flat) { return (flat % 7) - 3; }

// Four signed bytes into one fragment register, lowest k in the lowest byte.
__device__ __forceinline__ unsigned pack_s8x4(int v0, int v1, int v2, int v3) {
    return (static_cast<unsigned>(v0 & 0xFF)) | ((static_cast<unsigned>(v1 & 0xFF)) << 8) |
           ((static_cast<unsigned>(v2 & 0xFF)) << 16) | ((static_cast<unsigned>(v3 & 0xFF)) << 24);
}

// bf16 is the top 16 bits of the fp32. Exact for the values above (multiples of 1/8 -> the low
// 16 mantissa bits are already zero). The cuda_bf16.h conversions are guarded to sm_80 in the
// CUDA headers, so do it by hand, the same way mma.cuh does.
__device__ __forceinline__ unsigned bf16_bits(float v) { return __float_as_uint(v) >> 16; }
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    return (bf16_bits(hi) << 16) | bf16_bits(lo);
}
__device__ __forceinline__ unsigned pack_fp16x2(float lo, float hi) {
    const __half2 h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return *reinterpret_cast<const unsigned*>(&h);
}

__global__ void sm75_mma_selftest_kernel(float* err_out, int* bad_out, int* unmatched_out,
                                         float* sum_c_out, float* sum_ref_out) {
    __shared__ float ref[128];
    __shared__ int   hit[128];
    __shared__ float sum_c;
    __shared__ float sum_ref;

    const int lane   = threadIdx.x;
    const int lane_g = lane >> 2;   // row / column selector
    const int lane_t = lane & 3;    // k selector

    for (int check = 0; check < kChecks; ++check) {
        const int kdim = (check == 3) ? 16 : ((check == 4) ? 32 : 8);
        for (int i = lane; i < 128; i += kThreads) {
            const int r = i / 8;
            const int c = i % 8;
            float s     = 0.0f;
            if (check == 4) {
                for (int k = 0; k < kdim; ++k) {
                    s += static_cast<float>(gen_a8(r * 32 + k) * gen_b8(k * 8 + c));
                }
            } else {
                for (int k = 0; k < kdim; ++k) { s += gen_a(r * 16 + k) * gen_b(k * 8 + c); }
            }
            ref[i] = s;
        }
        for (int i = lane; i < 128; i += kThreads) { hit[i] = 0; }
        if (lane == 0) {
            sum_c   = 0.0f;
            sum_ref = 0.0f;
            for (int i = 0; i < 128; ++i) { sum_ref += ref[i]; }
        }
        __syncthreads();

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;
        if (check == 3) {
            // mma_bf16, m16n8k16: 4 A registers (8 bf16) + 2 B registers (4 bf16).
            // Hypothesis under test: one register holds one row's two k values,
            // k0 = 2*lane_t, and (a0,a1,b0) covers k0..7 while (a2,a3,b1) covers k8..15.
            const int k0   = 2 * lane_t;
            const int row  = lane_g;
            const int col  = lane_g;
            const unsigned ra0 = pack_bf16x2(gen_a(row * 16 + k0), gen_a(row * 16 + k0 + 1));
            const unsigned ra1 =
                pack_bf16x2(gen_a((row + 8) * 16 + k0), gen_a((row + 8) * 16 + k0 + 1));
            const unsigned ra2 = pack_bf16x2(gen_a(row * 16 + k0 + 8), gen_a(row * 16 + k0 + 9));
            const unsigned ra3 =
                pack_bf16x2(gen_a((row + 8) * 16 + k0 + 8), gen_a((row + 8) * 16 + k0 + 9));
            const unsigned rb0 = pack_bf16x2(gen_b(k0 * 8 + col), gen_b((k0 + 1) * 8 + col));
            const unsigned rb1 = pack_bf16x2(gen_b((k0 + 8) * 8 + col), gen_b((k0 + 9) * 8 + col));
            mma_bf16(c0, c1, c2, c3, ra0, ra1, ra2, ra3, rb0, rb1);
        } else if (check == 4) {
            // mma_s8, m16n8k32 bridged as 4x m8n8k16.
            // Hypothesis under test: a0 = rows 0-7 k0..15, a1 = rows 8-15 k0..15,
            // a2 = rows 0-7 k16..31, a3 = rows 8-15 k16..31; b0 = k0..15, b1 = k16..31;
            // (c0,c1) = rows 0-7, (c2,c3) = rows 8-15, two columns each.
            const int k0  = 4 * lane_t;
            const int row = lane_g;
            const int col = lane_g;
            const unsigned ra0 =
                pack_s8x4(gen_a8(row * 32 + k0), gen_a8(row * 32 + k0 + 1),
                          gen_a8(row * 32 + k0 + 2), gen_a8(row * 32 + k0 + 3));
            const unsigned ra1 =
                pack_s8x4(gen_a8((row + 8) * 32 + k0), gen_a8((row + 8) * 32 + k0 + 1),
                          gen_a8((row + 8) * 32 + k0 + 2), gen_a8((row + 8) * 32 + k0 + 3));
            const unsigned ra2 =
                pack_s8x4(gen_a8(row * 32 + k0 + 16), gen_a8(row * 32 + k0 + 17),
                          gen_a8(row * 32 + k0 + 18), gen_a8(row * 32 + k0 + 19));
            const unsigned ra3 =
                pack_s8x4(gen_a8((row + 8) * 32 + k0 + 16), gen_a8((row + 8) * 32 + k0 + 17),
                          gen_a8((row + 8) * 32 + k0 + 18), gen_a8((row + 8) * 32 + k0 + 19));
            const unsigned rb0 = pack_s8x4(gen_b8(k0 * 8 + col), gen_b8((k0 + 1) * 8 + col),
                                           gen_b8((k0 + 2) * 8 + col), gen_b8((k0 + 3) * 8 + col));
            const unsigned rb1 =
                pack_s8x4(gen_b8((k0 + 16) * 8 + col), gen_b8((k0 + 17) * 8 + col),
                          gen_b8((k0 + 18) * 8 + col), gen_b8((k0 + 19) * 8 + col));
            int i0 = 0, i1 = 0, i2 = 0, i3 = 0;
            mma_s8(i0, i1, i2, i3, ra0, ra1, ra2, ra3, rb0, rb1);
            c0 = static_cast<float>(i0);
            c1 = static_cast<float>(i1);
            c2 = static_cast<float>(i2);
            c3 = static_cast<float>(i3);
        } else {
            // m16n8k8: A is (row, k) / (row+8, k) / (row, k+4) / (row+8, k+4),
            //          B is (k, col) / (k+4, col).
            const float a0 = gen_a(lane_g * 16 + lane_t);
            const float a1 = gen_a((lane_g + 8) * 16 + lane_t);
            const float a2 = gen_a(lane_g * 16 + lane_t + 4);
            const float a3 = gen_a((lane_g + 8) * 16 + lane_t + 4);
            const float b0 = gen_b(lane_t * 8 + lane_g);
            const float b1 = gen_b((lane_t + 4) * 8 + lane_g);
            if (check == 0) {
                // The shipped bridge.
                mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1),
                              __float_as_uint(a2), __float_as_uint(a3), __float_as_uint(b0),
                              __float_as_uint(b1));
            } else if (check == 1) {
                // Packing P1: one register per row -> regA0 = {a0, a2}, regA1 = {a1, a3}.
                mma_f16_k8(c0, c1, c2, c3, pack_fp16x2(a0, a2), pack_fp16x2(a1, a3),
                           pack_fp16x2(b0, b1));
            } else {
                // Packing P2 (the alternative): regA0 = {a0, a1}, regA1 = {a2, a3}.
                mma_f16_k8(c0, c1, c2, c3, pack_fp16x2(a0, a1), pack_fp16x2(a2, a3),
                           pack_fp16x2(b0, b1));
            }
        }

        // Score: every produced value must be a bit-exact reference entry.
        const float c[4] = {c0, c1, c2, c3};
        for (int i = 0; i < 4; ++i) {
            float best  = 3.4e38f;
            int   found = -1;
            for (int j = 0; j < 128; ++j) {
                const float e = fabsf(c[i] - ref[j]);
                if (e < best) { best = e; }
                if (e == 0.0f) { found = j; }
            }
            err_out[(check * 128) + (lane * 4) + i] = best;
            if (found >= 0) { atomicAdd(&hit[found], 1); }
            atomicAdd(&sum_c, c[i]);
        }
        __syncthreads();

        if (lane == 0) {
            int bad = 0;
            int unmatched = 0;
            for (int j = 0; j < 128; ++j) {
                if (hit[j] == 0) { ++unmatched; }
            }
            // A value that matched nothing is recorded in err_out as nonzero; count those.
            for (int t = 0; t < kThreads; ++t) {
                for (int i = 0; i < 4; ++i) {
                    if (err_out[(check * 128) + (t * 4) + i] != 0.0f) { ++bad; }
                }
            }
            bad_out[check]       = bad;
            unmatched_out[check] = unmatched;
            sum_c_out[check]     = sum_c;
            sum_ref_out[check]   = sum_ref;
        }
        __syncthreads();
    }
}

const char* check_name(int i) {
    switch (i) {
    case 0: return "mma_tf32_bits()  shipped sm_75 bridge (fp16)";
    case 1: return "mma_f16_k8()     packing P1 regA={a0,a2},{a1,a3}";
    case 2: return "mma_f16_k8()     packing P2 regA={a0,a1},{a2,a3}";
    case 3: return "mma_bf16()       author's sm_75 bridge (k16)";
    case 4: return "mma_s8()         sm_75 bridge: m16n8k32 as 4x m8n8k16";
    default: return "?";
    }
}

// Check 6: the PRMT dequant magic vs the plain I2F path, bit-exact for every signed
// code and every scale. This calls the REAL function compiled into the sm_75 int8-KV
// prompt kernel (prompt_i8.cuh, upstream branch), not a copy of it.
constexpr int kPrmtScales = 4;

__global__ void sm75_prmt_dequant_kernel(const float* scales, int* mismatch_out) {
    const int code = static_cast<int>(threadIdx.x) - 128;   // all 256 signed codes
    for (int s = 0; s < kPrmtScales; ++s) {
        const __half scale = __float2half_rn(scales[s]);
        const __half2 s2   = __halves2half2(scale, scale);
        alignas(8) std::int8_t codes8[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) { codes8[i] = static_cast<std::int8_t>(code); }
        const int4 got = causal_prompt_i8_dequant_f16x8(codes8, scale);
        const unsigned got4[4] = {static_cast<unsigned>(got.x), static_cast<unsigned>(got.y),
                                  static_cast<unsigned>(got.z), static_cast<unsigned>(got.w)};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const __half2 ref2 = __hmul2(
                __floats2half2_rn(static_cast<float>(code), static_cast<float>(code)), s2);
            if (got4[i] != *reinterpret_cast<const unsigned*>(&ref2)) {
                atomicAdd(mismatch_out, 1);
            }
        }
    }
}

} // namespace

int main() {
    float* err          = nullptr;
    int*   bad          = nullptr;
    int*   unmatched    = nullptr;
    float* sum_c        = nullptr;
    float* sum_ref      = nullptr;
    if (cudaMalloc(&err, sizeof(float) * 128 * kChecks) != cudaSuccess ||
        cudaMalloc(&bad, sizeof(int) * kChecks) != cudaSuccess ||
        cudaMalloc(&unmatched, sizeof(int) * kChecks) != cudaSuccess ||
        cudaMalloc(&sum_c, sizeof(float) * kChecks) != cudaSuccess ||
        cudaMalloc(&sum_ref, sizeof(float) * kChecks) != cudaSuccess) {
        std::printf("sm75_mma_selftest: cudaMalloc failed\n");
        return 2;
    }

    sm75_mma_selftest_kernel<<<1, kThreads>>>(err, bad, unmatched, sum_c, sum_ref);
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        std::printf("sm75_mma_selftest: launch failed: %s\n", cudaGetErrorString(launch));
        return 2;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("sm75_mma_selftest: kernel aborted (device trap?)\n");
        return 2;
    }

    float h_err[128 * kChecks];
    int   h_bad[kChecks];
    int   h_unmatched[kChecks];
    float h_sum_c[kChecks];
    float h_sum_ref[kChecks];
    if (cudaMemcpy(h_err, err, sizeof(h_err), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_bad, bad, sizeof(h_bad), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_unmatched, unmatched, sizeof(h_unmatched), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(h_sum_c, sum_c, sizeof(h_sum_c), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(h_sum_ref, sum_ref, sizeof(h_sum_ref), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::printf("sm75_mma_selftest: memcpy failed\n");
        return 2;
    }

    std::printf("===== sm_75 MMA bridge self-test (fp32 CUDA-core reference, bit-exact) =====\n");
    int failures = 0;
    for (int i = 0; i < kChecks; ++i) {
        float worst = 0.0f;
        for (int j = 0; j < 128; ++j) {
            if (h_err[(i * 128) + j] > worst) { worst = h_err[(i * 128) + j]; }
        }
        const bool sum_ok = h_sum_c[i] == h_sum_ref[i];
        const bool ok     = (h_bad[i] == 0) && sum_ok;
        if (!ok) { ++failures; }
        std::printf("%-58s %s  bad=%d unmatched=%d worst_err=%.9g sum_ok=%s\n", check_name(i),
                    ok ? "PASS" : "FAIL", h_bad[i], h_unmatched[i], worst,
                    sum_ok ? "yes" : "no");
        if (!sum_ok) {
            std::printf("    sum produced=%.9g sum reference=%.9g\n", h_sum_c[i], h_sum_ref[i]);
        }
    }
    std::printf("\n");

    // Check 6: PRMT dequant magic vs I2F, all 256 codes x 4 scales, bit-exact.
    float* dq_scales    = nullptr;
    int*   dq_mismatch  = nullptr;
    if (cudaMalloc(&dq_scales, sizeof(float) * kPrmtScales) != cudaSuccess ||
        cudaMalloc(&dq_mismatch, sizeof(int)) != cudaSuccess) {
        std::printf("sm75_mma_selftest: cudaMalloc (dequant check) failed\n");
        cudaFree(err);
        cudaFree(bad);
        cudaFree(unmatched);
        cudaFree(sum_c);
        cudaFree(sum_ref);
        return 2;
    }
    const float h_dq_scales[kPrmtScales] = {1.0f, 0.125f, -0.875f, 7.5f};
    cudaMemcpy(dq_scales, h_dq_scales, sizeof(h_dq_scales), cudaMemcpyHostToDevice);
    cudaMemset(dq_mismatch, 0, sizeof(int));
    sm75_prmt_dequant_kernel<<<1, 256>>>(dq_scales, dq_mismatch);
    const cudaError_t dq_launch = cudaGetLastError();
    int h_dq = -1;
    if (dq_launch != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(&h_dq, dq_mismatch, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("sm75_mma_selftest: dequant check kernel failed: %s\n",
                    cudaGetErrorString(dq_launch));
        cudaFree(dq_scales);
        cudaFree(dq_mismatch);
        cudaFree(err);
        cudaFree(bad);
        cudaFree(unmatched);
        cudaFree(sum_c);
        cudaFree(sum_ref);
        return 2;
    }
    const bool dq_ok = (h_dq == 0);
    if (!dq_ok) { ++failures; }
    std::printf("%-58s %s  mismatches=%d (256 codes x 4 scales)\n",
                "prmt_dequant()   PRMT magic vs I2F, bit-exact", dq_ok ? "PASS" : "FAIL", h_dq);
    std::printf("\n");

    if (failures == 0) {
        std::printf("RESULT: all bridges bit-exact. The sm_75 tf32 and s8 bridges are safe to\n"
                    "        trust, and the ternary int8 prefill rung can stay enabled.\n");
    } else {
        std::printf("RESULT: %d check(s) failed. Reading the failures:\n"
                    "        - exactly one of P1/P2 passes -> mma.cuh packs the fp16 fragment\n"
                    "          the other way; flip it and rebuild.\n"
                    "        - mma_bf16 also fails -> the k16 hypothesis in this probe is wrong;\n"
                    "          fix the probe before drawing conclusions about mma.cuh.\n"
                    "        - mma_s8 fails but the fp32/bf16 checks pass -> the 4x m8n8k16\n"
                    "          fragment split is wrong. Set NINFER_TERNARY_S8=0 so prefill falls\n"
                    "          back to the bf16 rung, and fix the row/k split in mma.cuh.\n"
                    "        - prmt_dequant fails -> the byte-permute V-dequant diverged from\n"
                    "          I2F; the int8-KV prefill would produce wrong V values. Flip the\n"
                    "          >=750 block in causal_prompt_i8_dequant_f16x8 back to I2F and\n"
                    "          investigate before re-enabling the magic path.\n",
                    failures);
    }

    cudaFree(err);
    cudaFree(bad);
    cudaFree(unmatched);
    cudaFree(sum_c);
    cudaFree(sum_ref);
    cudaFree(dq_scales);
    cudaFree(dq_mismatch);
    return failures == 0 ? 0 : 1;
}
