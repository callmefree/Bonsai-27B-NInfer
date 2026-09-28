#pragma once

#include "ops/common/memory.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace ninfer::ops {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// Forward declaration so the sm_75 mma_bf16 branch below can dispatch to mma_f16
// (defined further down) without reordering the whole file.
__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1);

// k8 f16 MMA with f32 accumulation. Turing supports only the m16n8k8 shape of
// the m16n8 family (m16n8k16 - even the f16 flavor - requires sm_80).
__device__ __forceinline__ void mma_f16_k8(float& c0, float& c1, float& c2, float& c3,
                                           unsigned a0, unsigned a1, unsigned b0) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(b0));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// Turing (sm_75) has no bf16 tensor core. Convert the bf16 operands to fp16 and
// dispatch to the fp16 MMA, keeping f32 accumulation. Signature-compatible with
// the Ampere bf16 path, so every mma_bf16() call site works unchanged on sm_75.
// NOTE: the cuda_bf16.h bf16<->fp16 conversion intrinsics are themselves guarded
// to __CUDA_ARCH__ >= 800 in the CUDA headers, so convert manually: bf16 is the
// top 16 bits of an f32, widen by shift, then round into fp16 with the fp16
// intrinsics that exist on every arch >= sm_53.
__device__ __forceinline__ unsigned bf16x2_to_fp16x2(unsigned v) {
    const float lo = __uint_as_float((v & 0xFFFFu) << 16);
    const float hi = __uint_as_float((v >> 16) << 16);
    const __half2 h = __halves2half2(__float2half_rn(lo), __float2half_rn(hi));
    return *reinterpret_cast<const unsigned*>(&h);
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    mma_f16(c0, c1, c2, c3, bf16x2_to_fp16x2(a0), bf16x2_to_fp16x2(a1),
            bf16x2_to_fp16x2(a2), bf16x2_to_fp16x2(a3), bf16x2_to_fp16x2(b0),
            bf16x2_to_fp16x2(b1));
}
#else
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#endif

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // sm_75 has no m16n8k16 shape (even f16): split k16 into two k8 MMAs.
    // Fragment halves align: (a0,a1,b0) cover k0-7, (a2,a3,b1) cover k8-15.
    mma_f16_k8(c0, c1, c2, c3, a0, a1, b0);
    mma_f16_k8(c0, c1, c2, c3, a2, a3, b1);
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

// FP16-accumulate variant: full-rate on consumer parts where f32-acc HMMA runs at
// half rate. D/C are two packed half2 registers: c0 = rows 0-7 column pair,
// c1 = rows 8-15 column pair of the same fragment the f32 variant returns in c0..c3.
__device__ __forceinline__ void mma_f16_f16acc(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                               unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // sm_75: no k16 shape; split into two k8 MMAs with fp16 accumulation.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a2), "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // Turing DOES have integer tensor cores -- just not in an m16n8 shape. ptxas accepts
    // mma.sync.m8n8k16.row.col.s32.s8.s8.s32 at sm_75 and rejects m16n8k16 / m16n8k32 .s8
    // with "requires .target sm_80 or higher" (measured: .github/workflows/sm75-imma-probe.yml).
    //
    // m16n8k32 is exactly 4x m8n8k16 -- M splits into the two 8-row halves, K into the two
    // 16-deep halves -- so it is bridged here with four instructions and the caller sees no
    // difference at all:
    //
    //   A: a0 = rows 0-7  k 0..15    a2 = rows 0-7  k 16..31
    //      a1 = rows 8-15 k 0..15    a3 = rows 8-15 k 16..31
    //   B: b0 = k 0..15              b1 = k 16..31
    //   C: c0,c1 = rows 0-7          c2,c3 = rows 8-15
    //
    // The split is not guesswork: it is read off the callers' own ldmatrix addressing.
    // ternary_rowsplit_mma_s8.cuh loads A with ldmatrix_x4 at a_matrix = lane >> 3,
    // a_row = (lane & 7) + ((a_matrix & 1) << 3), a_col = (a_matrix >> 1) * 16 -- i.e. matrix0
    // is rows 0-7 at byte 0, matrix1 rows 8-15 at byte 0, matrix2 rows 0-7 at byte 16, matrix3
    // rows 8-15 at byte 16 -- and stores c0,c1 to row_lo / c2,c3 to row_hi. B is ldmatrix_x2
    // with b_col = ((lane >> 3) & 1) * 16, i.e. b0 = k 0..15 and b1 = k 16..31.
    //
    // s32 accumulation is exact and associative, so the four-instruction sum is bit-identical
    // to the single Ampere instruction. Still, no SASS check can prove the register split, so
    // apps/sm75_mma_selftest.cu re-derives the products on the CUDA cores and demands equality.
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a2), "r"(b1));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 "
                 "{%0,%1}, {%2}, {%3}, {%0,%1};\n"
                 : "+r"(c2), "+r"(c3)
                 : "r"(a3), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
    // SM120 (Blackwell): the unified f8f6f4 kind covers e4m3 x e4m3.
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
#else
    // sm_89 (Ada): FP8 mma.sync carries no .kind modifier (PTX ISA 7.8+);
    // identical m16n8k32 fragment layout (A = 4 x .b32, B = 2 x .b32, C = 4 x .f32).
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
#endif
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#else
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
    // FP8 MMA requires sm_89+; the fp8 variant kernels are never dispatched on
    // this arch (bf16/w8/ternary paths cover them). Trap loudly if reached.
    __trap();
}
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#else
// tf32 MMA requires sm_80+. Unlike mma_s8 / mma_fp8_e4m3, this one is NOT dead code on
// Turing: the GDN chunked prefill kernels reach it on every prefill (chunked/state_passing.cuh
// x2, output.cuh x1, prepare_wy_wu.cuh x3 -- see PORT_sm75_TURING.md risk P0-B), so trapping
// here means "prefill dies on the first chunk". Bridge it to the fp16 MMA instead, exactly the
// way mma_bf16() above does.
//
// The operands are bf16 values the callers widened to fp32, so rounding them back into fp16
// keeps every mantissa bit they carry (bf16 8 -> fp16 10); tf32 would have kept 10 as well.
// Fragment translation (m16n8k8 tf32 -> m16n8k8 fp16), using the callers' own A layout
//   a0 = (row, k)  a1 = (row+8, k)  a2 = (row, k+4)  a3 = (row+8, k+4)
//   b0 = (k, col)  b1 = (k+4, col)
// against the fp16 layout (one register per row, two k per register):
//   regA0 = {a0, a2}   regA1 = {a1, a3}   regB = {b0, b1}
// Slot i of A is multiplied by slot i of B, so the k index the hardware assigns to a slot is
// irrelevant -- only the A/B pairing has to hold, and it does.
// Range caveat: fp16 tops out at 65504 and flushes under 6e-5, where tf32 keeps bf16's range.
// The operands here are post-RMSNorm q/k/v/w/u, orders of magnitude inside that window.
// Validated bit-exactly against an fp32 CUDA-core reference by apps/sm75_mma_selftest.cu --
// run it on the target GPU (gate 0) before trusting prefill PPL.
__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    const __half2 ha0 = __halves2half2(__float2half_rn(__uint_as_float(a0)),
                                       __float2half_rn(__uint_as_float(a2)));
    const __half2 ha1 = __halves2half2(__float2half_rn(__uint_as_float(a1)),
                                       __float2half_rn(__uint_as_float(a3)));
    const __half2 hb  = __halves2half2(__float2half_rn(__uint_as_float(b0)),
                                       __float2half_rn(__uint_as_float(b1)));
    mma_f16_k8(c0, c1, c2, c3, *reinterpret_cast<const unsigned*>(&ha0),
               *reinterpret_cast<const unsigned*>(&ha1),
               *reinterpret_cast<const unsigned*>(&hb));
}
#endif

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
}

} // namespace ninfer::ops
