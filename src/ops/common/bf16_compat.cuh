#pragma once

#include <cuda_bf16.h>

// CUDA 13.x dropped the pre-sm_80 emulation paths for f32 -> bf16 conversions:
// __floats2bfloat162_rn emits a real `cvt.rn.bf16x2.f32`, which ptxas rejects
// below sm_80 (Turing has no bf16 units at all). Provide a manual
// round-to-nearest-even conversion instead: a bf16 is simply the top 16 bits
// of an f32, rounded by the standard magic-number RNE trick.
// Global scope so call sites in any namespace resolve it unqualified.
__device__ __forceinline__ unsigned ninfer_bf16_rne_bits(float f) {
    unsigned u = __float_as_uint(f);
    u += 0x7FFFu + ((u >> 16) & 1u); // round to nearest even on the low half
    return u >> 16;
}

__device__ __forceinline__ __nv_bfloat162 ninfer_f32x2_to_bf16x2_rn(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return __floats2bfloat162_rn(a, b);
#else
    const unsigned lo = ninfer_bf16_rne_bits(a);
    const unsigned hi = ninfer_bf16_rne_bits(b);
    const unsigned packed = (hi << 16) | lo; // .x = low half (a), .y = high half (b)
    return *reinterpret_cast<const __nv_bfloat162*>(&packed);
#endif
}
