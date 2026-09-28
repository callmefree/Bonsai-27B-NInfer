#pragma once

// Host-safe half of the int8 rung (#14): the token threshold, the scratch descriptor and its sizes.
//
// Separate from ternary_rowsplit_mma_s8.cuh on purpose: that header carries the CUDA kernel and
// device-only syntax, and it must never be included by a host translation unit. ternary_dispatch.cpp
// and ternary_rotation.cpp are host TUs, and including the kernel header there drags
// cuda_pipeline_helpers.h into MSVC, which fails with "unexpected volatile" -- measured, that is
// exactly how the first build of this rung broke.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace ninfer::ops::detail {

// A/B and rollback switch for the int8 rung, in the same style as NINFER_TERNARY_MMA / _HADAMARD.
// It is needed because this rung CHANGES THE NUMERICS (activations get quantized to int8), so it must
// be comparable inside one binary and switchable off without a rebuild:
//   NINFER_TERNARY_S8=0   -> stay on the bf16 rungs
// Read once, because the choice decides which kernel enters the captured CUDA graph.
[[nodiscard]] inline bool ternary_s8_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_TERNARY_S8");
        return value == nullptr || std::string(value) != "0";
    }();
    return enabled;
}

// Why this rung is live on sm_75 and not hard-off:
//
// It used to be disabled here, on the reasoning that "Turing has no s8 MMA". That reasoning was
// wrong in an important way -- it was only ever checked against the m16n8k32 shape. ptxas accepts
// mma.sync.m8n8k16.row.col.s32.s8.s8.s32 at sm_75, and mma_s8() now bridges m16n8k32 with four of
// them (see mma.cuh). Turing's IMMA is real; it is just shaped 8x8x16.
//
// Cost of not doing this: prefill takes the bf16 wide rung instead, measured at 1.20-1.30x slower
// than s8 by the author. Cost of doing it: a fragment split that no static check can verify, which
// is why apps/sm75_mma_selftest.cu has to pass on the target before this rung is trusted.
//
// NINFER_TERNARY_S8=0 remains the A/B switch -- it is the only way to tell "the numerics differ
// because activations are int8" apart from "the bridge is wrong", without a rebuild.

// First token count at which the int8 rung beats both bf16 rungs. Measured on the real shape mix
// (tscale_bench, 2026-09-20, quantization pass included): T=32 is a tie (43.25 vs 42.49 small_t),
// T=40 s8 wins by 1.20x, T=48 by 1.22x, T=64 by 1.30x; below T=32 the small-t rung's 10.2 ms per
// 8-token pass is still cheaper.
inline constexpr int kTernaryS8MinTokens = 33;

// Activation-quantization scratch for the int8 rung: one int8 code row per token (token-major, the
// same layout as the activation it is built from) plus one fp32 scale per token.
//
// The caller owns this memory. The ternary linear op runs inside captured CUDA graphs, so a lazy
// cudaMalloc at launch time is illegal; it must come from the op's workspace arena, and the arena's
// capacity must count it (see ternary_rotation_workspace_bytes()).
struct TernaryS8Scratch {
    std::int8_t* codes  = nullptr;
    float*       scales = nullptr;
};

inline constexpr std::size_t ternary_s8_codes_bytes(std::int32_t k, std::int32_t tokens) {
    return static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens);
}

inline constexpr std::size_t ternary_s8_scales_bytes(std::int32_t tokens) {
    return static_cast<std::size_t>(tokens) * sizeof(float);
}

} // namespace ninfer::ops::detail
