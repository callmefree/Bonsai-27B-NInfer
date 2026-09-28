// MODIFIED for the NInfer ternary port (Ternary Bonsai 2 27B on NInfer / Ada sm_89).
// This file differs from upstream NInfer; see patches/ in the release bundle
// for the change list, rebuild steps and required verification.
#include "ops/linear/ternary/ternary_dispatch.h"

#include "ops/linear/ternary/ternary_rotation.h"
#include "ops/linear/ternary/ternary_rowsplit_storage.cuh"

#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

TernaryLaunch select_ternary_launch(std::int32_t n, std::int32_t k, std::int32_t t,
                                    LinearPolicy policy) {
    if (n <= 0 || k <= 0 || t <= 0) {
        throw std::invalid_argument("ternary linear: unsupported shape or T");
    }
    // The reference kernel decodes whole 128-weight groups, so K has to be a whole number
    // of them. Every width the model uses (5120, 6144, 10240, 17408) satisfies this, and
    // row_split_geometry() pads to the same 128 boundary, so no padding groups exist.
    if (k % PTQ1RowSplitStorage::kGroupK != 0) {
        throw std::invalid_argument("ternary linear: K must be a multiple of the group size");
    }
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    // Both ternary formats take the same schedule; the atom is chosen from w.qtype inside
    // the launch. Decode (T = 1) and small prefill use different token tiles.
    return t == 1 ? launch_ternary_gemm_t1 : launch_ternary_gemm_t8;
}

void ternary_dispatch_basis_strided(const Tensor& x_folded, const Weight& w, Tensor& out,
                                    std::int32_t out_row_stride, LinearPolicy policy,
                                    cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x_folded.ne[1], policy);
    // No workspace on this entry point: the caller already folded the activation, and may hand in
    // the int8 scratch it allocated from its own arena. With an empty scratch these calls stay on
    // the bf16 rungs, which is what they did before the s8 rung existed.
    launch(x_folded, w, out, out_row_stride, stream, TernaryS8Scratch{});
}

void ternary_dispatch_basis(const Tensor& x_folded, const Weight& w, Tensor& out,
                            LinearPolicy policy, cudaStream_t stream) {
    ternary_dispatch_basis_strided(x_folded, w, out, w.n, policy, stream);
}

void ternary_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                      WorkspaceArena* workspace, cudaStream_t stream) {
    const TernaryLaunch launch = select_ternary_launch(w.n, w.k, x.ne[1], policy);

    if (!ternary_rotation_enabled()) {
        // NINFER_TERNARY_HADAMARD=0: run the GEMM against the raw activation so the full forward
        // pass (and therefore the M4 decode speed) is measurable. The result is numerically
        // meaningless -- the activation is in the wrong basis -- which is the point: it separates
        // "the ternary decode is broken" from "the rotation is broken" without a rebuild.
        launch(x, w, out, w.n, stream, TernaryS8Scratch{});
        return;
    }
    if (workspace == nullptr) {
        // Name the shape in the message: a folded ternary weight reaching the workspace-free
        // entry point is a call-site wiring gap, and the shape is what identifies the weight.
        throw std::invalid_argument(
            "ternary linear: the folded basis needs a rotation workspace [N=" +
            std::to_string(w.n) + ", K=" + std::to_string(w.k) + ", T=" +
            std::to_string(x.ne[1]) + ", qtype=" + std::to_string(static_cast<int>(w.qtype)) + "]");
    }

    // Scoped: the scratch is handed back when this op returns, so it does not accumulate across
    // the (many) graph constructions of one load. Sequential reuse on one stream is safe.
    // folded_activation() rejects a ternary weight with no sign block, so an unfolded artifact
    // fails loudly here instead of silently multiplying by unfolded weights.
    auto scope              = workspace->scope();
    const Tensor activation = folded_activation(x, w, *workspace, stream);

    // int8 rung scratch (#14): one int8 code row per token plus one fp32 scale per token. Taken
    // from the same arena the rotation just used, and counted by ternary_rotation_workspace_bytes()
    // so the planner sizes the arena for it -- allocating it lazily inside the launch would be
    // illegal, because this op runs inside captured CUDA graphs. Below the threshold the scratch is
    // left empty and the launch falls through to the bf16 rungs.
    TernaryS8Scratch scratch{};
    // Allocated only when the rung will actually run: without this the arena would reserve
    // k*T + T bytes per ternary linear for a kernel that never launches, and 16 GB of T10 VRAM
    // is the binding constraint. (This used to be a sm_75-specific skip, back when the rung was
    // hard-off there; it is now just the honest condition.)
    if (ternary_s8_enabled() && x.ne[1] >= kTernaryS8MinTokens) {
        const DeviceSpan codes  = workspace->alloc_bytes(ternary_s8_codes_bytes(w.k, x.ne[1]));
        const DeviceSpan scales = workspace->alloc_bytes(ternary_s8_scales_bytes(x.ne[1]));
        scratch.codes           = static_cast<std::int8_t*>(codes.data);
        scratch.scales          = static_cast<float*>(scales.data);
    }
    launch(activation, w, out, w.n, stream, scratch);
}

} // namespace ninfer::ops::detail
