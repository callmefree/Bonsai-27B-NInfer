#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

enum class Cache { ca, cg };

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

// ---------------------------------------------------------------------------
// Turing (sm_75) compatibility fallback
// sm_75 has no cp.async / pipeline async copy (those are Ampere+). Provide a
// synchronous, register-mediated global->shared copy so the existing kernels
// (written for Ampere+) keep compiling and running on Turing. The fallback is
// byte-wise (correct, but no async overlap) — throughput is lower, which is
// acceptable for the sm_75 port. sm_89 / sm_120 keep the fast cp.async path.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
template <int Bytes>
__device__ __forceinline__ void cp_async_fallback(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes >= 1 && Bytes <= 16, "cp_async_fallback supports 1..16 bytes");
    const unsigned char* src = reinterpret_cast<const unsigned char*>(gmem_src);
    unsigned char* dst = reinterpret_cast<unsigned char*>(smem_dst);
    #pragma unroll
    for (int i = 0; i < Bytes; ++i) dst[i] = src[i];
}
template <int Bytes>
__device__ __forceinline__ void cp_async_zfill_fallback(void* smem_dst, const void* gmem_src,
                                                       int src_bytes) {
    static_assert(Bytes >= 1 && Bytes <= 16, "cp_async_zfill_fallback supports 1..16 bytes");
    const unsigned char* src = reinterpret_cast<const unsigned char*>(gmem_src);
    unsigned char* dst = reinterpret_cast<unsigned char*>(smem_dst);
    #pragma unroll
    for (int i = 0; i < Bytes; ++i) dst[i] = (i < src_bytes) ? src[i] : 0;
}
#endif // sm_75 fallback helpers

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    cp_async_fallback<Bytes>(smem_dst, gmem_src);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
#endif
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    cp_async_zfill_fallback<Bytes>(smem_dst, gmem_src, src_bytes);
#else
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
#endif
}

__device__ __forceinline__ void cp_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // synchronous copies need no commit
#else
    asm volatile("cp.async.commit_group;\n");
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __syncthreads();
#else
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    cp_async_fallback<Bytes>(smem_dst, gmem_src);
#else
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
#endif
}

__device__ __forceinline__ void pipe_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    // no-op
#else
    __pipeline_commit();
#endif
}

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    __syncthreads();
#else
    __pipeline_wait_prior(Groups);
#endif
}

// ---------------------------------------------------------------------------
// sm_75 lacks the Ampere+ __reduce_*_sync warp intrinsics. Emulate
// __reduce_max_sync with a shfl_xor warp reduction so the sampler (and any
// other sm_75-compiled kernel) keeps building on Turing.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ unsigned int __reduce_max_sync_fallback(unsigned int mask,
                                                                  unsigned int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned int other = __shfl_xor_sync(mask, val, offset);
        val = (val > other) ? val : other;
    }
    return val;
}
#define __reduce_max_sync(mask, val) __reduce_max_sync_fallback((mask), (val))
#endif // sm_75 __reduce_max_sync fallback

} // namespace ninfer::ops
