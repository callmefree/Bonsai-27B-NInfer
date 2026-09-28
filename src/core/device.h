#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

// Streaming-multiprocessor count of the device this process runs on, queried once and cached.
// Launch geometry that deliberately fills exactly one resident wave reads the count from here
// instead of transcribing a per-part literal; the product runs one resident model on one device,
// so a single cached query is the whole device set.
int device_sm_count();

// Compile-time mirror of device_sm_count() for the architecture this build targets. __device__
// launch policies cannot query the runtime, and the host launcher that must reproduce such a
// policy exactly has to agree with it at compile time; those two sites use this constant, every
// other site uses device_sm_count().
// NOTE: NINFER_SM89 is a *kernel-family* switch in this tree (the i8 / rk4v4 kernels ported from
// sergiuszm/ninfer-4090 plus the 48 KiB-capped w8 schedules), not an architecture switch: Turing
// shares the 48 KiB static shared-memory wall with sm_86/sm_89, so the sm_75 build deliberately
// keeps NINFER_SM89=1. What is *not* valid for sm_75 is the SM count below, because 128 is the
// RTX 4090 / 5080 figure. NINFER_SM75 overrides only that, keeping every kernel path intact.
#if defined(NINFER_SM75)
#if defined(NINFER_SM75_SM_COUNT)
inline constexpr int kTargetSmCount = NINFER_SM75_SM_COUNT;
#else
inline constexpr int kTargetSmCount = 56; // NVIDIA Tesla T10 (TU102-890, 56 SM)
#endif
#elif defined(NINFER_SM89)
inline constexpr int kTargetSmCount = 128; // NVIDIA GeForce RTX 4090
#elif defined(NINFER_SM86)
inline constexpr int kTargetSmCount = 82; // NVIDIA GeForce RTX 3090
#else
#error "NInfer requires NINFER_SM75, NINFER_SM86 or NINFER_SM89"
#endif

// Non-owning execution facts passed to Ops whose launch policy depends on physical device
// capacity. DeviceContext remains the owner and authoritative source of both values.
struct DeviceExecutionView {
    cudaStream_t stream               = nullptr;
    std::int32_t multiprocessor_count = 0;
};

struct DeviceContext {
    int device                   = 0;
    cudaStream_t stream          = nullptr;
    cudaStream_t transfer_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    void bind_to_current_thread() const;
    void bind_to_current_thread_noexcept() const noexcept;
    int compute_capability() const noexcept;
    int multiprocessor_count() const noexcept;
    DeviceExecutionView execution_view() const noexcept;
    std::size_t total_vram() const noexcept;
    const char* sync_mode() const;
    void synchronize() const;
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    CudaEventTimer(const DeviceContext& ctx, cudaStream_t stream);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

// Reusable non-timing event for worker-driven asynchronous control transactions. The owning
// component records it after enqueueing one transfer batch and polls it from later boundaries.
class CudaCompletionEvent {
public:
    explicit CudaCompletionEvent(const DeviceContext& ctx);
    ~CudaCompletionEvent();

    CudaCompletionEvent(const CudaCompletionEvent&)            = delete;
    CudaCompletionEvent& operator=(const CudaCompletionEvent&) = delete;
    CudaCompletionEvent(CudaCompletionEvent&& other) noexcept;
    CudaCompletionEvent& operator=(CudaCompletionEvent&& other) noexcept;

    void record(cudaStream_t stream);
    void wait(cudaStream_t stream) const;
    [[nodiscard]] bool ready() const;
    void synchronize() const;

private:
    int device_        = 0;
    cudaEvent_t event_ = nullptr;
};

} // namespace ninfer
