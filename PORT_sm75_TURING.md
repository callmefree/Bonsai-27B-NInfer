# PORT: sm_75 / Tesla T10 (Turing) — 编译期移植记录

> 移植目标：把三元 Bonsai-2-27B 推理内核从 sm_89 / sm_120a 移植到 **Tesla T10（TU102, sm_75, 16GB）**。
> 状态（2026-09-28）：**编译期全绿**，运行期验证待上机。
> 分支：`sm75-port` ｜ 收官绿跑：CI run `36384008002` ｜ HEAD：`94635391ba`

## 0. 移植边界（先读这个）

1. **本分支只覆盖编译期**：nvcc 交叉编译 sm_75 + SASS 静态校验。GitHub Actions runner 无 GPU，运行期行为必须在真实 T10 上验证。
2. **单点咽喉原则**：所有架构差异收敛在 4 个文件（`mma.cuh` / `memory.cuh` / `bf16_compat.cuh` / `math.cuh`），40+ 调用文件近乎零改动。
3. **不做假模拟**：图灵物理不可行的内核（s8/fp8/tf32 MMA）编译放行但运行期 `__trap()`——宁可响亮崩溃，不给静默错误结果。

## 1. 七类架构障碍与修法

### 1.1 cp.async（Ampere+ 专属）— memory.cuh
Turing 无异步拷贝引擎。`cp_async` / `cp_commit` / `cp_wait` 加 `__CUDA_ARCH__ >= 800` 守卫，sm_75 走同步回退：`ld.global` + `st.shared` register 中转；`cp_commit` → no-op；`cp_wait` → `__syncthreads()`。CI 用 cuobjdump 校验 SASS 零 `CPASYNC`。

### 1.2 __reduce_max_sync — memory.cuh
warp 归约内建 sm_80+。回退：`__shfl_xor_sync` 树形归约（5 步覆盖 32 lane）。

### 1.3 bf16 MMA 无硬件 — mma.cuh
图灵 Tensor Core 只有 fp16/i8/u4 MMA，无 bf16。`mma_bf16` 加 `<800` 分支：手写 bf16x2→fp16x2 位级转换（shift 拓宽 + RNE），然后走 fp16 MMA 路径。三元权重 {-1,0,1} 天然无精度损失。

### 1.4 MMA 形状缺失 — mma.cuh
sm_80+ 的 m16n8k16(f16) / m16n8k32(s8) 形状图灵不存在（图灵上限 m16n8k8(f16) / m16n8k16(s8)）。`mma_f16` 拆 2×m16n8k8，`mma_s8` 拆 2×m16n8k16。

### 1.5 s8/fp8/tf32 物理不可行 — mma.cuh
fp8 MMA 需 sm_89+，tf32 需 sm_80+，s8 拆分后形状仍不满足 → sm_75 分支 `__trap()`。

### 1.6 f32→bf16 转换内建被删 — bf16_compat.cuh（新建）+ math.cuh
CUDA 13.x 删掉了所有 pre-sm_80 的 bf16 模拟路径：`__floats2bfloat162_rn`（71 处/38 文件）、`__float2bfloat16`（440 处/88 文件）在 sm_75 下全部真实发射 `cvt.rn.bf16x2.f32`。全局 shim `ninfer_f32x2_to_bf16x2_rn` / `ninfer_float_to_bf16`：sm_80+ 走原生，<800 走位级 RNE（bf16 = f32 高 16 位 + 0x7FFF + lsb 魔数舍入）。
**漏网之鱼**：`math.cuh` 的 `pack_bf16x2()` 内手写内联 asm `cvt.rn.bf16x2.f32`，前轮替换头文件内建时没扫到——靠 CI kept-PTX + `.loc` 源码映射精确定位后加守卫修复。教训：**grep 排查要连同内联 asm 一起扫，且 PTX 里指令带舍入修饰符（`cvt.rn.bf16x2.f32`），只有 ptxas 报错文本才缩写**。

### 1.7 GCC 拒绝 constexpr odr-use — bf16_gdn_gating_proj_kernels.cu
无捕获 lambda 内 odr-use 外层 constexpr 局部量（经 `std::min` 按引用）——MSVC 纵容、GCC 按 C++ 标准拒绝。上游在 Windows/MSVC 构建的潜伏可移植性 bug，Linux CI 首次曝光。修法：显式按值捕获 `[tuned = kTunedResidentCtasPerSm]`。

## 2. CI 基建（.github/workflows/sm75-compile.yml）

1. **容器**：`nvidia/cuda:13.1.0-devel-ubuntu24.04`（无需 GPU，nvcc 交叉编译）。
2. **配置**：`-DCMAKE_CUDA_ARCHITECTURES=75`，gcc-13/g++-13，C++20。
3. **校验**：cuobjdump 逐对象检查 SASS 零 `cp.async`（防 JIT 伪装、防架构穿透）。
4. **诊断**（失败时自动）：`--keep --keep-dir` 保留 PTX，awk 沿 `.loc` 源码映射打印每条 `cvt.rn.bf16x2.f32` 的最近源码位置。
   - 坑 1：keep 目录必须在 Configure 前 `mkdir -p`，否则 CMake try-compile 阶段 nvcc 直接崩。
   - 坑 2：grep 模式必须容忍舍入修饰符：`cvt\.[a-z0-9.]*bf16x2\.f32`。
5. **推送链路**：本地 git 443 被阻断，用 Git Data API（blob→tree→commit→PATCH ref）。注意 API 建的远端 commit 本地无对象，diff 基准必须用 `BASE_SHA` 环境变量显式指定。

## 3. CI 迭代台账（12 轮）

| 轮次 | run | 障碍 | 修复 |
|---|---|---|---|
| T2.0 | 36371026485 | cp.async | memory.cuh 同步回退 |
| T2.1 | 36372196400 | __reduce_max_sync | shfl_xor 回退 |
| T2.2 | 36372939995 | include 缺失 | sampling_device.cuh 补 include |
| T3 | 36375085006 | __bfloat162half2 拼错 | 正名 __bfloat1622half2 |
| T3.1 | 36375338741 | bf16→fp16 转换也被守卫 | mma.cuh 手写位级转换 |
| T3.2 | 36375696275 | m16n8k16 requires sm_80 | 暴露形状问题 |
| T4 | 36376274445 | s8/fp8 形状 | mma_f16 拆 2×k8、mma_s8 拆 2×k16 |
| T5 | 36377696035 | fp8 需 sm_89 等 | s8/fp8/tf32 → __trap |
| T6 | 36378910569 | cvt.bf16x2.f32（成对版） | 替换 71 处/38 文件 |
| T6.1 | — | cvt（单元素版） | 替换 440 处/88 文件，新建 bf16_compat.cuh |
| T7–T7.2 | 36380868989 / 36381187711 / 36382049542 | 诊断自身两坑后命中 | mkdir + grep 模式修正 → .loc 定位 pack_bf16x2 |
| T8 | 36382865499 | pack_bf16x2 手写 asm | sm_80+ 守卫 + RNE 位级打包 |
| T9 | **36384008002** | GCC constexpr odr-use | 显式按值捕获 → **全绿** |

## 4. 运行期验证清单（T10 上机后）

1. **第 0 项（TC 定案）**：PyTorch FP16 matmul 实测吞吐 ≥40 TFLOPS → Tensor Core 存在（回应"T10 TC 被砍"传言；现有实测证据 50–54 TFLOPS FP16 vs 8 TFLOPS FP32 支持存在论）。
2. 散热：T10 无风扇，临时散热装好前不跑持续负载。
3. 环境：仅 NVIDIA 驱动（编译产物已就绪，目标机无需 CUDA toolkit）。
4. 推理基准：三元制品 9.81 GiB 装入 16GB → decode tok/s 与 prefill 吞吐。
5. 正确性：FP16 MMA 路径输出对 sm_89 参考抽查；确认 __trap 分支未被误触。

## 5. 风险与未决

1. `__trap` 分支若被模型加载路径触达会直接崩溃（预期行为），需确认三元模型不走 s8/fp8/tf32 内核。
2. 2×m16n8k8 拆分的额外指令开销未实测（图灵 fp16 TC 理论吞吐与 Ada 同）。
3. RNE 位级转换与内建舍入在极端 NaN/Inf 下的一致性待正确性抽查覆盖。
4. 若第 0 项证实无 TC：FP16 MMA 路线重估，fallback 为 CUDA cores FP32（decode 吞吐大幅下降）。
