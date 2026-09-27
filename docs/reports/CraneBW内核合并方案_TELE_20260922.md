# CraneBW 三元 Tensor-Core 内核合并方案（TeleAgent）

> 撰写：TeleAgent，2026-09-22（M6 关单后 / 用户批准合并后、等"开始"信号前完成准备）
> 性质：合并实施方案。**GPU 相关实施动作一律等用户明确"开始"信号**（当前 18787 端口 k8v4 serve 正在跑用户的鹈鹕任务，不得动 GPU）。
> 结论先行：**建议合入，方案为"CraneBW 8 文件替换 + 两处适配改动"，预期 prefill 600→1200+ t/s、verify 提速（MTP K=1 预计 72→90+ t/s），DFlash2 verify 需一行 gate 放宽，全程可回退。**

---

## 一、总体结论

1. CraneBW/ninfer-ternary-bonsai-ada 是在**与我们完全同源的基线**（Ambolio/ninfer-4090-windows + shensanshu 三元 patches）上，用 7 个提交独立实现的 ternary tensor-core 路径。4070TiS（672GB/s）实测：prefill 1230 t/s（我们 600）、decode+MTP draft3 100.8 t/s。我们 5080（960GB/s，带宽高 1.5x）合入后预期更高。
2. 代码审核通过：解码语义与我们制品**逐位一致**（(code−1)×scale），数学结构自洽，边界守卫完整，且自带引擎侧 A/B 路由开关（`NINFER_TERNARY_PREFILL=ref/block/mma`、`NINFER_TERNARY_VERIFY=tile`）——正是作者验证法要求的形态。
3. 合并是**自包含替换**：作者 mma 符号（kTernaryMma*/ternary_mma_enabled/mma_admits）的全部引用只在 gemm.cu 与 mma_small_t.cuh 两个文件内部，无外部引用；替换后其余源码零改动。
4. 三处必须适配的点（详见 §四）：DFlash2 verify gate（T≤4→T≤8）、NINFER_TERNARY_MMA 总开关语义、gemm.cuh 头注释。
5. 主要风险：CUDA 13.3 cudafe++ 前端崩溃坑（作者版 gemm.cuh L15-18 有记录，CraneBW 用 CUDA 13.4 未踩）；sm_120a 编译适配（作者版 mma 已证可行，CraneBW 版同指令集家族）。

## 二、三方源流考证（证据）

### 2.1 克隆与基线

- 克隆：`https://github.com/CraneBW/ninfer-ternary-bonsai-ada` → `J:\Bonsai\landing\repos\cranebw-ninfer-ternary-bonsai-ada`（经 7890 代理；47890 残留监听拒绝转发，直连不通）
- 仓库 19 个提交：`e1b6621` = "baseline: NInfer v1.0.8 (Ada line) + Ternary Bonsai 2 27B port patch set"，其上为优化提交
- CraneBW README 自述直接基线 = Ambolio Windows line + shensanshu patches——与我们工作区同源
- baseline 快照：`J:\Bonsai\.temp\cranebw-baseline`（git worktree @ e1b6621；**合并完成并验证后需 `git worktree remove` 清理**）

### 2.2 哈希级全树对比（MD5，排除 .git/_build_5080/ffmpeg/__pycache__）

我们工作区（HEAD=6eb70a0784b68c87efcbeb0b08bd2c3a95914492，28 modified + ternary 目录 untracked）与 CraneBW baseline 的全部内容差异 = **10 个文件**：

| 文件 | 差异来源 | 处置 |
|---|---|---|
| ternary_rowsplit_gemm.cu (ours 8850B / base 6404B) | 我们侧=作者 mma 派发；base=纯 SIMT | **合并核心，见 §四** |
| ternary_rowsplit_mma_small_t.cuh（仅 ours 有） | 作者 mma verify/prefill 万能 kernel；CraneHEAD 另写两个新文件 | **合并核心，见 §四** |
| ternary_rowsplit_gemm.cuh (5464/5265B) | 我们侧多 mma 注释与头改动；CraneBW 未改此文件 | 保留我们版本，微调注释 |
| CMakeLists.txt（根）+ src/CMakeLists.txt | 我们侧 sm_120a patch（"89\|120a" 都定义 NINFER_SM89=1；ffmpeg EXISTS 守卫） | **保留我们的，不动** |
| layouts_impl.h | 我们侧含 M5 三元旋转 scratch 修复（修首个打分窗口 bad_alloc）；base=缺件原版；**CraneHEAD 也没有此修复**（Select-String ternary_rotation_workspace_bytes 无匹配） | **保留我们的——合并不触碰此文件** |
| program_impl.h | 我们侧多 `#include <cstdio>`（作者 patch） | 保留，不动 |
| bf16_gdn_gating_proj_kernels.cu | base 侧 GCC lambda 捕获修复（注释明言 MSVC 不需要） | 不移植 |
| q4_m64.cu | base 侧多 `#include <cuda_bf16.h>` | 无害，不移植 |
| tools/artifact/{layouts,numeric}.py | base 侧补全 shensanshu 发布包缺失的 Python 侧 TernaryFormat 注册（又一个发布缺件佐证） | 不影响引擎运行；已记录，仅重打包制品时才相关 |

**判定**：除 ternary 核心三文件外，两侧差异全部可解释且互不冲突；合并只动 `src/ops/linear/ternary/` 内 4 个文件（见 §四）。

### 2.3 CraneBW 的优化提交谱系（全部带实测数据）

按时间序：62c88a2 token/row-blocked GEMV（3.20x）→ b24daa6 负结果记录 → 2c26817 PQ2_0 tensor-core prefill（6.8x）→ 2b22b34 NCU 调优（再 3.51x）→ 3b67ec8 路由注释修正 → 1b0cc15 补两条不变量 → 1dadbb0 small-T verify tensor-core（MTP 转正：43.2→89.9 t/s）→ 45b3b36/2a95af3 SASS 驱动 decode 两轮（bit-identical）→ 6f3494c rotation launch packing → 34cef66 两个 m-tile 共享 activation staging → 35dbdf0 GDN residency 假设证伪记录。

## 三、8 个改动文件逐个审核结论

| 文件 | 改动 | 审核结论 |
|---|---|---|
| ternary_rowsplit_mma.cuh（新增 402 行） | prefill tensor-core GEMM。fp16 魔数 0x6400(1024.0) 解码 (1024+c)−1025=c−1，**scale 折入解码值**（精度损失已知：PPL 9.69192 vs ref 9.6905，0.015%）；K tile=64（半 group）换 2 CTA/SM；静态 smem 42.3KB（static_assert ≤48KB）**避开 graph capture 内 cudaFuncSetAttribute 非法**；边界 Full/非 Full 双模板 | **通过**。数学正确（c−1 语义与我们制品 decode_one 一致）；Schedule<64,128,64,32,32,2,2> 为 4070TiS 实测最优，5080 上可用 env/重编译扫描 |
| ternary_rowsplit_mma_small_t.cuh（新增 408 行） | verify（T=2..4）tensor-core GEMV。bf16 魔数 0x4300(128.0)，A 操作数**精确** {−1,0,+1} 无舍入；per-warp 一个 128-group，scale 在 K 归约后 fp32 乘（比 prefill 版更准）；8-warp shared 树归约；dead-column 清零一次复用；bank padding +16B | **通过**。K%1024==0 门满足本模型全部宽度（5120/6144/10240/17408）；TileCols=8 模板本身支持 T≤8（**这是 DFlash2 放宽 gate 的依据**）；kRowsPerCta 派生自模板参数而非 Schedule 的坑已自查（L144-148） |
| ternary_rowsplit_gemm.cu（+244/−43） | 派发重构：`NINFER_TERNARY_PREFILL`（mma/block/ref，读一次入 static）+ `NINFER_TERNARY_VERIFY=tile` 回退 + small_t/prefill-mma 接入；gemv_admits 加 `x.ne[0]==w.k` 守卫；launch 前显式 throw 校验 | **通过**。路由逻辑清晰；负结果与正结果都记录在注释 |
| ternary_rowsplit_gemv.cuh（+160） | token/row-blocked SIMT GEMV（fallback 与 A/B arm）。(code−1) 解码、per-group scale 乘部分和后 shuffle 归约、CTA 级早退+行级 active 守卫、coalesced flush | **通过**（作为 fallback 路径，默认不走） |
| ternary_rotation.cu（+41）/ _kernels.cuh（+15） | rotation launch packing：warp 总数 ≤128 时 1 warp/block（decode 形状摊 SM 降延迟），否则原 8 warps/block；kernel 改从 blockDim 读。**纯调度变化，数值不变**；env `NINFER_TERNARY_ROTATE_WPB` 可覆盖 | **通过**。低风险 |
| gated_delta_net/recurrent.cu（+19）/ .cuh（+25） | record kernel 加 MinBlocksPerMultiprocessor 模板参数（2/6/8），默认 2=原行为；residency 假设已被实测证伪（README 负结果记录） | **通过**。默认行为不变，可选合入或不合入（建议合入保持同源） |

**总评**：AI 辅助但质量显著高于一般水准——每处优化带 NCU 实测依据、负结果如实记录、A/B 路由内建、作者验证法（引擎侧 ref 对比）内建。两个新 kernel 的数值语义均与我们制品逐位一致。

## 四、合并设计（核心）

### 4.1 架构差异与取舍

| | 我们现版（作者 mma） | CraneBW |
|---|---|---|
| verify（T=2..4，MTP K=1/2） | 万能 small_t kernel（20522B） | 专用 small-T mma kernel |
| prefill（T≥128） | **同一个**万能 small_t kernel（ceil(T/8) 遍权重） | **专用 GEMM**（128-token tile，2.6x 来源） |
| T=5..63 | 万能 small_t | SIMT blocked GEMV |
| T=8（DFlash2 verify） | 万能 small_t 覆盖 ✓ | **gate T≤4 不覆盖 ✗ → 会掉 SIMT** |

取舍：**采用 CraneBW 架构**（prefill 专用路径是 2.6x 的来源，作者万能 kernel 的 prefill 天花板实测仅 600）。DFlash2 缺口用 §4.2 一行改动补上。

### 4.2 文件级合并清单（"开始"信号后执行）

全部在 `J:\Bonsai\landing\repos\ninfer-4090-windows\src\ops\linear\ternary\` 内：

1. **新增** `ternary_rowsplit_mma.cuh`（从 CraneHEAD 原样拷入）
2. **替换** `ternary_rowsplit_mma_small_t.cuh`（作者 20522B 版 → CraneHEAD 408 行版；作者版先改名留档 `ternary_rowsplit_mma_small_t.cuh.author_backup`，验证通过后删除）
3. **替换** `ternary_rowsplit_gemm.cu`（作者版 → CraneHEAD 版）
4. **保留** `ternary_rowsplit_gemm.cuh`（我们版本；仅更新头注释说明新路径）
5. **应用** rotation 两文件 diff、GDN 两文件 diff（直接 `git apply` CraneBW 的对应 diff，两侧基线哈希一致、可干净应用）
6. **适配改动 A（DFlash2 gate）**：gemm.cu 中 `launch_ternary_gemm_t8` 的 verify 分支把 `gemv_admits(x, w, 4)` 内的 small_t 判定改为 T≤8：
   `if (verify_uses_small_t() && (w.k % kSmallTGroupK) == 0 && x.ne[1] <= 8)`（原 4 来自外层 gemv_admits(x,w,4)；实现时以最小 diff 放宽，且**必须走引擎 A/B 验证 T=8 输出**——TileCols=8 时 live_cols=8 无 dead column，形状自洽）
7. **适配改动 B（总开关语义）**：作者版 `NINFER_TERNARY_MMA=0` 总开关在 Crane 版无对应物；Crane 的等价组合 = `NINFER_TERNARY_PREFILL=block` + `NINFER_TERNARY_VERIFY=tile`。**不改代码**，在 BAT 注释与本方案记录等价开关组合（避免加补丁面）
8. **编译注意**（作者 docs/01 §4.1 / docs/04 §3.1 + 本仓 gemm.cuh L15-18）：改 .cuh 后必须 touch 全部包含它的 .cpp/.cu；`VSLANG=1033`；若 cudafe++ 0xC0000409 复现，按作者 docs/04 规避法处理（拆 TU/改写触发构造），并在日志记录

不触碰：layouts_impl.h（含 bad_alloc 修复）、program_impl.h、CMakeLists×2、storage/gemv/launch/dispatch/rotation.h 等其余 ternary 文件。

### 4.3 编译与产出

- 构建目录沿用 `_build_5080`（ninja，CUDA 13.3 旁装于 `J:\Bonsai\landing\cuda-13.3`，sm_120a）
- 产出三 exe：ninfer.exe / ninfer-serve.exe / ninfer-perplexity.exe
- 构建前快照：三个 exe 已备份至 `J:\Bonsai\landing\backups\build_5080_pre_cranebw_20260922\`（各 237MB，与现行 serve 版本对应）

## 五、验证计划（作者法，全链路）

**前置**：确认用户鹈鹕任务结束、GPU 空闲（`nvidia-smi` 无计算进程）。

1. **编译验证**：0 error；三 exe 出齐
2. **引擎侧正确性 A/B（作者法核心，T=1 查不出 token-tile/layout 错误）**：
   - `NINFER_TERNARY_PREFILL=ref` vs 默认（mma）跑同一短 prefill 贪心输出，**逐 token 比对**（允许顺序差异位，最终文本应一致；若不一致即 FAIL）
   - `NINFER_TERNARY_VERIFY=tile` vs 默认（small_t）在 MTP K=1 下同样比对
3. **PPL A/B**：ninfer-perplexity 同判据窗数跑合并前后两版；判据：新 prefill 路径与 ref 差 ≤0.05%（CraneBW 自证 0.015%，我们 sm_120a 上留余量）；**同时与 M5 黄金参照 6.113/作者 6.8196 量级对账**
4. **性能复测（M6 三档，T=0 greedy 预热后中位）**：
   - 日常档 bf16+MTP K=1：基线 72.0 t/s，目标 ≥85（CraneBW 4070TiS verify 提速 43.2→89.9 的映射）
   - k8v4 长文档档：prefill 基线 600 t/s，**目标 ≥1200**
   - DFlash2 K=7 英文档：基线 104.7，验证 gate 放宽后 verify 不退化（若 mma small_t T=8 有效，目标 ≥120）
5. **稳定性**：每档连跑 ≥2 轮取优（排除 WDDM 偶发掉速；参考 M6 期 LM Studio 共存干扰教训，测试时关闭无关 GPU 进程）
6. **长上下文抽样**：k8v4 档 64K prompt 一次预填成功 + 末针召回抽测（对照 CraneBW fp8 6/6 结果形态）
7. 全部结论附 [EVIDENCE]（命令+输出摘要）；测试脚本沿用 `.temp/m6/bench_one.py`，新增 prefill 计量口径（ttft 段）

## 六、回退方案（三层）

1. **运行时回退（零重建）**：`NINFER_TERNARY_PREFILL=ref`（或 block）+ `NINFER_TERNARY_VERIFY=tile` → 全部回落 SIMT 路径，行为等同合并前
2. **二进制回退**：`J:\Bonsai\landing\backups\build_5080_pre_cranebw_20260922\` 三个 exe 覆盖回 `_build_5080\`（当前 serve 即此版本，SHA 与 0:24 构建一致）
3. **源码回退**：合并前记录 `git status` 全量输出 + 三文件改名留档；最坏情况 `git checkout -- . && 删除新增 untracked`（ternary 目录为 untracked，checkout 不覆盖，需手工恢复改名文件）

## 七、风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| CUDA 13.3 cudafe++ 前端崩溃（gemm.cuh L15-18 有先例记录） | 中 | 新 kernel 结构与已崩溃的 gemv kernel 不同族；若复现按 docs/04 规避，必要时把 mma.cuh 拆独立 TU |
| sm_120a 编译/运行差异 | 低-中 | 同指令集家族（cp.async/ldmatrix/mma.bf16），作者版 mma 已在 5080 运行；puzzling 参数按 4070TiS（66SM/637GB/s）取得，5080（84SM? 以 deviceQuery 为准/960GB/s）上默认配置可能非最优但应正确，性能扫描后置 |
| DFlash2 verify T=8 行为变化 | 中 | §4.2 改动 A 显式处理 + T=8 引擎 A/B 强制验证 |
| 5080 上 CUDA graph capture 新路径 | 低 | 新 kernel 静态 smem 设计已规避 capture 内 cudaFuncSetAttribute；k8v4/dflash2 档烟测覆盖 |
| prefill 提速不及预期（WDDM/带宽上限） | 低 | 600→1230 是 kernel 结构差异，非带宽运气；5080 带宽 1.5x 只会更有利 |

## 八、待用户裁决项（不阻塞合并，合并后汇报）

1. DFlash2 Full 头崩溃修复 A1/A2/A3（上轮遗留）
2. 日常档并发=2 是否保留（5.5GB 余量下可行，但与 MTP K=1 并发的显存峰值需在合并后复测）
3. 性能扫描参数（NINFER_TERNARY_SMALL_T_ROWS 16/32/48、ROTATE_WPB、prefill schedule）是否做成 BAT 档位——**遵循"一次性任务禁止产品化"**，默认只留 env 注释，不做 UI/档位

## 九、执行状态记录

- [x] 克隆 + baseline worktree（.temp/cranebw-baseline）
- [x] 三方源流考证 + 哈希全树对比
- [x] 8 文件代码审核（本方案 §三）
- [x] exe 备份 + git 状态记录（HEAD=6eb70a07…92，29 条 status）
- [ ] **等用户"开始"信号** → §4.2 执行 → §五验证 → §六收尾（清理 worktree）
- [ ] 合并完成后：更新落地实施方案主文档 + 撰写验证报告至 docs/reports/