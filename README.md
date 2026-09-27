---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '4d0602ee-d3e7-40a3-b509-433c10b2e934'
  PropagateID: '4d0602ee-d3e7-40a3-b509-433c10b2e934'
  ReservedCode1: 'dad92cf9-39df-431f-aad5-42b83229faab'
  ReservedCode2: 'dad92cf9-39df-431f-aad5-42b83229faab'
---

# Bowsai-NInfer — RTX 5080 三元量化复现指南

> 本仓库在 **RTX 5080（16GB）** 跑通了 **Bowsai-2-27B 三元量化模型**（`Ternary-Bonsai-2-27B.ninfer`）。
> 名字"4090"是上游沿用，**本 fork 实测平台是 5080**。
> 这份 README 是**复现索引**：每一步要去看什么（文件/命令/报告/源码位置）都列出来，
> 让复现者**自己照着查证推进**，而不是靠我口头转述。

---

## 0. 先看这四样（搞清楚全貌再动手）

| 要看什么 | 在哪 |
|---|---|
| 完整开发史（M0–M8 逐日原始记录） | Bowsai 主库 `docs/项目构建史.md` |
| 技术论文《三元-Bonsai-27B-NInfer-移植》 | `三元-Bonsai-27B-NInfer-移植-技术论文-20260920.pdf` |
| 作者工具链说明 docs/01–04 | `landing/tools/ninfer-ada-ternary/docs/` |
| 各里程碑报告（M0–M6 + CraneBW + A/B） | Bowsai 主库 `docs/reports/` |

---

## 一、我们要复现的东西

**16GB 卡 + 2.125bit 三元量化跑 27B**，智力不塌（PPL 6.1129）、显存可控、256K 长上下文可跑。

## 二、7 步复现路线（每步标明"要看什么"）

### 第 1 步 · 锁定来源
- **要看的**：构建史 §2 上游血统表；作者 ModelScope 仓 `shensanshu/ninfer-ada-ternary`；基线 `Ambolio/ninfer-4090-windows @ 6eb70a0`。
- 关键：补丁基线 = **Ambolio v1.0.8**，不是官方线。工具仓锁 commit（pack.py 拒绝非 groupwise-int 模板那版）。

### 第 2 步 · 下载权重（~70GB）
- **要看的**：HF `prism-ml/bonsai-2`；量化 GGUF 清单（PQ2_0 7.2G / PTQ1_0 5.9G / mmproj）；DFlash2 组件。
- **要看制品代际**：`<制品>.ninfer` magic 字节，确认引擎代。我们的 = 02（v2，与 v1.0.8 自洽）。

### 第 3 步 · 打包模板 + 三元制品
- **要看**：作者工具链 `landing/ninfer-ada-ternary/`（docs/01-04 + patches + `pack.py` + `MAPPING.json` + `verify`）。
- 产物（本机已有，可对照）：
  - 模板 `landing/artifacts/qwen3_8_27b.ninfer`（19.03 GiB/1190 对象）+ 它的 `qwen3_8_27b.ninfer.conversion.json`（转换报告，看配置）
  - 三元 `landing/artifacts/Ternary-Bonsai-2-27B.ninfer`（9.81 GiB/1192 对象）
  - `bonsai2-hadamard-meta.json`（折叠基变换元数据）
- **坑（发布缺件）**：作者包缺 `numeric.py` / `_ternary_ref.py` → 用运行时 shim `landing/tools/ternary_shim.py`，**不改上游**。
- 六项验证链（pack check/build/payload_order/signs/row_order/assembly）必须全 PASS。

### 第 4 步 · 构建引擎（M4）
- **要看**：
  - 基线 `build_v1.0.8.bat`（构建配方）
  - 基线 `CMakeLists.txt` **L59-62**（CUDA ≥13.1 硬门）、**L82-85**（仓根 `ffmpeg\` 硬依赖）
  - `device.h`（非 sm_89 的 `#error` 守卫）→ 改 CMake 架构列表 `89|120a`
  - 用 `cuobjdump` 验 **sm_120a SASS**（防 JIT 伪装）
- 本报告用 CUDA 13.3 旁装，构建目录 `_build_5080`。

### 第 5 步 · 验收（M5）
- **要看**：报告 `M5端到端验收报告`；`report.json`（PPL 原始值）。
- 判据：PPL ≤6.8（越低越好），我们 6.1129。
- 显存账本三场景，验证占用与账面吻合。

### 第 6 步 · 速度标定（M6）
- **要看**：`M6` 报告 + 三档验证脚本。
- 5080 实测：裸 decode 68 / MTP / DFlash2 英文 104.7（中文负收益）。

### 第 7 步 · 提速内核（M7 + 后续）
- **要看**：`CraneBW/ninfer-ternary-bonsai-ada`（比对 `verify`）；合并方案 `CraneBW内核合并方案`；报告 `CraneBW内核合并验证报告`。
- 结果：prefill 600→1288；后续 s8/wide_t 移植 → 冷 prefill **1.63k**。
- **坑**：CraneBW 声称 8 文件，漏合 `gemv.cuh`（新 kernel 未并）→ 首编 12 error。逐文件核对，别信"x 个文件"。

---

## 三、起服与档位（要看：serve --help / BAT 参数）

- 一套 BAT 在 Bowsai 主库，命令要点见下表：

| 档位 | 命令要点 | 5080 实测 |
|---|---|---|
| 日常 fp8 | `--kv-dtype fp8 --spec mtp --draft-tokens 2 --max-concurrency 2` | 114.3 t/s |
| k8v4-224K | `--kv-dtype k8v4 --kv-capacity ...` | 95.7 |
| k8v4-256K-MTP | 256K + MTP K=2 | 105.2（终极，余 ~0.6G 临界）|
| DFlash2 英文 | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | 139.5（中文勿用）|

- 每个档位**要看对应 BAT 的完整参数**（`起服-*.bat` 就是现成样板，改路径即用）。
- 具体参数语义看 `ninfer-serve --help`。

## 四、坑汇总（每坑标"去看什么"）

| 坑 | 现象 | 要看/对策 |
|---|---|---|
| 制品代际 | 引擎不认/乱码 | 看 `--help` 认代际 + magic 字节 |
| 发布缺件 | 打包失败 | `landing/tools/ternary_shim.py`（写 shim 不改上游）|
| 视觉 | fork 不支持图片 | 走外部视觉模型 |
| arch 守卫 | 非 sm_89 `#error` | 看 `device.h` + CMake 架构列表 |
| ffmpeg / CUDA | 编不过 | 基线 CMakeLists L59/L82 |
| CraneBW 漏合 | 首编 error | 逐文件核对，别信"x 个文件"|
| 并发挤爆 | `preparing CUDA graphs bad alloc` | 小余量档并发 1 |
| 思考空收尾 | 长思考无正文 | 已修：commit `145bccb`（frontend 强制进正文）|
| Agent 死循环 | "继续"失忆 | 恢复会话/投喂材料，别靠思考帽 |

## 五、关键文件索引（本仓库可直接查）

| 文件 | 看什么 |
|---|---|
| `landing/artifacts/Ternary-Bonsai-2-27B.ninfer` | 三元制品 |
| `landing/artifacts/qwen3_8_27b.ninfer` + `.conversion.json` | 模板 + 转换报告 |
| `landing/tools/ternary_shim.py` | 三元运行时 shim |
| `landing/tools/ninfer-ada-ternary/` | 作者工具链 |
| `*起服-*.bat` | 各档位现成命令 |
| `docs/项目构建史.md`（主库）| 逐日全史 |

---

**说明**：本仓库是 Bowsai-2-27B 三元部署的复现工程；上游作者工作看上游，这里列的是"我们怎么跑起来的每一步，要去看什么资料/命令"。实测数据全来自本机，非营销。