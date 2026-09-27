---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'bda15991-1de3-457b-91b8-6aa2a7e225aa'
  PropagateID: 'bda15991-1de3-457b-91b8-6aa2a7e225aa'
  ReservedCode1: 'e09d8d33-fd83-4be4-b93f-96879ea82e22'
  ReservedCode2: 'e09d8d33-fd83-4be4-b93f-96879ea82e22'
---

# Bowsai-NInfer — RTX 5080 三元量化复现指南

> 本仓库在 **RTX 5080（16GB）** 上跑通了 **Bowsai-2-27B 三元量化模型**（`Ternary-Bonsai-2-27B.ninfer`）。
> 名字里的 "4090" 是上游沿用，**本 fork 实测平台是 5080**。
> 这份 README 是**能照做的复现手册**：从拿到论文 → 到跑起来，每一步怎么做、踩过什么坑，全在这里。
> 逐日原始记录：Bowsai 主库 `docs/项目构建史.md`。

---

## 一、你要复现的东西

在 16GB 显卡上，用 **2.125bit 三元量化**跑 **27B 模型**，达到：
- 智力不塌（PPL 6.1129）、显存可控（权重 6.7GiB + KV 分页）、速度可接受
- 长上下文（256K）可跑

一句话：**27B + 三元量化 + 16GB 卡，不是传说，是能复现的路线。**

## 二、先准备什么（前置材料）

| 材料 | 来源 | 作用 |
|---|---|---|
| 技术论文《三元-Bonsai-27B-NInfer-移植》 | 作者分发（ModelScope） | 宪法级文档，先读 |
| 作者三元工具链 `ninfer-ada-ternary` | ModelScope `shensanshu/ninfer-ada-ternary` | docs/01-04 + patches + pack.py + MAPPING.json + verify |
| 引擎基线 | Ambolio/ninfer-4090-windows @ 6eb70a0（v1.0.8） | 本 fork 的起点 |
| 优化内核 | CraneBW/ninfer-ternary-bonsai-ada | prefill 提速 |
| 模型权重 | Qwen3.8-27B（HF 18 分片）+ 量化 GGUF（PQ2_0/PTQ1_0）+ DFlash2 | 见"第3步" |

## 三、复现路线（7 步，照做）

### 第 1 步：锁定来源
- 作者补丁基线 = **Ambolio v1.0.8 @ 6eb70a0**，不是上游官方线（官方线走法不同，别搞混）。
- 工具仓锁定 commit（pack.py 拒绝非 groupwise-int 模板 up front 那版）。

### 第 2 步：下载权重（约 70GB）
- Qwen3.8-27B 底座 18 分片（HF `prism-ml/bonsai-2`）
- 量化 GGUF：`PQ2_0`（7.2G）/ `PTQ1_0`（5.9G）二选一 + DFlash2 草稿组件
- ⚠️ **坑1（代际）**：制品有 v1/v2/v3 代际，引擎**只认本代**。确认制品的 magic 字节 = 对应引擎代（我们的 = v2，与 v1.0.8 自洽）。

### 第 3 · 打包模板制品（M2–M3）
1. 先转 **bf16 模板**：`qwen3_8_27b.ninfer`（19.03 GiB / 1190 对象）。需要 **torch 2.2+cu130 + RAM≥64GB**（本机 127.8GB），转换约 114 秒。
2. 再用作者 `pack.py` 打**三元制品**：`Ternary-Bonsai-2-27B.ninfer`（9.81 GiB / 1192 对象 = 1190 + mtp 2）。
3. ⚠️ **坑2（发布缺件）**：作者发布包缺 ①Python 侧格式注册（`numeric.py`）②`_ternary_ref.py`。**不用改上游包**，写运行时 shim（`ternary_shim.py`）注入即可。
4. **六项验证链必须全 PASS**：pack check / build / payload_order / signs / row_order / assembly（15/15，duplicated=0）。
5. ⚠️ **坑3（visual）**：视觉（ViT/mmproj）**不打包**——该 fork 不支持图片输入，视觉走外部模型。

### 第 4 · 构建引擎（M4）
1. 前置硬门槛（基线 CMakeLists 检查）：**CUDA ≥13.1**（本机 13.0 不够，旁装 13.3）+ **仓根 `ffmpeg\`**（MSVC 分支硬编码，MEDIA_ACQUIRE 强制 ON，缺目录编不过）。
2. ⚠️ **坑4（arch 守卫）**：基线 `device.h` 对非 sm_89 直接 `#error`。解法 = CMake 架构列表改 `89|120a` 复用 SM89 路径，生成**原生 sm_120a SASS**（用 `cuobjdump` 验证，防 JIT 伪装）。
3. 编译：`cmake -B _build_5080 -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 ...` + `cmake --build`。

### 5 · 验收（M5）
- **PPL 6.1129**（1,044,557 tokens）——注意 PPL 判据带 ≤6.8，越低越好。
- **显存账本三场景**实测，验证占用与账面吻合。
- 这里有个未的坑见"坑汇总"。

### 6 · 速度标定（M6）
- 三档：裸 decode / MTP / DFlash2。我们 5080 上：裸 68、MTP 见矩阵、DFlash2 英文 104.7（中文负收益，慎用）。

### 7 · 提速内核（M7 + 后续）
- **合并 CraneBW 内核**：prefill 600→1288 t/s，decode K1 zh +44%。
- ⚠️ **坑5（漏合）**：CraneBW 声称 8 文件，实际漏了 `gemv.cuh`（新 kernel 未并），首编 12 error 才补上——合并要**逐文件核对，别信"x 个文件"**。
- 后续移植 **s8（int8）+ wide_t prefill 内核**：冷 prefill **1.63k tok/s**（主推档）。

## 四、起服与档位（你跑起来要看这个）

一套 BAT（Bowsai 主库）可选用：

| 档位 | 命令要点 | 5080 实测 |
|---|---|---|
| 日常 fp8 | `--kv-dtype fp8 --spec mtp --draft-tokens 2 --max-concurrency 2` | **114.3 t/s** |
| k8v4-224K | `--kv-dtype k8v4 --kv-capacity 224000...` | 95.7 |
| k8v4-256K-MTP | 256K + MTP K=2 | **105.2**（终极，余 ~0.6G 临界）|
| DFlash2 英文 | `--spec dflash2 --draft-tokens 7 --lm-head-draft` | **139.5**（中文勿用）|

⚠️ **坑（并发/显存）**：并发 2 会挤爆剩余显存（`preparing CUDA graphs bad allocation`），小余量档位**并发 1**。MTP K 从 1→2 转正，K=3 过临界回落，5080 上 **K=2 最优**。

## 五、踩坑汇总（重点）

1. **制品代际不对齐** → 引擎不认/乱码。锁 commit + 认 HEAD。
2. **发布缺件**（numeric.py / _ternary_ref.py）→ 写 shim，不改上游。
3. **视觉** → fork 不支持图片，外部模型。
4. **arch 守卫** → CMake 架构列表级改动，验 SASS。
5. **ffmpeg / CUDA 版本** → 硬前置，缺则编不过。
6. **CraneBW 合并漏文件** → 逐文件核对。
7. **并发挤爆显存** → 小余量档并发 1。
8. **思考区空收尾** → 已修（frontend 方案 C，commit `145bccb`）：思考区 stop 强制进正文。
9. **Agent 思考死循环** → 贫瘠输入 + xhigh + "继续"会失忆循环；恢复会话或投喂材料，别靠思考帽。

## 六、成果与授权

- 模型 Apache-2.0（Qwen 团队），可公开分发/复现。
- 本 fork 增量全在 git，可逐条回退。
- **实测数据全来自本机，不是营销**。

---

**说明**：本仓库是 Bowsai-2-27B 三元部署的复现工程。原作者工作请看上游，这里是"我们怎么把它在 5080 跑起来的完整路线、做法与坑"。