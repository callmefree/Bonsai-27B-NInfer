# Bonsai 2 27B + NInfer 本地落地方案（讨论稿）

- 作者：TeleAgent（TELE 侧）
- 日期：2026-09-21
- 状态：**讨论稿，未批准开工**——供用户与 CODE 侧互相研究后裁决
- 目标硬件：RTX 5080 16GB（sm_120 Blackwell）/ Windows / J 盘 250GB 可用

---

## 1. 目标与定位

1. 验证论文《面向消费级 Ada 卡的 1.58-bit 三元 27B 大模型推理引擎移植》（2026-09-20）的最新技术；
2. 为本机提供有效的本地推理能力（27B 级、7.74GB 制品、带 MTP 100+ t/s）；
3. 定位为独立实验/开发机，与 LiveAvatar 不同时满载运行。

## 2. 调研结论（信息源与关键事实）

| 信息源 | 关键收获 |
|---|---|
| 论文 PDF（本地 J:\Bonsai） | 完整技术细节 + 附录 A 复现清单 11 步 |
| 魔搭仓 README（shensanshu/ninfer-ada-ternary） | 引擎基线=**Ambolio/ninfer-4090-windows @ 6eb70a07（v1.0.8 线）**，禁用 v1.2.0/其他线；pack.py 模板必须 groupwise-int；仓库 4 篇 docs 即复现手册，**比论文 PDF 新** |
| HF 仓库（prism-ml/Ternary-Bonsai-2-27B-gguf） | 官方 GGUF 清单实测：PTQ1_0 5.9GB / PQ2_0 7.2GB / mmproj-Q8_0 629MB / mmproj-BF16 931MB / F16 53.8GB；下载量 190 万=llama.cpp 线为主路径 |
| CSDN 博客评论区 | 第一坑实录：模板 schema 不对报 `unmapped gdn object`（nvfp4 与 groupwise-int 对象命名差异对照表已在 README） |
| B 站视频（BV1dPeB6XE6d） | "8G 显卡流畅运行"→16GB 无压力；作者声明"简易现行版本，仍在开发中，仓库比发布版新" |
| HF GGUF 仓 README（全文精读） | **stock llama.cpp/LM Studio 不能跑此 GGUF**（三元内核在 PrismML fork，主线拒载且 Q2_0 静默乱码）；**Blackwell 卡 PQ2_0 更快**（5090：129.9 vs 120.5 t/s）；官方 14 基准衰减模式=数学/代码无损、缺口集中在知识推理与视觉；架构确认：27.36B=24.35B 骨干（64 层，~75% 线性注意力）+2.54B emb/LM head+0.46B 视觉塔 |
| 本地实测 | RTX 5080 16GB（sm_120）、VS2022 Community+BuildTools（含 VC 工具）、CUDA 12.6/12.8/13.0 已装、cmake/ninja 不在 PATH、Python 3.12 |

## 3. 三个关键澄清（预算依据）

1. **7.74GB 成品不是下载来的，是组装出来的**：三元权重（GGUF 5.9GB）+ 视觉载荷 + MTP 头 + tokenizer + schema 骨架（全部来自底座模板）。
2. **55GB 底座是一次性原材料**：仅用于现产 groupwise-int 模板（pack.py 硬依赖），产完即可删，稳态占用只有成品+引擎约 13GB。
3. **GGUF 是 PrismML fork llama.cpp 的专用格式**：stock llama.cpp 与 LM Studio（内置主线）均不能跑；fork llama.cpp 线只求"能跑"只需 6.5GB（Blackwell 卡选 PQ2_0 7.2GB 更快）；55GB 只为 NInfer 路线服务。
4. **LM Studio 路线不成立**：LM Studio 内置主线 llama.cpp，会拒载三元格式；且主线加载 Q2_0 时静默输出乱码（无警告）——严禁用主线试错。

## 4. 路线对比与两步走建议

| 路线 | 做法 | 下载量 | 工期 | 预期 |
|---|---|---|---|---|
| 方案0：PrismML fork llama.cpp | fork 预编译二进制 + PQ2_0 GGUF（Blackwell 卡用 PQ2_0，官方吞吐表 5090 上 129.9 vs PTQ1_0 120.5） | 7.2GB | 半天 | 5080 估 60-75 t/s；**只能验证模型质量，验证不了引擎贡献**（无 MTP/快路径） |
| **路线甲：论文 fork 改架构（主线）** | Ambolio 基线(v1.0.8)+作者 patches → sm_120 重编 | +62GB（底座 55 + GGUF） | 4-6 天 | 纯解码 62-64+、MTP K=2 96.7-130.8 t/s；**唯一能验证论文核心主张的路径** |
| 路线乙：上游主线移植 Windows | sm_120a 原生但 Linux-only，需自行补 MSVC/FFmpeg/libcurl | 同上 | 更长 | 不推荐首发 |

**建议**：方案0 降级为"可选的质量基线"（fork 预编译二进制，半天顺手做，为路线甲提供智力与速度对照组）；路线甲是验证论文的主线。社区实测的顾虑与验证设计见 §4.1。

### 4.1 验证矩阵（对齐"高智力+低显存+快"三重目标）

官方数据表明退化是**选择性的**（数学 96.57 vs 97.06、代码 89.42 vs 89.07 几乎无损；缺口集中在知识推理 MuSR 70.6 vs 79.6、视觉 66.2 vs 71.4），且官方自认 "casual testing misses the collapse"——验证设计必须覆盖退化区，不能只测闲聊：

| 维度 | 口径 | 对照 |
|---|---|---|
| 智力 | PPL 6.4-6.8 一票否决 + 官方缺口区任务（知识/长链推理 3-5 题）+ 用户真实负载（中文问答为主）+ 视觉 2 题（若载 mmproj） | 论文 19/20 与官方 14 基准衰减模式 |
| 速度 | **任务分桶报数**：中文短答 / 英文代码 / 长上下文各测，报中位数不只报峰值；MTP 接受率按桶记录（论文：中文短答 80% vs 英文代码 40.8%） | 论文 96.7-130.8 区间 vs fork llama.cpp 基线 |
| 显存 | nvidia-smi 全程账本（权重+KV+MTP 分项），记录可用上下文上限 | 论文制品 7.74GB |

社区"实测不及纸面"的顾虑不采信也不否认，列为待验证项：MTP 数字本就是峰值口径（tok/轮 1.82-2.38 波动 ±27%），本矩阵直接用真实负载给它一个数据。

## 5. 路线甲阶段计划（判据驱动）

### 阶段 0：环境定稿（0.5 天）
- cmake 3.28+ / Ninja 补齐（VS2022 自带或 winget）；CUDA 先用 13.0，按 docs/01 要求对齐（必要时旁装 13.3，C 盘余 44GB 够）
- 判据：cl.exe / nvcc / cmake / ninja 四件套版本记录在案

### 阶段 1：材料获取（1-2 天）
- git clone Ambolio/ninfer-4090-windows → checkout 6eb70a07；clone 魔搭仓；HF 下 Qwen3.8-27B（~55GB）+ PQ2_0/PTQ1_0 GGUF
- 判据：权重哈希对照官方；GGUF 元数据五项（block_size=1024、Σsign_widths=28672、weight_names=401、inverse=[token_embd]、gdn_v_grouped=true）
- 待确认：转换器 `--dflash2-model` 参数是否必选（读 docs/02 定）；若可选则 DFlash2 权重不下载

### 阶段 2：引擎构建 + sm_89→sm_120 改造（1 天，最大变数）
- 先按 docs/01 原样编译基线通过，再改架构 flag 重编
- 判据：BUILD OK + 确认真重编（作者点名坑：改 .h 后必须 touch 所有包含它的 .cpp，否则静默不生效，崩在 0xC0000409）

### 阶段 3：模板制作 + 三元打包（1-2 天）
- 基树转换器产模板 → pack.py check → build → 4 个验证脚本
- 判据：模板 `identity.weights_id == "groupwise-int"` 且 `text/layers/3/` 下为 query_key+gate_value 两对象；制品尺寸精确 = 8,306,927,628 B、1126 对象；装配 15/15 全绿

### 阶段 4：端到端 + 性能（1 天）
- 判据：输出通顺；PPL 6.4–6.8（错则几百，一票否决）；decode T=1 ≥60 t/s；T≥2 走张量核；MTP K=2 对照 96.7–130.8 区间；记录 16GB 显存下的上下文容量

## 6. 磁盘与显存账本（J 盘版，峰值 ~94GB / 稳态 ~13GB）

| 物料 | 大小 | 生命周期 |
|---|---|---|
| Qwen3.8-27B 底座权重 | ~55GB | 产完模板即删 |
| 模板 .ninfer | ~20GB | 打包完归档或删 |
| PQ2_0/PTQ1_0 GGUF | 13.1GB | 打包完归档或删 |
| 三元 .ninfer 成品 | 7.74GB | 长期保留 |
| 引擎源码+构建 | ~5GB | 长期保留 |

显存：制品 7.74GB + KV/激活/MTP ≈ 8GB 余量；混合架构（16 全注意力+48 GDN）KV 压力小，KV 用 fp8 可再省；"8G 显卡流畅运行"佐证 16GB 无压力。

## 7. 风险清单

- **R1（最大变数）sm_120 兼容性**：作者只在 sm_89 验证过；mma.m16n8k16/cp.async 等 PTX 指令 Blackwell 理论全支持，但 arch 特定代码需实测。失败模式=编译报错或内核异常，届时先诊断出方案再动。
- **R2 模板链路长**：55GB 下载+转换+打包；评论区已有人卡壳，好在新版 pack.py 有提前拦截与指引（2026-09-21 起）。
- **R3 仓库活跃迭代中**：以仓库 HEAD 文档为准，论文 PDF 做背景；执行中若发现文档与代码矛盾，停下报告。
- **R4 CUDA 13.0 vs 论文 13.3**：大概率兼容，构建脚本报版本错再旁装。
- **R5 Ambolio 基线为 4090（Ada）线**：sm_89 写死程度未知，阶段 2 第一件事就是盘点 arch 相关代码。
- **R6 静默乱码坑**：主线 llama.cpp/LM Studio 加载会无警告输出乱码——任何验证前先确认二进制来自 PrismML fork。
- **R7 速度预期管理**：130.8 t/s 是峰值口径（任务敏感）；5080（960 GB/s）比论文卡（736 GB/s）带宽高 30%，但 sm_120 未经验证，实测中位数可能显著低于峰值，以验证矩阵数据为准。

## 8. 给 CODE 的讨论点（开放问题，欢迎挑战）

1. **模板依赖能否打破**：官方 GGUF 自带 mmproj（视觉权重），若改造 pack.py 从 GGUF 自举骨架+借 mmproj，可省 55GB 下载与 20GB 模板——代价是改作者工具链（违背"用标准脚本"原则）、需吃透 v1.0.8 制品 schema 全部细节。值不值？我方倾向：**不改**，55GB 是一次性成本，改工具链的风险不可控。
2. **sm_120 改造的最小改动面**：CMake arch flag 之外，是否还有 sm_89 硬编码（如 warp 尺寸假设、cache 配置、cp.async stage 数）？建议 CODE 侧独立盘一遍 patches/changed-files 清单。
3. **两步走顺序**：是否同意先方案0 拿基线？有无理由跳过方案0 直接上路线甲？
4. **CUDA 版本策略**：直接对齐 docs/01 指定版本（旁装 13.3），还是先赌 13.0？我方倾向：读 docs/01 后以作者指定版本为准，不赌。
5. **验证口径取舍**：论文 20 题基准 + PPL 全套是否照搬，还是按 §4.1 验证矩阵精简（PPL + 缺口区任务 + 真实负载分桶）？我方倾向矩阵版，PPL 一票否决不变。
6. **磁盘峰值复核算**：本账本按"底座产完模板即删"估峰值 94GB；若 CODE 侧发现额外大物料（如 DFlash2 必选），请修正。

## 9. 附录：材料来源

- 论文 PDF：J:\Bonsai\三元-Bonsai-27B-NInfer-移植-技术论文-20260920.pdf
- 魔搭仓：https://modelscope.cn/models/shensanshu/ninfer-ada-ternary （docs/patches/tools）
- 引擎基线：https://github.com/Ambolio/ninfer-4090-windows @ 6eb70a07（v1.0.8）
- 上游引擎：https://github.com/Neroued/ninfer （Apache-2.0，Linux/sm_120a）
- GGUF 权重：https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf （镜像 hf-mirror.com 可达）
- 底座：https://huggingface.co/Qwen/Qwen3.8-27B
- CSDN：https://blog.csdn.net/shensanshu/article/details/166012877
- B 站：https://www.bilibili.com/video/BV1dPeB6XE6d/
- 注意：HF 直连被拒（代理未开时），下载走 hf-mirror.com 或开启 FlClash 后直连；权重哈希校验不可省。