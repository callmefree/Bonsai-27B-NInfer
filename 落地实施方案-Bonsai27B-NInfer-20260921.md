---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '2ba0492b-e540-4b4d-b8e8-464c2293ced4'
  PropagateID: '2ba0492b-e540-4b4d-b8e8-464c2293ced4'
  ReservedCode1: '4eff7457-1d1f-4fd5-8e5b-9b0db3bfdd9a'
  ReservedCode2: '4eff7457-1d1f-4fd5-8e5b-9b0db3bfdd9a'
---

# Ternary Bonsai 2 27B + NInfer 本地落地方案

> 版本：v1.2（2026-09-21） | 状态：**双评审收敛定稿（CODE↔TELE 三轮交叉评审全部同意），待用户批准 M0 开工**
> v1.0→v1.1 修订映射 TELE 第二轮评审；v1.2 吸收 TELE 第三轮（机制澄清 + 路线 D 登记 + 社区情报），变更日志见 §10/§11。
> 目标：在本机 RTX 5080（16 GB，Windows）上完整复现"三元 Bonsai 2 27B → NInfer 引擎"全链路，跑通推理服务并验证论文"高智力 + 低显存 + 快"三重主张。

---

## 1. 一手资料清单

| 渠道 | 内容 | 状态 |
|---|---|---|
| 论文 PDF（本地 `j:\Bonsai\`） | 12 页技术报告，权威技术细节 | ✅ 已研读 |
| 魔搭 `shensanshu/ninfer-ada-ternary` | 工具链（~1.4 MB）：4 篇 docs + pack.py + MAPPING.json + verify/ + patches/，Apache-2.0，**无权重** | ✅ 已核实存在，2026-09-21 仍在更新 |
| B站视频 `BV1dPeB6XE6d` | 介绍视频（2026-09-20）：**"简易现行版本，不代表最终成品"**；只发复现细则与工具包，权重需自行组装；"还在开发中，相比发布版本已有更大进展" | ✅ 已抓取 |
| CSDN 博客 166012877 | 复现工程细则——**付费墙后**（0.47 元/天） | ⚠️ 仅摘要；免费替代=魔搭 docs/ |
| 夸克网盘 `4ce82c915551` | 论文文件 | ⚠️ 动态页需浏览器手动查看；本机已有论文 PDF，非阻塞 |

**作者实测口径（视频/博客摘要）**：
- RTX 4080 SUPER：显存占用 **6 GB**，130+ t/s
- RTX 5090：显存占用 **7 GB**，**220+ t/s**
- "8G 显存显卡流畅运行"（视频标题）
- CSDN 报 20 题 16/20 = 论文所述 low 档口径（xhigh 为 19/20），不矛盾

## 2. 本机环境实测（2026-09-21）

| 项目 | 现状 | 判定 |
|---|---|---|
| GPU | RTX 5080 16 GB（Blackwell **sm_120a**），驱动 616.92（UMD CUDA 13.4） | ⚠️ 与论文 sm_89 不同，但=上游原生架构 |
| CUDA Toolkit | 本机最高 v13.0（驱动支持 13.4） | 需补装 13.1+（装 13.4） |
| MSVC | 14.44（VS2022 Community） | ✅ 接近论文 14.51 |
| 内存 | 127.8 GB | ✅ |
| J 盘 | **252.4 GB 空闲**（已腾出） | ✅ 全程用 J 盘 |
| 工具 | git ✅；cmake/ninja 随 VS/CUDA 补 | ✅ |

## 3. 关键判断（相比初版评估的两处修正）

### 3.1 显存论证重写（v1.1，采纳 TELE 抓漏）
v1.0 用"作者实测 6–7 GB"推"余量充足"——**论证作废**：制品权重本身 7.74 GB，总占用不可能小于权重；"6G 显存占用"是营销口径（大概率指官方 PTQ1_0 打包 5.9 GB 或文本权重 6.7 GiB），不能作工程数据。

正确账目（分项）：
- 固定项：权重 7.74 GB + CUDA context/驱动 ~0.5–1 GB + MTP 状态与 workspace
- 可变项：KV 池——上游机制为"启动时按权重后剩余显存自动定容"[v1.0.8 是否同机制，**M0 核实**]
- **16 GB 卡预期**：总占用自动长到 ~14–15 GB（KV 池 ~6 GB）；262K 满上下文不可达，估 bf16 KV ≈ 90–100K token、fp8 KV ≈ 190–200K（16 全注意力层粗算，**待 M5 实测**）
- 8G 卡"流畅运行"与弹性 KV 池自洽：KV 池被压到 <1 GB → 短上下文流畅；占用数字与卡显存正相关，不存在单一"实测占用"

结论不变：**可行，余量够用**——但后续排查以分项账本为基线，M5 显存判据=三分项实测记录（权重/KV/其他）。

### 3.2 架构改写是唯一主线工程点（确认）
- 论文补丁绑定 `Ambolio/ninfer-4090-windows @ 6eb70a07`（v1.0.8 线，sm_89）
- 本机是 sm_120a（与上游原生支持的 RTX 5090 同架构）
- mma.m16n8k16 / cp.async / warp shuffle 均为 sm_80+ 通用指令 → 预期改动=CMake 架构标志级 + 少量调优常数重校
- **待验证**：M0 拉下补丁后立即确认绑定深度；若涉及 sm_89 专属内联 PTX 则工程量上修，回报重议
- 注意：上游 master（v1.2.0 线）目录结构不同，**不可直接套补丁**（README 明确警告）

### 3.3 版本漂移风险（v1.1 补强）
"简易现行版 ≠ 最终成品"，仓库仍在更新。v1.0 对策只防"仓库将来变"，防不了"**当前 HEAD 已与论文 PDF 不一致**"。补强（采纳 TELE）：
- **认 HEAD，不认论文**：执行以仓库 HEAD 文档为准，论文做背景
- M0 拉仓后先跑"文档 vs 工具链"一致性抽查（docs/03 判据 vs pack.py 实际参数），不一致以 HEAD 为准并记录差异
- 全程记录魔搭仓 fetch 的 commit hash（可锁）；每阶段开工前核对更新

## 4. 技术路线

**主线（路径 A）**：v1.0.8 基线 + 论文补丁 + 目标架构改写 sm_89 → sm_120a，完整复现三元全链路。

不选的路径（备查）：
- 路径 B（上游 master 跑 groupwise-int/nvfp4）：不认识三元格式，背离论文；且 nvfp4 27B ~14.5 GB 权重，16 GB 卡含 KV 必爆（TELE 独立证据，双重否决）
- 路径 C（PrismML fork llama.cpp）**v1.1 升级为"M6 前置基线"（不再是降级备选）**：同卡跑一次 PQ2_0（官方预编译，半天，7.2 GB）即得**同卡分母**，NInfer 加速比可量化——这是"NInfer 相对 fork llama.cpp 在 5080 上到底快多少"这一核心问题的唯一直接回答。速度预期修正：官方吞吐表 4090 PQ2_0=81.2 t/s、5090=129.9 → 5080（960 GB/s，接近 4090）估 **75–90 t/s**。⚠️ 必须用 PrismML fork 二进制：stock llama.cpp/LM Studio 拒载三元格式，主线加载 Q2_0 会**无警告输出乱码**（HF README 原话）
- 路径 D（KVMem fork 兜底，v1.2 新增，仅 NInfer sm_120 移植失败时启用）：CrKcel/kvmem-llama.cpp 已实证可跑 Bonsai（4060 8G PTQ1_0 ~20 t/s），核心 KV 卸载使小卡也能跑长上下文；日常主线仍是 NInfer（更快），长文档任务可切 KVMem 线（详见 §9 路线 D）

核心链路：
```
三元 GGUF ──pack.py──▶ .ninfer（7.74 GB / 1126 对象）
                             │
Qwen3.8-27B + DFlash2 ──convert──▶ groupwise-int 模板（打包底座 + MTP 头来源）
                             │
v1.0.8 基线 + 补丁 + sm_120a ──构建──▶ 引擎 ──▶ 服务/基准
```

## 5. 阶段计划与判据（不满足即停）

### M0 环境准备（0.5 天，零风险只读/拉取，已获批先行）
- CUDA：**读 docs/01 拿作者指定版本**；拿不到则装 **13.3**（论文验证值）；13.4 不作首选（上游验证 13.1，最新 toolkit 对老内核代码行为未知）；已有 13.0 可先试构建脚本报错再旁装
- 克隆 `Ambolio/ninfer-4090-windows` checkout `6eb70a07`（GitHub 走代理；**克隆到本地后脱网工作，此后不再依赖 GitHub**）；下载魔搭工具仓（国内直连，记录 commit hash）
- **绑定深度判据客观化**（v1.1，采纳 TELE）：对 patches/changed-files 全量 `grep __CUDA_ARCH__|sm_89` 逐处归类（arch 宏分支/内联 PTX/注释），输出清单回传 TELE 交叉核对——**grep 全零 = CMake 标志级，有内联 PTX = 停，回报工程量重议**
- 一致性抽查：docs/03 判据 vs pack.py 实际参数，**认 HEAD 不认论文**，差异记录
- 确认 PrismML fork releases 是否含 Windows 预编译（方案0 前提）
- 核实 v1.0.8 引擎长上下文机制：是否内建"Resource-aware long-context reuse"（Device/Host 双层 KV，--host-kv-mib/--host-state-slots）+ KV 池是否"按剩余显存自动定容"（上游 v1.2.0 已内建，v1.0.8 待核实，直接决定 16GB 卡上下文上限与 §3.1 显存账本）
- 判据：四件套版本记录在案（nvcc ∈ {docs/01 指定值, 13.3}）；基线源码树完整；arch 归类清单产出

### M1 权重获取（1–2 天，下载 ~70 GB）
| 文件 | 来源 | 大小 | 渠道 |
|---|---|---|---|
| PTQ1_0.gguf + PQ2_0.gguf（两种都下，装配互证用） | HF prism-ml/Ternary-Bonsai-2-27B-gguf | 5.9 + 7.2 GB | HF 需代理 / hf-mirror 已实测可达 |
| Qwen3.8-27B（bf16，模板底座） | **ModelScope 优先**（Qwen 官方镜像直连） | ~54 GB | 国内直连 |
| Qwen3.8-27B-DFlash2 | ModelScope | ~1–2 GB（估） | 国内直连 |
- **DFlash2 表述对齐**（v1.1）：MTP 头来源两说（底座自带 NextN 层 vs DFlash2 仓提供），`--dflash2-model` 参数角色**以 docs/02 为准**；下载清单按"必选"预案，docs/02 证实可选则回退
- **判据**（v1.1 前移，采纳 TELE）：下载完成立即跑 GGUF 元数据 5 判据（block_size=1024、Σsign_widths=28672、weight_names=401、inverse=[token_embd]、gdn_v_grouped=true）+ 哈希对照官方——哈希证"文件对"，5 判据证"格式认知对"，二者缺一不可；齐了才进 M2

### M2 模板生成
- 基树 convert：`--model Qwen3.8-27B --dflash2-model <path> --out template.ninfer`（参数以 docs/02 实际为准）
- **判据**：`identity.weights_id == "groupwise-int"`；`text/layers/3/` 为两个分离对象；模板含 `mtp/*` 12 张量；**勿用 convert_nvfp4**
- **验收边界声明**（v1.1，采纳 TELE）：模板两判据只证 schema，证不了 borrow 来的 vision/MTP 数值正确性——该风险由 M5 端到端 PPL/输出比对兜底；**视觉不在 NInfer 线验收范围**（该 fork 不支持图片输入，论文 §7.2），视觉验收在方案0 线做

### M3 打包与验制品（1–2 天）
- GGUF 元数据 5 判据：已前移至 M1（此处复验通过即可）
- `pack.py check` → `pack.py build`；产物精确 **8,306,927,628 B / 1126 对象**
- verify/ 4 脚本全绿（装配 15/15）
- **教训铁则**（论文 §5）：常规校验全绿 ≠ 正确，四级验证一个不许跳

### M4 引擎构建（1 天，最大变数）
- 先按 docs/01 原样编译基线通过，再改架构 flag 重编（先试 `sm_120a` 与上游一致，NVCC 报不识别再降 `120`）
- 补丁落盘；改 `.h` 后 **touch 所有相关 .cpp**（MSVC 假"没活干"→ 0xC0000409）
- **判据**（v1.1 补强）：① BUILD OK 且确认改动确实重编（注入无效改动验证构建系统未缓存）；② **`cuobjdump --list-elf` 确认产物含 sm_120 SASS**——编出 sm_89 PTX 靠 JIT 也能跑但性能崩，构建成功 ≠ 产物 arch 正确

### M5 端到端验证
- PPL ≈ 6.4–6.8（错则几百）；输出通顺 + 贪心逐字比对
- **备注**（v1.1）：PPL 走 prefill，**测不到 T=1 decode**（论文 §5.3 原话）——T=1 正确性由贪心逐字比对单独保证，两判据不可互相替代
- **显存判据**（v1.1 重写）：分项账本实测记录（权重 / KV 池 / 其他），并实测可用上下文上限（对照 §3.1 预估：bf16 ≈ 90–100K / fp8 ≈ 190–200K）
- 前置待办：PPL 计算语料口径从 docs/04 确认（论文"黄金参照 6.8196±0.5365"语料未注明）
- **判据**：PPL 落带内且贪心比对一致；否则按论文"修一个真 bug 看数字动多少"定位法排查

### M6 速度标定与基准
- **前置**：方案0 同卡基线（PrismML fork + PQ2_0，官方预编译，半天）——M6"达标"以**同卡分母**为准，不做跨卡外推（v1.1 作废"4080S 130 → 5080 应不低于"的论证：带宽差 30% + 架构差，无依据）
- decode（T=1）与 T≥2 张量核分别实测；等效带宽按 flush-L2 口径；**预热后多次取中位数**（论文 §5.4 idle 时钟陷阱）
- **MTP 接受率按任务分桶记录**（中文短答 / 英文代码 / 长上下文）——MTP 错误只表现为变慢不变错，接受率崩是唯一信号
- 20 题基准（可选，报数须附思考档）
- 收工：显存回落基线、无残留进程

## 6. J 盘目录规划（峰值 ~140 GB / 可用 252 GB）

```
J:\Bonsai\                      ← 工作区（方案/文档/记忆，已有论文）
J:\Bonsai\landing\
  ├── repos\ninfer-4090-windows\   ← 引擎基线 @6eb70a07 + 补丁
  ├── tools\ninfer-ada-ternary\    ← 魔搭工具仓
  ├── weights\                     ← 三元GGUF + Qwen3.8-27B bf16 + DFlash2（~60 GB）
  ├── artifacts\                   ← 模板制品 + 三元制品（~8 GB）
  └── build\                       ← 引擎构建树 + 测评输出
```

## 7. 风险登记

| # | 风险 | 等级 | 对策 |
|---|---|---|---|
| R1 | 补丁深度绑定 sm_89（内联 PTX） | 中（正确性风险≈0，性能重校为主，TELE 独立核查） | M0 grep 归类清单判定；有内联 PTX 则停，回报评估或等作者新版 |
| R2 | 5080 显存/调度差异致性能不达标 | 低 | sm_120a=上游原生；调优常数（CTA/K分块）按 5080（SM 84）重校，不影响正确性 |
| R3 | 仓库持续更新致版本漂移 | 中 | **认 HEAD 不认论文**；M0 一致性抽查；锁 commit hash；每阶段开工前核对 |
| R4 | 长上下文 OOM | 低 | 分项账本基线；fp8 KV 开关；上下文上限 M5 实测（bf16 估 90–100K） |
| R5 | CSDN 细则为付费内容 | 低 | 魔搭免费 docs 为权威；付费解锁可选（0.47 元） |
| R6 | 权重许可限制 | — | 仅自用，制品不分发；遵守 PrismML/Qwen 条款 |
| R7 | 网络/代理依赖（v1.1 新增） | 中 | M0 基线 clone 到本地脱网；hf-mirror.com 兜底（已实测可达）；魔搭/ModelScope 直连 |
| R8 | stock llama.cpp/LM Studio 静默乱码（v1.1 新增） | 低（预防性） | 任何验证前确认二进制来自 PrismML fork；主线拒载三元格式且 Q2_0 无警告乱码 |

## 8. 待决策项

1. ~~方案路线确认（路径 A）~~ ✅ 已定（用户确认路线甲 + 独立实验/开发机定位）
2. ~~CUDA 13.4 安装~~ ✅ 策略已改：以 docs/01 为准，13.3 首选
3. CSDN 付费细则是否解锁（非必需，魔搭 docs 已覆盖）
4. 批准 M0 开工（TELE 裁决"放行 M0 无需等待修订"，M0 零风险只读/拉取）

## 9. KVMem 与 NInfer 的关系评估（v1.1 新增，回应用户问题）

**KVMem 是什么**：llama.cpp fork（github.com/kvmem/kvmem-llama.cpp），核心=KV 缓存分层搬运（显存/内存/磁盘），16G 卡跑 27B & 256k 上下文（5060ti 30–40 t/s）。Windows CUDA 12/13 预编译版覆盖 20–50 系。

**与本方案的关系**：
0. **机制层面无需移植（v1.2 澄清）**：KVMem 的 KV 分层搬运思想，上游 NInfer v1.2.0 官方 README 已有"Resource-aware long-context reuse"节（Device/Host 双层 KV + --host-kv-mib/--host-state-slots 状态检查点 + 按恢复成本调度的淘汰机制）——即 KVMem 同款理念的官方实现。关键未知是论文基线 v1.0.8 线是否含此机制（**M0 核实项**）；若含，16GB 卡长上下文能力已内建，无需移植 KVMem。
1. **代码级移植到 NInfer：不可行也不值得**。两者代码异源（llama.cpp fork vs 从零写的 NInfer），KV 管理层 API/内存模型完全不同，"结合"=重写。且在本方案链上再叠 KVMem 等于第四次魔改（上游→Ambolio Windows 线→作者三元补丁→KVMem 分层），错误面爆炸，违背"标准工具链+可证伪验证"原则（与"不改 pack.py"同一逻辑）。
2. **思想级借鉴：列为 M6 后未来方向**。若 M5 实测上下文上限（bf16 估 90–100K）不够用，可单独立项评估 KV 分层扩展（锚点=上游已内建 Device/Host 分层），走新变更流程。
3. **务实替代优先**：fp8 KV 是上游现成开关，比引入 KVMem 便宜一个量级；先实测够不够用。
4. **KVMem 已知代价**（评论区实录）：长上下文后段注意力下降、猜词幻觉（UP 主自认）、部分用户报 33k 后掉速——KV 卸载不免费，长上下文质量衰减是所有方案共性痛点（NInfer 的 Host KV 同样会有），不是 KVMem 独有缺陷。
5. **社区生态速记**（方案0 变体/路线 D 实证）：CrKcel/kvmem-llama.cpp（Bonsai 适配魔改，4060 8G 跑 PTQ1_0 ~20 t/s，证实 KVMem fork 是 NInfer sm_120 失败时比 PrismML 官方 fork 更强的兜底——多 KV 卸载能力使 8G 卡也能跑）；社区拼装 `Qwen3.8-27B-Ternary-PQ2_0-MTP.gguf`（7.66G，5060Ti 实测"不弱于 GSQ-IQ3_S 且更快"，**官方仓无此文件、来源待考**，方案0 线可做可选实验）。社区对智力评价分歧大（"不如 9B" vs "绝对比 9B 强"）——与官方"选择性衰减、casual testing 测不出"结论互证，**验证矩阵必须含缺口区任务**。
- **路线 D 正式登记（v1.2 吸收 TELE 第三轮）**：NInfer sm_120 移植失败时的兜底路线——沿用 KVMem/PrismML fork 跑 Bonsai，靠 KV 卸载在 16GB 卡跑长上下文；日常主线仍是 NInfer（更快），长文档任务可"软共存"切换（双引擎显存互斥，按任务加载，代价=重载权重）。是否启用待 NInfer 线跑通、拿到真实上下文上限后再定。

## 10. v1.0 → v1.1 变更日志（映射 TELE 第二轮评审）

| TELE 评审条目 | 处置 | v1.1 落点 |
|---|---|---|
| 显存论证"6-7GB 物理不可能" | **接受**（附口径澄清：若指权重占用可自洽，但作工程预期无效） | §3.1 重写、M5 分项账本 |
| M4 补 cuobjdump --list-elf | 接受 | M4 ② |
| M1 5 判据前移 + 文件清单明确 | 接受 | M1 |
| M2 视觉验收边界声明 | 接受 | M2 |
| M5 PPL 测不到 T=1 备注 | 接受 | M5 |
| M6 idle 时钟预热 + 接受率分桶 | 接受 | M6 |
| M6 跨卡外推无依据 → 同卡基线 | 接受（路径 C 升 M6 前置） | §4、M6 |
| R7 网络依赖 + 脱网策略 | 接受 | M0、R7 |
| CUDA 13.4 降为不首选 | 接受 | M0 |
| 版本漂移"认 HEAD 不认论文" | 接受 | §3.3、M0 |
| 绑定深度判据客观化（grep 归类） | 接受 | M0 |
| DFlash2 角色待 docs/02 | 双方立场收敛：按"必选"预案准备，docs/02 一锤定音 | M1 |
| 交叉核对 #4（6-7GB） | 接受修正 | §3.1 |
| 交叉核对 #8（路径 C 速度偏保守） | 接受（60-75 → 75-90） | §4 |
| sm_120a 命名与 flag 细节 | 接受（先 120a 后 120） | M4 |

## 11. v1.2 变更日志（吸收 TELE 第三轮，2026-09-21）

| TELE 第三轮条目 | 处置 | v1.2 落点 |
|---|---|---|
| KVMem 机制澄清（上游 v1.2.0 已内建 Device/Host 双层 KV，Resource-aware long-context reuse） | 接受，补机制层（无需移植 KVMem） | §9 第0点、M0 核实项 |
| DFlash2 表述纠偏（勿把推断写事实） | 已对齐（v1.1 M1 已写"角色待 docs/02"，本轮确认） | M1 |
| PQ2_0 不可平移 NInfer 线 | 接受 | §4、M1 |
| 视觉题移方案0 线 | 接受 | M2、§9 |
| PPL 语料口径待办（真缺口，M5 前必须补） | 接受 | M5 |
| 路线 D 登记（KVMem fork 兜底） | 接受，正式登记 | §4、§9 |
| 社区情报（CrKcel / 社区 MTP GGUF / KVMem 代价） | 接受，入册 | §9 |

**总体裁决：CODE↔TELE 三轮交叉评审全部收敛，本方案定为 v1.2 定稿。M0 全程只读/拉取（拉仓→读 docs→盘 arch 清单→回传交叉核对→核实 v1.0.8 KV 机制），零风险，待用户一句话即开工。**

## 12. M0 执行结果（2026-09-21，CODE 执行，详见 `docs/reports/M0-核实报告-CODE-20260921.md`）

| M0 核实项 | 结果 | 影响 |
|---|---|---|
| 基线锁定 | NInfer HEAD=`6eb70a0784b68c87efcbeb0b08bd2c3a95914492`（v1.0.8 线，commit msg 确认）；remote=Ambolio/ninfer-4090-windows | 已锁，R3 缓解 |
| 绑定深度（R1） | `__CUDA_ARCH__` 0 匹配；内联 PTX 仅 1（注释）；架构处理=CMake 分支（TMA 仅 120a / w8_config capped sm_89） | **R1 中→低**，放行 M1 |
| v1.0.8 KV 机制 | **已内建 Device/Host 双层 KV**（`--host-kv-mib`/`--host-state-slots`/`core/host_kv_arena.h`） | 推翻"待核实未知"，16GB 卡长上下文原生具备；R4 实质缓解；路线 D 必要性下降 |
| docs↔工具链一致性 | pack.py 解码与 ggml-quants.c 逐字一致；MAPPING.json 标 2 条 INFERRED 置换；verify 脚本 zero_share 0.3278 | 自洽，认 HEAD |
| CUDA 版本 | 13.3.33 旁装（与方案 M0 对齐） | — |
| DFlash2 角色 | docs/02 明确"可选草稿头" | **v1.3 候选**：M1 必选→可选 |
| 视觉矛盾 | 上游有 vision 基础设施，论文 §7.2 称三元线不支持 | **v1.3 候选**：M2 仍移方案0，M1 实测待核对 |
| PPL 语料口径 | docs/04 未给具体语料；PATCHES 给终值 6.445 | M5 前置待办仍成立（论文 PDF 不可解析） |
| 路径 C（PrismML fork） | 确认 release `prism-b10658+` 覆盖三元三档 | 非阻塞，M6 前置确认 Windows 预编译 |

**M0 判定：放行 M1。** 方案 §3.1 "KV 池自动定容" 措辞不准确（实为显式参数化 --host-kv-mib）。**v1.3 候选修订点**：A 经 M2 源码终审已证伪（方案 M1 必选正确，DFlash2 `required=True` 见 convert.py:340）——撤回；B（视觉矛盾）/C（PPL 语料口径）待 M2/M5 确认。**M1 已 PASS（TELE 报告落盘），M2 进行中。**

---

## 13. M7：CraneBW tensor-core 内核合并（2026-09-22，TELE 执行，用户批准）

- 合并 CraneBW/ninfer-ternary-bonsai-ada 的优化提交（prefill tensor-core GEMM / verify small-T kernel / rotation launch packing / GDN 参数化）至本仓 ternary 目录，+TELE 适配（DFlash2 verify T≤8 gate）。
- **验证全过**：PPL 6.112648 vs M5 黄金 6.1129（差 0.005%）；贪心 A/B 一致（DFlash2 T=8 新旧 verify 路径逐字一致；ref vs mma 长贪心分歧为 fp16 scale 折入的预期内数值漂移）。
- **性能（M6 同口径 T=0 中位）**：MTP K=1 decode 72.0→**104.0 t/s（+44%）**；DFlash2 en 104.7→**139.5（+33%）**；prefill score rate 295.2→**712.2（2.41x）**、7.6K prompt TTFT 口径 **1288 t/s**。
- 详见 `docs/reports/CraneBW内核合并验证报告_TELE_20260922.md`（方案 `CraneBW内核合并方案_TELE_20260922.md`）；回退三层（env 开关 / 备份 exe `landing\backups\build_5080_pre_cranebw_20260922` / `_author_backup` 源码留档）。
- 遗留候选：MTP K=2 复评（verify 提速后临界移动）、5080 调优参数扫描、DFlash2 Full 头 A1/A2/A3。
- **遗留处置（2026-09-22 追扫后，用户批准）**：①K=2 转正（zh 109.3）；②SMALL_T_ROWS 默认即最优；③fp8 K=2 全场最优（zh 114.3/en 118.9），**日常档 BAT 已更新为 `起服-日常档fp8.bat`**（fp8+K=2+并发2 复核通过：双流 wall 2.71s、峰值 10.2GiB），旧 bf16 档归档 `landing\backups\bat_pre_fp8_20260922\`；④DFlash2 Full 头按 **A3** 处置（文档记录"DFlash2 必须加 `--lm-head-draft`"，DFlash2 BAT 注释与 M6 报告 §4.1 已载；A1/A2 代码修复方案保留在 M6 报告待需要时立项）；⑤作者版 3 个源码留档移至 `landing\backups\cranebw_author_backup_20260922\`。