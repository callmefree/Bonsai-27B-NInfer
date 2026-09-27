---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'f2f41a40-ba2d-4de6-a2ce-4e80bf206218'
  PropagateID: 'f2f41a40-ba2d-4de6-a2ce-4e80bf206218'
  ReservedCode1: '2d81b0e1-a343-420d-bf85-fef7d86eaa6a'
  ReservedCode2: '2d81b0e1-a343-420d-bf85-fef7d86eaa6a'
---

# Bowsai-NInfer — 本 fork 说明

> 本 fork 用于在 **RTX 5080（16GB）** 上运行 **Bowsai-2-27B 三元量化模型**（`Ternary-Bonsai-2-27B.ninfer`）。
> 仓库名沿用上游 "`-4090-windows`"，但**本 fork 实测平台是 5080，勿按名字误判**。
> 完整开发过程见私库 `Bowsai-27B-NInfer` 的 `main` 分支 `docs/项目构建史.md`（本文档为摘要+复现入口）。

---

## 一、项目是什么

- 目标：把作者（沈三殊）《三元-Bonsai-27B-NInfer-移植技术论文》的 **Bowsai-2-27B 三元量化 + NInfer 引擎**落地到本机，验证"高智力 + 低显存 + 高速度"的三元推理效果。
- 制品：`Ternary-Bonsai-2-27B.ninfer`（约 10.5 GiB，Q2=PQ2_0 2.125bit，权重 6.70 GiB）
- 模型根基：Qwen3.8-27B（**Apache-2.0**，可公开分发/衍生）+ 折叠基变换（PTQ1_0/PQ2_0 逐字节无损打包）+ 三元量化
- 实测硬件：**RTX 5080 16GB**（sm_89 编译路径，CUDA 13.1/13.3，VS2022，Ninja）

## 二、上游血统（本 fork 站在谁的肩膀上）

| 角色 | 仓库 | 贡献 |
|---|---|---|
| 官方上游 | `Neroued/ninfer` | C++20/CUDA 引擎、DFlash2、ReplaySSM、Paged KV |
| 4090 fork 创始 | `UDPSendToFailed/ninfer-4090` | E8 lattice / rk4v4-e8 KV、WDDM bypass |
| Windows 整合 | `Ambolio/ninfer-4090-windows` | v1.0.8，5 fork 融合（本 fork 基线）|
| 三元补丁作者 | `shensanshu/ninfer-ada-ternary`（ModelScope） | 三元铺丁 + 技术论文 |
| 优化内核 | `CraneBW/ninfer-ternary-bonsai-ada` | prefill 提速内核（M7 合并）|

## 三、开发历程（约 36 小时，双 AI 协作，M0–M8 + 后续优化）

| 里程碑 | 内容 | 关键结果 |
|---|---|---|
| M0 环境 | 双仓锁定、R1 架构评估 | 确认可在 5080（sm_120a）编译 |
| M1 权重 | 下载底座 + 量化 GGUF + DFlash2 | 5.7~7.2GB 三档 |
| M2 模板 | bf16 模板制品 | 19.03 GiB / 1190 对象 |
| M3 打包 | 三元制品 + shim | 9.81 GiB / 1192 对象，6 项验证链全 PASS |
| M4 构建 | CUDA 13.3 + 铺丁 + 编 5080 | sm_120a SASS（cuobjdump 验证）|
| M5 验收 | PPL + 显存 + 速度 | **PPL 6.112947**，三场景显存账本 |
| M6 标定 | 三档速度 | 裸 decode 68 / MTP / DFlash2 英文 104.7 |
| M7 合并 | CraneBW 内核合并 | prefill 600→1288，decode K1 zh +44% |
| 后续 | s8/wide_t 内核、启动器 GUI、思考修复 | 见下 |

## 四、本 fork 特有改动（相对上游）

| commit | 内容 |
|---|---|
| `50af9d1` | baseline：M4+M7 引擎改动（三元 + CraneBW）|
| `10723b4` | upstream A/B（CUDA sync switch + Q5 split4 band）|
| `513949d` | 移植 **s8（int8）+ wide_t prefill** 内核，冷 prefill **1.63k tok/s** |
| `145bccb` | **frontend 思考区空收尾修复**：思考区 stop 时强制进正文区（方案 C）|

## 五、关键成果（5080 实测）

- **PPL 6.1129**（1,044,557 tokens）——低于论文判据带上限 6.8
- **性能矩阵**（起服档位）：日常 fp8 K2 114.3 t/s · k8v4-224K 95.7 · k8v4-256K-MTP 105.2（终极档）· DFlash2 英文 139.5
- **长上下文**：k8v4 256K 满窗实测，KV 分页 4096/4096，显存 15.36GB（16GB 卡余 ~900MB）
- **并发推理**：s8/wide_t/shm 派发三档互斥，prefill 快速档

## 六、如何复现（简要）

1. **获取代码**：本引擎仓库（engine-main 分支）+ Bowsai 主库（docs、论文、启动器）
2. **获取模型**：`Ternary-Bonsai-2-27B.ninf`（HF/ModelScope，Apache-2.0）或按 M1–M3 流程自行打包
3. **构建**：CUDA 13.3 + VS2022 + Ninja，`cmake --build _build_5080 --target ninfer-serve -j`
4. **起服**：`ninfer-serve <model.ninf> --port 18787 ...`（各档位参数见 Bowsai 主库 BAT）
5. **调用**：OpenAI / Anthropic 兼容 HTTP

## 七、参考价值 / 教训（给复现者）

- 三元量化（2.125bit）在代码生成/长任务上基本无损，且显存占用大幅降低——16GB 卡可跑 27B 满血长上下文。
- 版本间制品"代际"必须对齐（v1/v2/v3 头号坑），引擎只认本代。
- 长上下文靠 host-KV 分页兜底，但 Agent 任务要防"贫瘠输入→雷霆思考→失忆循环"。
- 本 fork 所有优化均有 git 历史可溯，勿以"仓库名 4090"判断支持范围（实为 5080）。