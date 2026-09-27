---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '0d105a08-81d1-411e-bd7f-969332b45317'
  PropagateID: '0d105a08-81d1-411e-bd7f-969332b45317'
  ReservedCode1: '065faa73-3178-431d-82a6-48151894c0da'
  ReservedCode2: '065faa73-3178-431d-82a6-48151894c0da'
---

# Bowsai-NInfer — RTX 5080 三元量化推理

> 本仓库目标：在 **RTX 5080（16GB）** 上运行 **Bowsai-2-27B 三元量化模型**（`Ternary-Bonsai-2-27B.ninfer`），
> 并把 prefill / decode / 长上下文压到这台卡的极限。
> 仓库名沿用上游 "`-4090-windows`"，但**本 fork 实测平台是 5080**，别被名字误导。
>
> 本 README 只写**我们做了什么、怎么跑、值不值得抄**；上游作者的工作请看上游仓库，不在此复述。
> 完整开发过程见 Bowsai 主库 `docs/项目构建史.md`。

---

## 一、我们做了什么（相对上游的全部增量）

| commit | 内容 | 结果 |
|---|---|---|
| `50af9d1` | M4+M7 落地：三元制品接入 + CraneBW 内核合并 | 5080 跑通 Bowsai 三元模型，PPL 达标 |
| `10723b4` | 移植上游 A/B（CUDA sync switch + Q5 split4 band） | 编译 172 目标、PPL 无退化 |
| `513949d` | **移植 s8（int8）+ wide_t prefill 内核** | 冷 prefill **1.63k tok/s**（prefill 主推档）|
| `145bccb` | **frontend 思考区空收尾修复（方案 C）** | 长思考不再"想完无正文" |

配套项目侧（Bowsai 主库 `main` 分支）：
- **启动器 GUI**（`ninfer_launcher.py`）：图形化配置 上下文/KV/投机/思考预算/prefill 档，保存 profile
- **上下文档位**：新增 150K/180K/200K，修正 224K=229376
- **一套起服 BAT**：日常 fp8 / k8v4-224K / k8v4-256K-MTP / DFlash2 英文

## 二、开发历程（摘要，全史见 Bowsai 主库 `docs/项目构建史.md`）

约 36 小时双 AI 协作，从论文到落地的完整过程：

- **M0–M4**：环境核实 → 权重下载 → 模板制品 → 三元打包 → 5080 构建（CUDA 13.3 + sm_120a SASS）
- **M5**：端到端验收，**PPL 6.112947**，显存账本三场景
- **M6**：三档速度标定（裸 decode 68 / DFlash2 英文 104.7）
- **M7**：CraneBW 内核合并，prefill 600→1288 t/s，decode K1 zh +44%
- **M7 后**：s8/wide_t 内核、启动器 GUI、180/200K 档位、思考空接收修复

## 三、实测成果（5080 16GB）

- **PPL 6.1129**（低于判据带 6.8，达标）
- **性能矩阵**：日常 fp8 K2 **114.3 t/s** · k8v4-224K 95.7 · k8v4-256K-MTP **105.2**（终极档）· DFlash2 英文 **139.5**
- **长上下文**：k8v4 256K 满窗实测，显存 15.36GB（16GB 卡余 ~900MB）
- **prefill**：s8 档冷 prefill **1.63k tok/s**

## 四、怎么构建 / 复现

1. **代码**：本引擎仓库（`engine-main` 分支）+ Bowsai 主库（`main`，docs/论文/启动器）
2. **模型**：`Ternary-Bonsai-2-27B.ninfer`（约 10.5 GiB，Apache-2.0 可公开），或按 M1–M3 流程自行打包
3. **构建**：CUDA 13.3 + VS2022 + Ninja → `cmake --build _build_5080 --target ninfer-serve -j`
4. **起服**：`ninfer-serve <model>.ninfer --host 127.0.0.1 --port 18787 ...`（档位参数见 Bowsai 主库 BAT）
5. **调用**：OpenAI / Anthropic 兼容 HTTP（流式、工具、token 计数）

## 五、能力与边界

**能力**（引擎本身支持，非本 fork 新增）：文本/图像/视频多模态、思考与非思考模式、分块预填充、MTP3/DFlash2 投机解码、bf16/int8/fp8/格点 KV、前缀缓存与设备/主机 KV 分页、离线困惑度评分、OpenAI/Anthropic API。

**边界**：单 GPU + 常驻单模型；启动固定并发 1–8；FIFO 入口，无抢占/优先级；工具由客户端执行。

## 六、给复现者的实话（教训）

- 三元量化（2.125bit）对代码/长任务基本无损，把 27B 塞进 16GB 卡——这条路可行。
- **版本代际要对齐**（v1/v2/v3 头号坑），引擎只认本代制品。
- 长上下文靠 host-KV 分页，但 Agent 任务要防"贫瘠输入 → 思考死循环"（见 commit `145bccb`）。
- 所有改动都在 git 历史，可逐条回退、逐条追溯。

---

## 许可与归属

- 本 fork 基于 **Neroued/ninfer**（上游，Apache-2.0）+ Windows/4090 生态各分支（上游 lineage）。本仓库为**派生**，遵守 Apache-2.0 §4：保留原始版权与归属声明；上游技术档案/性能基准见历史版本 `README.md`。
- 模型源自 **Qwen 团队（阿里云）**，Apache-2.0 许可（模型权重不随本仓库分发，按 Qwen 许可自行下载）。
- 第三方依赖（FFmpeg 等）保留各自许可证。
- **数据与性能均来自本机实测 / `docs/项目构建史.md`，非营销话术。**