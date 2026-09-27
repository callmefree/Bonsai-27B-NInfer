---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'e645e175-6a10-4dd1-969a-4c31a39079a6'
  PropagateID: 'e645e175-6a10-4dd1-969a-4c31a39079a6'
  ReservedCode1: '124cccdc-6ab0-4a8e-a095-21030f904d28'
  ReservedCode2: '124cccdc-6ab0-4a8e-a095-21030f904d28'
---

# Bowsai-NInfer — 本 fork 说明

> 本 fork 目标：在 **RTX 5080（16GB）** 上跑通 **Bowsai-2-27B 三元量化模型**，并把 prefill / decode / 长上下文压到这台卡的极限。
> 仓库名沿用上游 "`-4090-windows`"，但**本 fork 实测平台是 5080**，别被名字误导。
> 本说明只写**我们做了什么**；上游作者的工作请看上游仓库，不在这里复述。

---

## 一、我们做了什么（相对上游的全部增量）

| commit | 做了什么 | 结果 |
|---|---|---|
| `50af9d1` | M4+M7 落地：三元制品接入 + CraneBW 内核合并 | 5080 跑通 Bowsai 三元模型，PPL 达标 |
| `10723b4` | 移植上游 A/B（CUDA sync switch + Q5 split4 band） | 编译 172 目标、PPL 无退化 |
| `513949d` | **移植 s8（int8）+ wide_t prefill 内核** | 冷 prefill **1.63k tok/s**（prefill 主推档）|
| `145bccb` | **frontend 思考区空收尾修复（方案 C）** | 长思考不再"想完无正文" |
| `59ef76c`/`f5d1d2e` | 本 fork 专属说明 | 现在这份 README |

配套项目侧（Bowsai 主库 `main`）：
- **启动器 GUI**（`ninfer_launcher.py`）：图形化配置 上下文/KV/投机/思考预算/prefill 档，保存 profile
- **上下文档位**：新增 150K/180K/200K，修正 224K=229376
- **一套起服 BAT**：日常 fp8 / k8v4-224K / k8v4-256K-MTP / DFlash2 英文

## 二、开发历程（摘要，全史见 Bowsai 主库 `docs/项目构建史.md`）

约 36 小时双 AI 协作，从论文到落地的完整过程：

- **M0–M4（09-21）**：环境核实 → 下载权重 → 模板制品 → 三元打包 → 5080 构建（CUDA 13.3 + sm_120a SASS）
- **M5（09-22）**：端到端验收，**PPL 6.112947**，显存账本三场景
- **M6**：三档速度标定（裸 decode 68 / DFlash2 英文 104.7）
- **M7**：CraneBW 内核合并，prefill 600→1288 t/s，decode K1 zh +44%
- **M7 后**：s8/wide_t 内核、启动器 GUI、180/200K 档位、思考空收尾修复

## 三、实测成果（5080 16GB）

- **PPL 6.1129**（低于判据带 6.8，达标）
- **性能矩阵**：日常 fp8 K2 **114.3 t/s** · k8v4-224K 95.7 · k8v4-256K-MTP **105.2**（终极档）· DFlash2 英文 **139.5**
- **长上下文**：k8v4 256K 满窗实测，显存 15.36GB（余 ~900MB）
- **prefill**：s8 档冷 prefill **1.63k tok/s**

## 四、怎么复现 / 怎么用

1. **代码**：本引擎仓库（`engine-main` 分支）+ Bowsai 主库（`main`，docs/论文/启动器）
2. **模型**：`Ternary-Bonsai-2-27B.ninfer`（约 10.5 GiB，Apache-2.0 可公开），或按 M1–M3 自行打包
3. **构建**：CUDA 13.3 + VS2022 + Ninja → `cmake --build _build_5080 --target ninfer-serve -j`
4. **起服**：`ninfer-serve <model>.ninfer --port 18787 ...`（档位参数见 Bowsai 主库 BAT）
5. **调用**：OpenAI / Anthropic 兼容 HTTP

## 五、给复现者的实话（教训）

- 三元量化（2.125bit）对代码/长任务基本无损，且把 27B 塞进 16GB 卡——这条路是可行的。
- **版本代际要对齐**（v1/v2/v3 头号坑），引擎只认本代制品。
- 长上下文靠 host-KV 分页，但 Agent 任务要防"贫瘠输入 → 思考死循环"（详见思考修复 commit `145bccb`）。
- 所有改动都在 git 历史里，可逐条回退、逐条追溯。

---

**数据与过程均来自本机实测 / `docs/项目构建史.md`，不是营销话术。**