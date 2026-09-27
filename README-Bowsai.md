---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '2ac25f0b-5510-4a9f-b19a-914a658fc431'
  PropagateID: '2ac25f0b-5510-4a9f-b19a-914a658fc431'
  ReservedCode1: 'ccd14864-e642-4160-af96-5f1bdbb910c5'
  ReservedCode2: 'ccd14864-e642-4160-af96-5f1bdbb910c5'
---

# Bowsai-NInfer — 本 fork 说明（非上游原样）

> 本文件是本 fork 的**项目专属说明**，区别于上游作者的 README。上游 README 讲的是原作者
> （Neroued / UDPSendToFailed 等）在 RTX 4090 上的移植；**本 fork 是 Bowsai 三元量化模型的
> 本地推理部署，实测平台为 RTX 5080**，两者不同，请勿按仓库名「4090」误判。

---

## 一、本 fork 是什么

- 基础：fork 自 `Ambolio/ninfer-4090-windows`（NInfer 的 Windows 移植）
- 目标：本地运行 **Bowsai-2-27B 三元量化模型（`Ternary-Bonsai-2-27B.ninfer`）**
- 实测硬件：**RTX 5080 16GB**（构建目录 `_build_5080`）
- 推理：文本/图像/视频，OpenAI / Anthropic 兼容 HTTP API

> ⚠️ **名字纠正**：仓库沿用了上游"`-4090-windows`"的名字，但**本 fork 实测是 5080**。
> 需要兼容 5080，命名是历史沿用，不代表只支持 4090。

## 二、模型

- 制品：`Ternary-Bonsai-2-27B.ninfer`（约 10.5 GiB）
- 基础：Qwen3.8-27B（**Apache-2.0** 许可，可公开分发/衍生）
- 量化：三元（Ternary）+ 折叠基变换（PTQ1_0 / PQ2_0，逐字节无损打包）
- 视觉：内置视觉塔，`--vision` 直接加载，无需额外 mmproj

## 三、相对上游/作者分支的改动（本 fork 特有）

| commit | 内容 |
|---|---|
| `50af9d1` | baseline：固话 M4+M7 引擎改动（三元移植 + CraneBW 内核合并） |
| `10723b4` | upstream A/B：CUDA sync switch + Q5 split4 band |
| `513949d` | 移植 wide_t 与 **s8 int8 prefill** 内核（作者三元树），prefill 提速 |
| `145bccb` | **frontend 思考区空收尾修复**：模型思考区直接 stop 时强制进入正文区 |

## 四、构建

- 工具链：VS2022 + CUDA 13.1 + Ninja
- 构建目录：`_build_5080`（sm_89）
- 命令：`cmake --build _build_5080 --target ninfer-serve -j`

## 五、复现与完整文档

完整开发过程（M0–M8）、构建方案、性能报告、启动器使用见：
- 私库 `Bowsai-27B-NInfer` 的 **`main`** 分支（文档、开发史、报告、论文）
- 本 fork 的 **`engine-main`** 分支（引擎代码）

## 六、已知说明

- 本 fork 与上游功能同构，但面向 Bowsai 三元模型与 5080 调优，非作者原样分发。
- 与原作者仓库的差异与本 fork 改动以 git 历史为准，不在此逐条复制。