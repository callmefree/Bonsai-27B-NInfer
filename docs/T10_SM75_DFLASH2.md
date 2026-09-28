# NInfer 三元内核 sm_75 (Tesla T10 16G) 移植 — DFlash2 可行性与 cc 门修复

## 结论

- **DFlash2 投机解码在 sm_75 / Tesla T10 (16G) 技术上可行**。
- cc 运行时门已从 `{120,89}` 放宽到 `{120,89,75}`，提交 `b92be4a` 已合入 `callmefree/Bonsai-27B-NInfer@sm75-port`，GitHub Actions 交叉编译（CUDA 13.1, sm_75）**通过**（run `36415264883`，conclusion=success）。
- 运行期四道门（起得来 / 数值 / 质量 / 性能）待 T10 上机执行（见 §5 验证清单）。

## 1. 前置静态验证结论（四项）

| # | 项 | 判定 | 证据 |
|---|---|---|---|
| 1 | cc 运行时门（致命→已修） | ✅ 已放宽到 75 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h`：`validate_target_options` 末尾 `if (cc != 120 && cc != 89)` → 追加 `&& cc != 75`。T10(cc=75) 不再在启动期抛异常 |
| 2 | dflash2 子系统 sm_75 兼容 | ✅ | `propose_dflash2_batch` 复用已移植标准 op；专属 op `linear_topk` / `candidate_selector` / `prepare_masked_block` 均无 fp8/s8/MMA 硬依赖 |
| 3 | fp8 红线 | ✅ 被 `--lm-head-draft` 化解 | `linear_topk.cpp::resolve_profile`：强制 lm-head-draft 时走 **W8Full (W8G32_F16S)** = 8bit 权重组量化 + fp16 累加，图灵兼容；不强制才走 FP8 optimized head 报错 |
| 4 | KV dtype | ⚠️ 参数约束 | `layouts_impl.h` 把 `rk4v4/rk4v4-e8` 硬门控到 cc==89；T10 必须 `--kv-dtype bf16` |
| 5 | 显存 | ✅ 非瓶颈 | 三元 dflash2 制品 9.10 GiB 权重 + bf16 KV@32K ≈ 13–14 GB < 16 GB |

## 2. cc 门补丁

文件：`src/targets/qwen3_6/impl/runtime/layouts_impl.h`

```diff
-    if (device.compute_capability() != 120 && device.compute_capability() != 89) {
+    if (device.compute_capability() != 120 && device.compute_capability() != 89 &&
+        device.compute_capability() != 75) {
         throw std::invalid_argument("Qwen3.6 family runtime requires compute capability 12.0 or 8.9");
     }
```

> rk4v4 门控（cc==89）无需改动；T10 启动不传 `--kv-dtype rk4v4` 即可。

## 3. T10 部署启动参数

```
--spec dflash2 --draft-tokens 7 --lm-head-draft --kv-dtype bf16
```

- K 工程甜点 5/7（K≤7；K=8 起验证宽度越过小批档，单轮成本悬崖）。
- 与 `--vision` 互斥。

## 4. 编译验证

- 仓库：`callmefree/Bonsai-27B-NInfer`，分支 `sm75-port`
- 提交：`b92be4a` runtime: relax Qwen3.6 cc gate to also allow sm_75
- CI：`sm75-compile.yml`（CUDA 13.1 交叉编译 sm_75 + 校验无 cp.async）
- 结果：run `36415264883` = **success**（Configure / Build ninfer_ops 全绿）
- 链接：https://github.com/callmefree/Bonsai-27B-NInfer/actions/runs/36415264883

## 5. 运行期验证清单（T10 上机后）

用 `bonsai2_27b_ternary_v2-dflash2.ninfer` 制品 + 上述参数起服务，过四道门：

1. **起得来**：不报 cc 运行时门 / unsupported head profile
2. **数值门**：三元权重下 FP16 MMA 输出与 sm_89 参考对比，`__trap` 分支未误触
3. **质量门**：可核对答案 + 负控变红
4. **性能门**：dflash2 decode 相对基础推理加速比，目标 +15–25%

> KV 必须走 bf16（rk4v4/rk4v4-e8 仅 Ada 8.9，sm_75 不可用）。详细运行期清单见资料库 fD14 §6。

## 6. 关联

- 资料库归档：`NInfer 三元内核 → Tesla T10 (sm_75) 移植归档`（folder `t17PcRtlORzQLUL0jBk6Nl`）
- sm_75 编译收官文档：fD14
- 部署技术储备报告：emQ11
