# CraneBW 内核合并验证报告（TeleAgent）

> 执行：TeleAgent，2026-09-22 03:00–04:30（用户"开始"信号后）
> 方案：`docs\reports\CraneBW内核合并方案_TELE_20260922.md`（§4.2 合并清单 + §5 验证计划）
> 结论先行：**合并成功，全部验证通过，三档性能全面超额达标。合并后引擎已在 18787 恢复运行（k8v4 Agent 档）。**

---

## 一、总体结论

| 验收项 | 方案目标 | 实测 | 判定 |
|---|---|---|---|
| 编译 | 0 error | 0 error（3 exe 链接成功） | PASS |
| 数值等价（PPL 全量对账） | 与 M5 黄金差 ≤0.05% | **6.112648 vs 6.1129（0.005%）** | PASS |
| prefill 吞吐 | ≥1200 t/s | **1288.4 t/s**（7.6K token TTFT 口径，k8v4 档） | PASS |
| prefill 吞吐（PPL score rate 同口径） | 提升 | **295.2 → 712.2 tok/s（2.41x）** | PASS |
| 日常档 decode（MTP K=1, zh） | ≥85 t/s（基线 72.0） | **104.0 t/s（+44%）** | PASS |
| DFlash2 en decode | ≥120 t/s（基线 104.7） | **139.5 t/s（+33%）** | PASS |
| DFlash2 T=8 gate 修复 | verify 不掉 SIMT 且输出正确 | 新旧路径贪心输出**逐字一致** | PASS |

## 二、合并内容（实际落地）

全部在 `J:\Bonsai\landing\repos\ninfer-4090-windows\src\ops\linear\ternary\`：

| 文件 | 动作 | 说明 |
|---|---|---|
| ternary_rowsplit_mma.cuh | 新增（19718B） | CraneBW prefill tensor-core GEMM |
| ternary_rowsplit_mma_small_t.cuh | 替换（20522→21444B） | CraneBW verify small-T kernel；作者版留档 `.author_backup` |
| ternary_rowsplit_gemm.cu | 替换（8850→~21KB）+ 适配 A | 派发重构 + **DFlash2 T≤8 gate（TELE 新增段，注释标明）**；作者版留档 |
| ternary_rowsplit_gemv.cuh | 替换（8469→~11KB） | tile_block SIMT GEMV（fallback 路径）；原版留档 |
| ternary_rotation.cu / _kernels.cuh | git apply | rotation launch packing（纯调度，数值不变） |
| gated_delta_net/recurrent.cu / .cuh | git apply | GDN record MinBlocks 参数化（默认=原行为） |
| ternary_rowsplit_gemm.cuh | 注释更新 | 新架构与等价回退开关说明 |

未触碰（按方案）：layouts_impl.h（含 bad_alloc 修复）、program_impl.h、CMakeLists×2、storage/launch/dispatch/rotation.h。适配 B 未改代码（NINFER_TERNARY_MMA=0 的等价组合 = `NINFER_TERNARY_PREFILL=block` + `NINFER_TERNARY_VERIFY=tile`）。

**执行偏差记录**：首次编译失败（12 error：`ternary_pq2_gemv_tile_block_kernel` 未定义）——方案 §4.2 清单漏列 gemv.cuh（审核时标注为"fallback 路径"而未入清单，属方案编制失误非审核失误）。补合后通过。已按 T-VERIFY 记录三要素。

## 三、验证证据

### 3.1 编译 [EVIDENCE]

- 首编：`ninja -C _build_5080` → `ternary_rowsplit_gemm.cu(164): error: identifier "ternary_pq2_gemv_tile_block_kernel" is undefined` ×12（单一根因）
- 补合 gemv.cuh 后：`[4/6] Linking CXX executable apps\ninfer-serve.exe ... [6/6] ninfer-perplexity.exe`，0 error
- 产物：`_build_5080\apps\{ninfer, ninfer-serve, ninfer-perplexity}.exe` 2026-09-22 3:31 构建；已同步至 `_build_5080\` 根（BAT 引用路径），根目录旧版（0:24）已备份

### 3.2 引擎侧贪心 A/B（作者法）[EVIDENCE]

| 测试 | 配置对比 | 结果 |
|---|---|---|
| 短输出（~280 tok prefill，MTP K=1） | ref / mma / block / vtile 四配置 | 输出全部 `fox`，**逐字节一致** |
| 长生成（~1.4K tok prefill，220 tok，MTP K=1） | ref vs mma | 前 13 token 一致后**贪心分歧**（见下） |
| 长生成 DFlash2 T=8（gate 专项） | small_t(新 gate) vs tile(回退) | **逐字一致**（305B），DFlash2 统计正常（22.2% 接受，2.56 tok/round） |

长生成 ref vs mma 分歧定性：分叉点前两段完全一致（"The fox watched the dog from behind a large oak tree. "），分叉后双方均为流畅英文（无乱码/循环）。根因 = CraneBW prefill 将 fp16 scale 折入解码值（相对差 5-6e-3，其 README 自述），长贪心下累积漂移跨越 argmax 平局点——**预期内数值特性，非布局错误**（若 tile/layout 错，首 token 即错）。权威判据按作者法为 PPL 对账（3.3）。

### 3.3 PPL 全量对账 [EVIDENCE]

- 命令：`ninfer-perplexity.exe <artifact> --corpus eval\corpora\perplexity-1m\manifest.json --kv-dtype bf16 --output <dir>`（M5 同协议 context 4096/stride 2048）
- **合并后（CraneBW mma）：overall PPL 6.112648**，四域 chinese 9.1237 / en_long 10.3019 / en_ref 7.8991 / code 1.8662
- M5 基线（作者 mma 路径）：**6.1129** → 差 **0.005%**（判据 ≤0.05%）✓
- score rate：M5 295.2 → 合并后 **712.2 tok/s**（同口径 2.41x）
- 小样本（wikitext-00，65304 tok）：mma 31 窗终值 **7.8208**；ref 中途 27 窗 7.9361（33 t/s，被 30min 超时截断；ref 全量约 8.8 小时不具备可行性）——替代证据链：①全量 M5 对账 0.005% ②CraneBW 上游自证 0.015%（110 窗）③贪心一致性。**ref 全量对账未完成，如实披露。**

### 3.4 性能复测（M6 同口径：T=0 greedy，预热后中位数）[EVIDENCE]

| 档位 | 基线（M6，作者路径） | 合并后 | 提升 | 证据 |
|---|---|---|---|---|
| 日常档 bf16 + MTP K=1，zh，256 tok | 72.0 t/s | **104.0 t/s** | **+44%** | bench_one 3 run：103.7/104.0/105.3 |
| 日常档 en | （M6 未单测） | **113.1 t/s** | — | 3 run：113.1/110.4/115.0 |
| DFlash2 K=7 + lm-head-draft，en | 104.7 t/s | **139.5 t/s** | **+33%** | 3 run：139.5/140.4/135.1 |
| prefill（k8v4 档，7583 tok prompt TTFT 法） | M6 未测（间接 ~295 PPL rate） | **1288.4 t/s** | 达标 ≥1200 | TTFT 5.886s |
| prefill（1751 tok，含固定开销） | — | 1197.7 t/s | — | TTFT 1.462s |

serve 端实测（合并后 k8v4 256K、bf16 32K 均正常加载，5s 内就绪）。

## 四、运行状态与资产

- **18787 端口当前运行**：k8v4 256K Agent 档（合并后引擎，后台 job-20260921-212130-2fa477c5，PID 9244）——与用户鹈鹕前状态一致，DSH 可直连
- 回退资产（三层）：
  1. 运行时：`NINFER_TERNARY_PREFILL=block` + `NINFER_TERNARY_VERIFY=tile`（零重建回落 SIMT）
  2. 二进制：`J:\Bonsai\landing\backups\build_5080_pre_cranebw_20260922\`（3 exe，合并前版本）
  3. 源码：3 个 `.author_backup` 留档于 ternary 目录 + git 状态记录（HEAD 6eb70a07）
- worktree `.temp\cranebw-baseline` 已清理；CraneBW 克隆仓库保留于 `landing\repos\cranebw-ninfer-ternary-bonsai-ada`（其 bench/linear_bench 可作后续 kernel 级扫描工具）

## 五、遗留与建议（不阻塞，待用户裁决/择机）

1. ~~**MTP K=2 复评**~~ → **已完成，见 §六**：K=2 转正
2. ~~5080 调优参数扫描~~ → **已完成，见 §六**：默认参数即最优
3. 3 个 `.author_backup` 留档：建议下一批源码整理时移入 `_safety_backups` 或删除
4. 上轮遗留待裁决项不变：DFlash2 Full 头修复 A1/A2/A3、日常档并发 2 保留与否
5. 三个 BAT 注释可补一行"CraneBW merged 20260922"（本次未动 BAT，避免与用户工作流冲突）

## 六、追加扫描：MTP K 复评 + 调优参数 + fp8 KV（2026-09-22 04:30–05:00）

口径与 §3.4 完全一致（T=0 greedy，预热后 3 run 中位数，bf16 32K 日常档；fp8 组仅换 `--kv-dtype fp8`）：

| 配置 | zh t/s | en t/s | 备注 |
|---|---|---|---|
| bf16 K=1（现行 BAT） | 104.0 | 113.1 | 验证报告 §3.4 值 |
| **bf16 K=2** | **109.3** | 114.6 | **K=2 转正**（合并前 68 白干；verify 提速后临界移动） |
| bf16 K=3 | 104.7 | 117.7 | zh 已回落，临界在 K=2~3 |
| NINFER_TERNARY_SMALL_T_ROWS=16 | 99.3 | 107.3 | 差于默认 |
| NINFER_TERNARY_SMALL_T_ROWS=48 | 91.7 | 96.8 | 更差 |
| **fp8 KV K=1** | 105.5 | 116.4 | 略快于 bf16，KV 显存减半 |
| **fp8 KV K=2** | **114.3** | **118.9** | **全场最优** |

结论：
1. **K=2 转正**：M6 "K=2 白干"结论基于作者旧 verify 路径，合并后失效——新最优 draft 为 K=2（zh +5.1% vs K=1）。
2. **CraneBW 默认 SMALL_T_ROWS=32 在 5080 上即最优**（16/48 均变差），无需任何调参；`NINFER_TERNARY_ROTATE_WPB` 默认自适应逻辑（decode 形状已走 1 warp/block）无调整空间，未测。
3. **fp8 KV 双赢**：decode 再 +4~5%，且 KV 显存减半（32K 档省 ~1GiB；同显存长上下文上限 bf16 ~120K → fp8 ~238K，CraneBW 6/6 召回到 192k fp8 佐证质量）。
4. ~~建议的日常档新参数~~ → **已更新，见 §七**：`--kv-dtype fp8 --spec mtp --draft-tokens 2`（zh 114.3，较合并前原日常档 72.0 = **+59%**）。
5. 扫描脚本与原始输出：`.temp/m6/sweep/`（summary.json + 各配置 serve 日志）。

## 七、遗留处置落地（2026-09-22 05:20–05:40，用户批准"更新，并完成遗留"）

1. **日常档 BAT 已更新**：新 `起服-日常档fp8.bat`（fp8 KV + MTP K=2 + 并发 2，注释含扫描数据/回退开关）；旧 bf16 档归档 `landing\backups\bat_pre_fp8_20260922\`。
2. **fp8+K=2 并发 2 复核 PASS**：双流请求同完（wall 2.71s，140/160 tok），显存峰值 10.2 GiB（余 ~6 GiB）。
3. **DFlash2 Full 头按 A3 处置**（M6 报告建议项）：不修代码；用法限制"DFlash2 必须加 `--lm-head-draft`"已载于 DFlash2 BAT 注释 + M6 报告 §4.1 + 主文档 M7 节；A1/A2 代码修复方案保留在 M6 报告，需要时再立项。
4. **k8v4 / DFlash2 BAT 注释更新**：补 CraneBW merged 标注；DFlash2 档数字 104.7 → 139.5。
5. **作者版源码留档归位**：3 个 `_author_backup` 从 ternary 源码树移至 `landing\backups\cranebw_author_backup_20260922\`（源码树恢复干净，ternary 目录现全部为 CraneHEAD 版 + TELE gate）。
6. **最终运行状态**：18787 = k8v4 256K Agent 档（PID 7480）；日常 fp8 档由用户按需用新 BAT 启动。