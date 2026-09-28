# T10 待机降功耗机制调研（fishensw/VLLM-SM75）

> 调研对象：<https://github.com/fishensw/VLLM-SM75>（v0.1.6，2026-09）。
> 该项目是 Turing/SM75 专用 vLLM 0.30.0 分支（T4 / T10），整合 FlashInfer 0.6.18 与 SM75 后端补丁，4×T10 TP4 生产实测（131K ctx prefill ≈ 978 tok/s）。
> 其降功耗为**两级体系**：P-State 常驻省电（模型/KV 不动）与自动休眠（释放显存/退出引擎），均经 4×T10 实测。

## 机制一：P-State 常驻省电（v0.1.5+ 默认，`POWER_MODE=pstate`）

原理：**不释放任何 GPU 资源**，只把 T10 的性能状态钳在低档。T10 空闲 P8 实测 **9.97–15.31 W/卡**（满载 TDP 150W）。

- **底层工具 `nvidia-pstated` v1.0.9**（[sasha0552](https://github.com/sasha0552/nvidia-pstated)，GPL-3.0）：守护进程逐 GPU 检查温度/利用率，用 `libnvidia-api.so.1`（**NVAPI**，非 NVML）设置 P-state、NVML 读状态，默认 100ms 轮询。关键参数 `-psl`（低档）/`-psh`（高档）。
- **容器化适配**：pstated 上游声明"必须在宿主层跑、不能进容器"；该项目把匹配宿主驱动的 `libnvidia-api.so.1` 通过 `PSTATE_NVAPI_LIB` 挂进容器解决，NVML 由 NVIDIA Container Toolkit 的 `utility` 注入。
- **监督层 `docker/helpers/pstate-supervisor.sh`**：自建活跃判断、托管 pstated 内置阈值（`-ut 0 -ibs 1`）：
  - 活跃判定：`nvidia-smi` 利用率 ≥5%（`PSTATE_UTIL`）**或** vLLM `/metrics` token 计数变化；
  - 活跃 → `-psl 16 -psh 16`（**16 = 交还驱动自动控制，不是硬件 P16**）；
  - 空闲 1800s（`PSTATE_IDLE_TIMEOUT`，秒）+ 60s 确认（`PSTATE_CONFIRM`）→ `-psl 8 -psh 16`（钳下限 P8）；
  - 安全护栏：拒绝 `-psl 0 -psh 0`（会把 T10 钉死 ~645MHz）。
- **入口 `pstate-entrypoint.sh`**：后台监督器 + 前台 `vllm serve`；容器停止时先恢复 16 再退出。一个 GPU 只跑一个 P-State 控制器。

## 机制二：引擎自动休眠（v0.1.4，`POWER_MODE=sleep` + `auto_sleep.py`）

- **核心问题**：上游 vLLM 空闲时 EngineCore 主循环阻塞在 `input_queue.get()`，循环内无法查超时；sleep/wake 又只能由外部 `POST /sleep`、`/wake_up` 触发。
- **解法（补控制平面）**：用一次性 `_idle_state_callbacks` 观察"引擎进入空闲"→ 起 daemon `threading.Timer` 计时 → 到点后 `input_queue.put_nowait(WAKEUP)` 哨兵把决策交回主循环线程（sleep/wake 只允许在该线程执行）→ 下一个请求**透明唤醒**，无需 API 调用。
- **四档 offload target**：
  - `cpu`（level 1）：权重复制到 pinned RAM，唤醒 1–2s，需等量主机内存；
  - `reload`（level 2）：丢弃权重，唤醒 `reload_weights` 重跑加载管线（含量化 repack），20–60s；
  - `exit`（深睡）：**EngineCore 进程退出**，释放权重 + CUDA 上下文 + TP workers → GPU 稳进 P8；API server 常驻，下个请求冷启动重建 1–3min（由该请求 TTFT 承担）；
  - `disk`（实验）：快照显存分配原位恢复，KV 按协议失效。
- **唤醒加速**：`posix_fadvise(POSIX_FADV_WILLNEED)` 把 safetensors 预热进 OS page cache；reload 模式睡眠期间每 600s 后台重预热（exit 模式退出前一次性预热，page cache 跨进程存活）。
- **实测**（T10×4，FP8 DFlash2，2026-09-08）：60s 自动 exit → 四卡连续三次 P8、显存 3 MiB、9.97–15.31 W/卡、`/health` 200；两个并发请求完整唤醒 135.7s；AOT 编译缓存 12 处全命中、零重编译。

## 两级对比

| | P-State 常驻 | 自动休眠 exit |
| --- | --- | --- |
| 显存 / KV | 全保留 | 全释放（3 MiB） |
| P8 保证 | 低负载时钳入（取决于驱动/其他进程） | 确定性进入（CUDA 上下文已释放） |
| 唤醒 | 无感（P-state 即时回升） | 1–3 分钟冷重建 |
| 额外内存 | 无 | 无（exit 不留 pinned 备份） |
| 适用 | 常驻服务 | 闲置时段 / 主机内存紧张 |

## 对本项目（ninfer sm_75 / T10）的启示

1. T10 待机 **~10–15 W/卡 @P8** 可达，两条路径都有 4×T10 实测背书。
2. **P-State 路线引擎无关**（纯驱动层 NVAPI），ninfer 引擎零改动即可借用：`nvidia-pstated` + 活跃监督脚本可直接搬到 T10 主机（Linux 侧），临时散热上机验证时顺手装上即可。
3. exit 模式的"进程退出换 P8"依赖引擎可安全退出重建，ninfer 侧如需同等效果要走系统级（杀进程/停服务）而非引擎内建。
4. P8 前提：无其他进程持有 CUDA 上下文；`nvidia-smi` 显存不为绝对 0 属正常。
