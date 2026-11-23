# VLA autotune pipeline：工具池、按游选工具、worktree 对比

这份文档说明 FlashRT 里 agent 驱动的 VLA autotune 是怎么搭起来的，对应和学长讨论时的三层：

| 层 | 是什么 | 在仓库里的位置 |
|---|---|---|
| tool | 某一游（stage）上现成的优化方法或 autotune 项目 | `flash_rt/vla/autotune/tools/`（注册表 + 外部项目 adapter） |
| skill | 人沉淀下来的优化流程，告诉 agent 怎么用这些 tool | `.agents/skills/`（Claude Code 经 `.claude/skills/` 软链接访问） |
| loop | profile → 选 tool → 实验 → 再 profile 的闭环 | 总 skill `vla-autotune`，加上 controller / batch / worker |

**一句话：**先测准，再把时间归到“游”。每一游先从工具池挑现成的 autotune 方法；没有现成方法的游，用沉淀下来的 skill 做；每一步都记录是人还是 agent 做的决定。

## 一、整体流程

```text
定目标（人）
  → baseline：同一硬件上的官方实现 + 当前 FlashRT 配置
  → profile，按 stage 归因（stages attribute）
  → 写诊断（症状 → 假设 → 下一步观测）
  → 选 stage 和 tool（tools list / show）
  → 生成候选：
        config lane  改配置，不开 worktree（tools propose / levers）
        patch lane   改源码，每个候选一个源码快照（snapshots）
        port lane    把方法或后端移植进 FlashRT 原生路径
        kernel lane  ForgeKernel / KernelAgent 写 kernel → 算子级门槛 → 接进 patch
  → campaign：reference → 候选 → 复测（controller / batch / worker）
  → 决策：accept / reject / keep，记录谁决定的（decisions）
  → 对新 baseline 重新 profile，进入下一轮
```

人在 4 个地方拍板：

- **H1** 定目标和质量门槛；
- **H2** 进入 patch / port / kernel lane，或者使用近似方法；
- **H3** 启动外部长任务（ForgeKernel、KernelAgent）；
- **H4** 合入主线。

## 二、代码模块

| 模块 | 作用 |
|---|---|
| `stages.py` | stage 地图：每个模型每一游对应哪段代码，profile 区间怎么归到 stage，Amdahl 上限 |
| `tools/` | 工具注册表：内置的 config 工具、被挡住的方法（附原因和代码行号）、外部项目 adapter |
| `tools/levers.py` | 多个旋钮的组合实验：每个单独开、全开、全开去掉一个，用来看旋钮之间有没有耦合 |
| `tools/patches.py` | 把一个 worktree 里的改动变成 content-addressed patch 和对应的候选 |
| `tools/capture.py`、`artifacts.py` | 在 GPU 上抓调用点的形状和张量；定义 kernel 和 patch 的产物格式 |
| `tools/external/` | ForgeKernel、KernelAgent、ApxInf、AITune 的 adapter：生成任务文件和作业脚本，收集结果。adapter 本身不执行外部工具 |
| `snapshots.py` | 源码快照：每个 patch 候选一个 git worktree，带身份校验 |
| `decisions.py` | 决策日志：每条都写明 actor（human / agent / human+agent / search），能导出“谁主导”的步骤表 |
| `controller.py`、`batch.py`、`worker.py` | 原有的 campaign 流程，增加了快照模式（plan schema 3） |

## 三、游（stage）

| stage | 含义 |
|---|---|
| `host.input` | 预处理、分词、H2D |
| `vision.encode` | 视觉编码器（FastWAM 是首帧 VAE） |
| `vision.project` | 投影层 / merger |
| `prefix.prefill` | VLM / prefix 前向，产出 KV cache（FastWAM 是 video expert prefill） |
| `denoise.velocity` | 一次 action expert 调用 |
| `denoise.schedule` | 调用次数（Euler N / ProbeFlow） |
| `denoise.routing` | 步与步之间 host 上的路由和积分 |
| `host.launch` | CPU 下发 kernel 的空隙 |
| `host.output` | 拷回 CPU 和解码 |
| `kernel.*`、`precision.weights` | 跨 stage 的 kernel 类别和权重精度 |

各模型能拆到什么程度，由 profile 区间决定：

- π0 / π0.5：Euler + CUDA Graph 只有一个不可拆的 `full` 区间，要拆就另做诊断请求。
- Hy：有 `vision`、`prefix`、`velocity` 三个区间。
- FastWAM：有 `condition_vae_video_prefill` 和 `velocity` 两个区间。
- DM0.5：`vision` 包含视觉编码、池化和投影，`prefix` 从视觉输出开始，`velocity` 记录十次动作专家调用。整图模式用独立诊断图插入计时事件。每个请求返回绝对 FP32 CPU `[50,14]` 动作。

```bash
python -m flash_rt.vla.autotune.stages list --model hyvla
python -m flash_rt.vla.autotune.stages attribute result.json --rank
```

## 四、工具池

```bash
python -m flash_rt.vla.autotune.tools list --all --markdown
```

| 类型 | 工具 | 状态 |
|---|---|---|
| 配置（现成） | `sampler.euler_steps`、`sampler.probeflow`、`launch.cuda_graph`、`gemm.fastwam_linear_backend`、`compile.fastwam_inductor`（FastWAM 的 torch.compile）、`prefix.vision_pool`、`prefix.state_prompt_mode`、`precision.hy_numerical_profile` | ready |
| 分析 | `profile.stage_intervals`、`profile.pi05_dataflow`、`analysis.lever_pricing` 为 ready；`profile.ncu_kernel` 需要 ncu 权限 | — |
| 官方基线 | `baseline.official_impl`：π0 / π0.5 / FastWAM 用 `official_native_latency.py`，Hy 用 `official_hyvla_latency.py`，DM0.5 用 `official_dm05_latency.py` | ready |
| DM0.5 配置 | `compile.dm05_inductor` 生成 Torch 编译候选，`launch.cuda_graph` 可从 eager 基线生成开图候选。固定 BF16、三视角、Euler10 | ready |
| 现成的外部项目 | `kernel.forgekernel`（OpenBMB/ForgeKernel）、`kernel.kernelagent`（meta-pytorch/KernelAgent）、`engine.apxinf`（infinigence/ApxInf，只支持 π0.5）、`compile.aitune`（NVIDIA AITune） | needs_setup |
| 需要原生扩展的 SM80 配置 | `fusion.hy_rope_qknorm_kvwrite`、`quant.int8_pi05` | needs_setup，`tools list --all` 可见，`tools propose` 可先生成候选 |
| FastWAM 请求内缓存 | `cache.fastwam_text_crossattn_kv` 在附加当前 proprio 后计算各层文本 K/V，供同一请求内所有去噪步使用 | ready |
| 在 A800 上做不了 | `quant.fp8`、`quant.nvfp4`；以及语义会变的 `cache.pi05_cross_frame_kv` 和面向吞吐的 `throughput.pi05_batch` | blocked |

外部项目在 A800 上要先做的准备：

- **ForgeKernel**：A800 不在原生支持范围，需要 5 处补丁（`tools.external.forgekernel.sm80_enablement()` 会列出来）。agent 只能用 Claude Code，每个算子要几小时 GPU。
- **KernelAgent**：需要单独的 venv 和 LLM 凭据。它返回第一个通过测试的 kernel，不保证是最快的。优化流程需要 ncu，A800 要用 `gpu_name="NVIDIA A100 SXM4 80GB"` 代替。
- **ApxInf**：需要 Rust + CUDA 构建，只覆盖 π0.5，A800 未验证。
- **AITune**：见 `tools show compile.aitune`。

## 五、worktree 对比怎么实现

以前候选里的 `source_ref` / `patch_ref` 只是标签，所有候选其实都跑同一份代码。现在有了快照模式：

1. **patch**：在独立的 git worktree 里改代码，`proposal_from_worktree` 把改动存成 `sha256:` patch。
2. **materialize**：`snapshots materialize` 在快照库里建一个 detached worktree，打上 patch，生成确定性的 commit，最后写 manifest。
3. **native**：
   - 纯 Python 改动用 `adopt-native` 复用编译好的 `.so`，前提是 `csrc`、CMake 等原生输入完全相同；
   - 改了 `csrc` 就用 `build-script` 渲染编译作业，编译完再执行 `record-built`。
4. **init**：`controller init --snapshot-store` 写出 plan schema 3。每个 attempt 绑定一个快照，快照同时登记在 ledger 里。
5. **batch**：先重新核对所有快照。每个 attempt 以自己的快照为 cwd，且 PYTHONPATH 上只有这一份 FlashRT。
6. **worker**：加载模型之前核对实际 import 的源码，包括 commit、tree、父 commit、harness tree、工作区是否干净、`.so` 的 sha256。对不上就记 `source_identity` 失败。

测量 harness（`flash_rt/vla/autotune/`）是固定的：patch 不能改它，也不能在仓库根目录加 Python 模块；同一个 campaign 里所有快照的 harness tree 必须相同。这防止候选**无意中**改动计时方式和质量指标。

要注意的是，候选在 `flash_rt/` 下的代码和 harness 跑在同一个进程里，而且会在源码校验之前被 import，所以这条规则防的是无意改动，不是故意篡改。

config 候选不开 worktree，仍走原来的 plan schema 2，行为和以前完全一样。

## 六、skill

| 类别 | skill | 作用 |
|---|---|---|
| 总流程 | `vla-autotune` | 主循环、human checkpoint、按游路由、停止规则 |
| 分 lane | `vla-stage-profile`、`vla-config-sweep`、`vla-patch-candidate`、`vla-kernel-candidate` | 各 lane 的具体步骤和命令 |
| 改编自 ApxInf | `vla-model-impl-map`、`vla-operator-gate`、`vla-copy-drift-audit`、`vla-port-workflow` | 静态执行地图、算子级门槛、漂移审计、移植流程 |
| 原样引入 | `forge-start`（ForgeKernel）；`perf-torch-cuda-graphs`、`perf-torch-sync-free`、`perf-nsight-systems`、`perf-nsight-compute-analysis`、`perf-workload-profiling`、`kernel-triton-writing`（TensorRT-LLM）；`kernel-numerics-verifier`（torchtitan）；`aitune-inspect`、`aitune-tune`、`aitune-benchmark`、`aitune-validate`（AITune） | 各游通用的现成方法 |

- 第三方 skill 都保留了上游原始字节、`LICENSE.txt` 和 `NOTICE`。逐字节副本、来源和校验脚本在 `third_party/agent-skills/`。
- 调研过但没有引入的 skill（没有许可证、只支持 AMD、绑定特定服务框架等）也列在那里，附了原因。
- Codex 从 `.agents/skills/` 读取 skill；Claude Code 从 `.claude/skills/<name>` 读取，这些都是指回 `.agents/skills/<name>` 的软链接。

## 七、记录谁主导

```bash
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind decision --actor human \
  --action redirect --redirected-from "agent 想继续调 GEMM" --tool launch.cuda_graph --stage host.launch \
  --summary "人纠偏：GEMM 改为 CUDA Graph" \
  --rationale "示例：插桩请求的 kernel 时长合计远小于不插桩的请求中位数，推测空隙来自 CPU 下发"
python -m flash_rt.vla.autotune.decisions summary --log $AT/decisions.jsonl --markdown
```

导出的步骤表带 actor 列，可以直接贴进报告，回答“哪些步骤是人引导的、哪些是 agent 自己做的”。

## 八、实验范围与环境

- Hy 配置实验已在 A800 作业 `356212` 中跑完整轮，使用 7 个输入、每配置 35 次正式计时和首尾两次基线。项目需要分别记录带源码快照的实验和各外部工具的实测情况。
- `controller init --author-baseline /absolute/author.json` 将作者实现纳入同一轮实验，支持现场运行和绑定已有正式输出。所有候选及首尾基线同时检查项目误差 `quality` 和作者误差 `author_quality`。具体参数见[作者基线流程](../examples/vla_autotune/author_baseline.md)。
- ForgeKernel、KernelAgent 和 AITune 的外部环境按各自 adapter 的 `setup` 字段部署。FastWAM、DM0.5 的 ApxInf 历史运行边界见[五模型延迟表](reports/2026-11-14-vla-latency-comparison.zh-CN.md)。

### 从已测配置生成候选

```bash
python -m flash_rt.vla.autotune.tools propose --tool cache.fastwam_text_crossattn_kv \
  --baseline fastwam.json --out fastwam-text-kv.json
python -m flash_rt.vla.autotune.tools propose --tool fusion.hy_rope_qknorm_kvwrite \
  --baseline hy.json --out hy-fused.json
python -m flash_rt.vla.autotune.tools propose --tool quant.int8_pi05 \
  --baseline pi05-euler.json --param 'modes=["encoder","encoder_decoder"]' --out pi05-w8a8.json
```

Hy 在 `-DGPU_ARCH=80 -DFLASHRT_ENABLE_HYVLA=ON` 构建中使用独立目标 `flash_rt_hyvla_sm80`。π0.5 在 `-DGPU_ARCH=80 -DENABLE_SM80_INT8_CUTLASS=ON` 构建中使用 `flash_rt_kernels` 的 W8A8 路径。运行 π0.5 候选时需要移除部署脚本导出的 `FVK_PI05_RTX_FORCE_BF16`。`encoder` 量化前缀投影，`encoder_decoder` 还量化动作专家投影，两种配置均使用动态逐行激活尺度。
