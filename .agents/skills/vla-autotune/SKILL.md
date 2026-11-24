---
name: vla-autotune
description: Master loop for stage-aware, agent-driven autotuning of FlashRT VLA inference (pi0, pi0.5, Hy-Embodied-0.5-VLA, FastWAM) on Slurm GPUs. Use when asked to autotune or speed up VLA inference under a quality budget, plan the next optimization step, compare candidate worktrees, or report which steps the human and which the agent decided. Covers same-hardware baselines, profiling and stage attribution, picking a tool per pipeline stage from the FlashRT tool registry (config sweeps, patch or port worktrees, ForgeKernel, KernelAgent, AITune), reference plus bookend campaigns, decision logging, budgets and stop rules. Chinese triggers - autotune, 推理加速, 优化延迟, 按游选工具, worktree 对比, 记录谁主导.
license: Apache-2.0
metadata:
  short-description: "Stage-aware VLA autotune loop for FlashRT"
---

# VLA autotune 主循环

一句话：**先测准，再把时间归到“游”（stage），每一游从工具池挑现成的 autotune 方法；没有现成方法的游，用沉淀下来的 skill 做；每一步都记录是谁做的决定。**

这个 skill 只负责编排。具体动作交给分 lane 的 skill 和 `flash_rt/vla/autotune/` 里的确定性代码：

| 你要做的事 | 用哪个 |
|---|---|
| 测 profile、把延迟归到 stage | `vla-stage-profile` |
| 只改配置（步数、PF、CUDA Graph、池化…） | `vla-config-sweep` |
| 改源码（patch lane，独立 worktree） | `vla-patch-candidate` |
| 把方法/后端移植进 FlashRT 原生路径（port lane） | `vla-port-workflow` |
| 某个算子换 kernel（ForgeKernel / KernelAgent） | `vla-kernel-candidate` → `vla-operator-gate` |
| 编译器/后端自动调优（AITune） | `aitune-inspect` → `aitune-tune` → `aitune-validate` → `aitune-benchmark` |
| 静态执行地图（哪个算子在哪个 stage、跑哪个 kernel） | `vla-model-impl-map` |
| 晋升 winner 前核对 baseline 是否已漂移 | `vla-copy-drift-audit` |

通用技法另有 vendored skill：`perf-torch-cuda-graphs`、`perf-torch-sync-free`、`perf-nsight-systems`、`perf-nsight-compute-analysis`、`perf-workload-profiling`、`kernel-triton-writing`、`kernel-numerics-verifier`、`forge-start`。

## 不要用这个 skill 的情况

- 只问概念或解释报告：直接回答。
- 只写或调试一个 kernel：直接用 `kernel-triton-writing`、`forge-start` 或 `vla-kernel-candidate`。
- 机器人闭环任务评测：本循环的质量门槛是固定输入下的数值误差，不是任务成功率。

## 分工和 human checkpoint

| 谁 | 负责 |
|---|---|
| 确定性代码 | 能力检查（`registry.validate_config`）、候选展开（`tools propose/levers`）、源码快照身份（`snapshots`）、计时和质量门槛（`worker`）、选 winner（`controller ingest`） |
| agent | 读代码和 profile、提出可反驳的假设、选 stage 和工具、写 patch、解释失败 |
| 人 | 定目标和门槛、批准花钱的步骤、纠偏、批准合入 |

遇到下面 4 个点必须停下来问人，并把人的决定记成 `actor=human`：

- **H1 目标**：优化哪个模型、什么硬件、哪个指标（中位数还是 P95）、质量门槛、总 GPU 预算。
- **H2 进入 patch / port / kernel lane，或者用近似方法**（`exactness=approximate`）。
- **H3 启动外部长任务**：ForgeKernel 每个算子要数小时独占 GPU，外加 Claude agent token；KernelAgent 要调用 LLM。启动前确认预算和凭据。
- **H4 合入主线或修改默认配置**。

人改了方向时，记一条 `action=redirect`，并用 `--redirected-from` 写下 agent 原本想做什么。学长要求标出哪些是人引导的，这条记录就是证据。

## 目录约定

集群共享盘上，每个模型一个工作根目录（下面写作 `$AT`）：

```text
$AT/decisions.jsonl        决策日志（append-only，谁决定的）
$AT/snapshots/             源码快照库（patch lane 的 worktree、manifest、patch）
$AT/campaigns/<id>/        controller 根目录（ledger、plan.json、summary.json）
$AT/results/<id>/          worker 输出
```

本机的临时分析放 `devlocal/<feat>/`，它已在 `.gitignore` 里。

## 主循环

每一轮都按下面的顺序走。完整命令见 [references/loop-commands.md](references/loop-commands.md)，按需阅读。

### 0. 定目标（H1）

写下模型、硬件（如 `cuda_sm80`）、指标、质量门槛（如 relative L2 ≤ 0.05 且 cosine ≥ 0.999，以及评价哪些通道）、总预算，记一条 note：

```bash
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind note --actor human --model fastwam --summary "目标：A800 上中位数延迟最低；门槛 relL2<=0.05, cos>=0.999；预算 N GPU 小时"
```

**门槛在跑候选之前就要固定，看到结果后不能再改。**

### 1. 建 baseline（两个）

1. **同一硬件上的官方实现**：先看 `python -m flash_rt.vla.autotune.tools show baseline.official_impl`，五个模型都有作者实现的计时脚本：
   - π0 / π0.5 / FastWAM：`examples/vla_autotune/official_native_latency.py --model <m>`；
   - Hy：`examples/vla_autotune/official_hyvla_latency.py`；
   - DM0.5：`examples/vla_autotune/official_dm05_latency.py`。

   用 `controller init --author-baseline /abs/author.json` 绑定作者参考。`produce` 在分配内运行作者原实现，`saved` 读取绑定过同一输入和权重的结果。读取 `quality` 和 `author_quality` 两组数值，并根据 `summary.author_baseline.latency_comparisons` 查看延迟可比性。配置格式见 `examples/vla_autotune/author_baseline.md`。
2. **当前 FlashRT 配置**：建一个只有 baseline 的 campaign。不传候选时，`controller init` 会只生成 reference + bookend，用来测噪声底线（两次测量的漂移），同时拿到带 profile 的结果。

### 2. 测 profile，归因到 stage

用 `vla-stage-profile`：

```bash
python -m flash_rt.vla.autotune.stages attribute $AT/results/<id>/<attempt>/result.json --rank
```

- 看 `coverage` 和 `notes`。π0/π0.5 在 Euler + CUDA Graph 下只有一个不可拆的 `full` 区间，需要另做诊断才能拆开。
- 用 `amdahl_ceiling(share, local_speedup)` 算每个 stage 的理论上限。
- 证据不够时，先补一次最便宜、又能区分不同假设的观测，不要直接宣布瓶颈在哪。

### 3. 写诊断

```bash
python -m flash_rt.vla.autotune.stages show denoise.velocity     # 看这一游对应的 playbook 症状行
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind diagnosis --actor agent \
  --model fastwam --stage denoise.velocity --summary "..." --symptom "..." --hypothesis "..." \
  --alternative "..." --next-observation "..." --candidate-tool cache.fastwam_text_crossattn_kv \
  --invalid-when "..." --confidence medium --provenance source_fact --evidence-ref <result.json>
```

`provenance` 只能取三个值之一：`source_fact`（读代码得到）、`mechanistic_inference`（机制推断）、`measured_here`（本机实测）。

### 4. 选 stage 和工具

```bash
python -m flash_rt.vla.autotune.tools list --model fastwam --hardware cuda_sm80 --stage denoise.velocity --all
python -m flash_rt.vla.autotune.tools show cache.fastwam_text_crossattn_kv
```

- **优先复用现成工具**。`status=ready` 的 config 工具直接用；`needs_setup` 是现成项目但要先装环境（ForgeKernel、KernelAgent、ApxInf、AITune）；`needs_implementation` 说明这一游还没有现成方法，要写 patch 或 port。
- **按 lane 分流**：
  - 同源、无 patch 的现成 config 旋钮不另开实现 worktree；
  - 非 config 或未知工具、不同源码、任一点带 patch，campaign 必须用源码快照；
  - kernel 先过算子级门槛，再进入模型级 campaign。
- 选择前回答 4 个问题：
  1. 它减少关键路径上的哪一段？
  2. 这一段占多少？Amdahl 上限是多少？
  3. 它会新增什么开销，比如 probe、拷贝、同步、编译、graph 重录？
  4. 是 exact 还是 approximate？

把选择记下来，并注明是谁选的：

```bash
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind decision --actor human+agent \
  --action select_tool --tool launch.cuda_graph --stage host.launch --model fastwam \
  --summary "host.launch 改用 CUDA Graph" --rationale "..."
```

每个 stage 先看哪些工具、需要什么证据，见 [references/stage-tool-routing.md](references/stage-tool-routing.md)。

### 5. 生成候选（按 lane）

| lane | 怎么做 | 产物 |
|---|---|---|
| config | `vla-config-sweep`：`tools propose` | `proposals.json`（同一份代码） |
| analysis 组合旋钮 | `tools levers` / `analysis.lever_pricing`，保留 analysis provenance，按非 config 规则用快照测量 | 同源配置提案 + ready baseline 快照 |
| patch | `vla-patch-candidate`：worktree 里改代码、commit，再 `tools.patches.proposal_from_worktree` | `sha256:` patch，加上 40 位 `source_ref` |
| port | `vla-port-workflow`，最后交给 patch lane | 同上，另附执行 ledger 和 capture 验收 |
| kernel | `vla-kernel-candidate`：capture 调用点，交给 ForgeKernel / KernelAgent，过 `vla-operator-gate`，再把 kernel 接进 patch | `KernelArtifact`，再变成 patch 候选 |
| compiler | `aitune-*`（工具 `compile.aitune`，注册 lane 为 patch） | 外部包装并接入 patch 快照；按 `tools show compile.aitune` 操作 |

一个 campaign 最多 8 个候选；同时在测的源码变体（patch/port）最多 2 个。

### 6. 测量

- **同源、无 patch、注册 lane 为 config 的候选**：普通 campaign（plan v2），`controller init --proposals proposals.json`。旧版没有 `tool` 的纯配置提案保留此兼容路径。
- **源码候选与其他非 config 候选**：
  1. 对 baseline（不带 patch 的 `$BASE`）和每个 patch 都执行 `snapshots materialize`。reference 和复测也要从快照运行，所以 baseline 快照不能漏。
  2. 每个快照都要就绪：纯 Python 改动用 `adopt-native`；改了 `csrc` 就编译，再执行 `record-built`。
  3. `controller init --snapshot-store $AT/snapshots`（plan v3）。batch 和 worker 会在加载模型前核对每个候选实际运行的源码。
- 同一块 GPU 一次只跑一个 campaign。
- 接受新的步数或近似方法之前，先扩大输入集（换噪声、换图像、换 state），确认结论在更多输入上仍然成立。

### 测量前强制清单（B）

- [ ] 用注册表核对每个 `Candidate.tool`。非 config 或未知工具、相对 baseline 不同的 `source_ref`、任一点非 null `patch_ref`，任一成立就必须传 `--snapshot-store`；baseline 只有 reference/bookend 时也检查。
- [ ] `source_ref: null` 表示隐式 checkout，不继承另一点的 commit。不同源码的实现各用独立 worktree；基线和所有候选固定完整 commit / patch hash。
- [ ] baseline、candidate、bookend 的快照全部 ready，原生扩展有身份记录，harness tree 相同；同源候选可共用同一个不可变快照。
- [ ] 确认 plan schema 3、每个 attempt 的 `snapshot_id` 和 ledger manifest 对应；batch/worker 复核实际 import 的源码后才认测量结果。
- [ ] 跑相关 CPU controller/snapshot 回归；拒绝初始化不能留下 plan 或部分 ledger 注册。决策日志写 actor 和证据。

`analysis.lever_pricing` 虽只生成配置组合，但注册 lane 为 analysis，仍满足字面的“非 config”条件，必须走 plan v3；不能删 `tool` 把新提案伪装成旧版配置。

### 7. 决策

读 `$AT/campaigns/<id>/summary.json`：

- `winner` 为空时，看原因：没有点通过质量门槛、reference 或 bookend 没测到、首尾漂移超过 10%。
- **KEEP 不等于 best**：合法但没变快的中间结果可以留作下一步的基础，但它不是 winner。
- **first-valid 不等于 fastest**：KernelAgent 返回的是第一个通过测试的 kernel，不是最快的那个。
- 一个 kernel 局部变快，不算端到端变快。只有模型级 campaign 的结果算数。

每个候选都要记 accept / reject / keep，再写一条 experience，说明结论在什么条件下成立、什么变化会让它失效。`accept` 和 `merge` 只能由人记（actor 为 `human` 或 `human+agent`），决策日志会拒绝 agent 单独记的这两种动作：

```bash
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind decision --actor agent \
  --action reject --candidate-id <id> --summary "reject <id>：没过质量门槛" \
  --rationale "relL2 0.07 > 0.05 on noise case" --campaign <id>
python -m flash_rt.vla.autotune.decisions add --log $AT/decisions.jsonl --kind experience --actor agent \
  --summary "..." --outcome "..." --context-json ctx.json --invalidation "hardware" --invalidation "shape/precision"
```

### 8. 重新定位（两层循环）

- 对新的 baseline（上一轮 winner）重新 profile。
- patch winner 怎么当下一轮的 baseline：patch 不能叠在带 patch 的 baseline 上。
  1. 在 winner 的 worktree 里 commit。
  2. 用 `snapshots resolve --repo <worktree> --ref HEAD` 拿到新的 40 位 commit。
  3. 新写一个 `baseline.json`：`source_ref` 填这个 commit，`patch_ref` 为 null，options 和 winner 一致。
  4. 对它 `materialize` 并使其就绪，下一个 patch 的 worktree 也从这个 commit 开。
  合入主线仍然是单独的 H4 步骤。
- 当前 stage 占比降下来以后，回到第 2 步找下一游，不要在同一个方法上一直加搜索次数。
- 停止规则见 [references/budgets-stop-rules.md](references/budgets-stop-rules.md)。主要几条：
  - 连续 2 轮提升都低于噪声底线；
  - 预算用完；
  - 剩下的 stage 都没有可用工具，且需要人决定是否投入实现。

### 9. 晋升和汇报（H4）

1. patch 类 winner 先跑 `vla-copy-drift-audit`，对照已经前进的 baseline。
2. 由人决定是否合入（`action=merge`，`actor=human`）。
3. 汇报时导出“谁主导”的步骤表：

```bash
python -m flash_rt.vla.autotune.decisions summary --log $AT/decisions.jsonl --markdown
```

决策日志的字段和示例（包括人纠偏的写法）见 [references/decision-log.md](references/decision-log.md)。

## 不能做的事

- 不改质量门槛去迁就结果，也不放宽门槛来掩盖输入或执行上的错误。
- 不跨硬件、跨计时边界比较毫秒数。ApxInf 的 graph 包含 CUDA 预处理，比较前要先对齐边界。
- patch 不能修改 `flash_rt/vla/autotune/`。测量 harness 是固定的，`snapshots` 会拒绝这样的 patch。
- 禁止在 live checkout 轮流改源码、复用可变目录或临时拷贝 `.py`/`.so` 来代替独立源码快照。
- 禁止清除/伪造 `tool`、`source_ref`、`patch_ref`，按 `config.*` 前缀信任未知工具，或把 analysis lane 整体豁免；legacy 兼容不是新源码候选的入口。
- 禁止漏掉 baseline 快照、降级 plan/result schema、伪造 `source_identity` 或用 `PYTHONPATH` 绕过 worker 的 import 身份检查。
- 不在 skill 里记录 winner、最佳 ε 或 kernel 配置。这些数字进决策日志和实验记录（`docs/vla_autotune_experiments.md`）；skill 只写流程。
- 不擅自启动 ForgeKernel 或 KernelAgent。ForgeKernel 会在它自己的沙箱里用 `claude -p --dangerously-skip-permissions` 运行 agent，必须先经过 H3。
- 不 push、不改默认配置，除非人明确要求。
