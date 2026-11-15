# 决策日志：记录谁主导

写决策日志、或导出“谁主导”的步骤表时读这一篇。

日志是 append-only 的 JSONL 文件（`flash_rt/vla/autotune/decisions.py`），每条记录都必须带 `actor`：

| actor | 什么时候用 |
|---|---|
| `human` | 人提出或拍板的，包括人纠偏 |
| `agent` | agent 自己提出并执行的 |
| `human+agent` | agent 提出、人确认，或者双方讨论后定的 |
| `search` | 确定性代码自动生成的，比如 `tools propose` 展开的网格 |

## 四种记录

每条记录都必须带 `actor` 和非空的 `summary`。

- **`diagnosis`**：`symptom`、`hypothesis`、`alternatives`、`next_observation`、`candidate_tools`、`invalid_when`、`confidence`（low / medium / high）、`provenance`（source_fact / mechanistic_inference / measured_here）。
- **`decision`**：`action` 取 `select_stage`、`select_tool`、`run_campaign`、`keep`、`accept`、`reject`、`redirect`、`stop`、`merge` 之一，外加 `rationale`。
  - `select_tool` 要带 `--tool`；`select_stage` 要带 `--stage`。
  - `redirect` 要带 `--redirected-from`，而且 actor 只能是 `human` 或 `human+agent`。
- **`experience`**：`context`（workload 签名的 JSON）、`outcome`、`invalidation`。`invalidation` 列出哪些变化会让这条结论失效，默认是：模型/权重、源码/运行时版本、硬件、shape/精度、输入分布、计时边界、质量协议。
- **`note`**：自由文本，比如目标或阶段小结。

## 示例：人纠偏

agent 想继续调 GEMM，人判断 GEMM 不在关键路径上，改去做 CUDA Graph：

```bash
python -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human --action redirect \
  --model fastwam --stage host.launch --tool launch.cuda_graph --summary "人纠偏：GEMM 改为 CUDA Graph" \
  --redirected-from "agent planned another GEMM plan-cache iteration (kernel.gemm)" \
  --rationale "illustrative: instrumented kernel sum is far below the uninstrumented request median; launch gaps hypothesised" \
  --evidence-ref $AT/results/r3/r3-000-reference/result.json
```

## 示例：agent 提出、人确认

```bash
python -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor human+agent --action select_tool \
  --tool sampler.euler_steps --stage denoise.schedule --model pi05 --summary "选 Euler 步数扫描" \
  --rationale "agent proposed; human approved widening to 4 inputs before accepting"
```

## 汇报

```bash
python -m flash_rt.vla.autotune.decisions list --log $LOG --markdown       # 按时间顺序的步骤表，带 actor 列
python -m flash_rt.vla.autotune.decisions summary --log $LOG --markdown    # actor × kind 计数
```

这张步骤表可以直接贴进报告，回答“哪些步骤是人引导的、哪些是 agent 自己做的”。只有当时就记下来的才算数；不要在事后凭印象补记成 `agent` 或 `human`。确实是事后补的记录，要在 summary 里写明“事后补记”。
