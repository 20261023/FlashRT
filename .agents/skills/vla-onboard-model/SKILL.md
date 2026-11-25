---
name: vla-onboard-model
description: Use when adding a new VLA model to FlashRT, declaring its inputs or action head, or qualifying its original-author baseline, graph execution and repeated hardware evidence. Also use when a model is only declared, fixtures are missing, or onboarding remains unverified. Chinese triggers - 新模型接入, onboarding, 原作者入口, 接入验收.
license: Apache-2.0
metadata:
  short-description: "VLA model onboarding and evidence qualification"
---

# VLA 新模型接入

用公共 ModelSpec 描述模型，通过 `flash_rt.vla.onboard` 建立和检查接入材料。模型计算和实验由现有 backend/author/campaign 机制完成；成功验收必须有原作者参考和内部重复测量。

完整字段与命令见 [接入指南](../../../docs/vla_onboarding.zh-CN.md)。这是入口参考，`--help` 是 CLI 参数的权威来源。

## 适用范围

- 新模型、checkpoint 变体或新的输入/动作契约需要接入。
- 声明已存在，但原作者入口、fixture、profile 或实测证据尚不完整。
- 为一个明确的硬件和数值预算结束 `phase=onboarding` 工作段。

已有合格模型的提速迭代使用 `vla-autotune`。单个新 kernel 使用 `vla-kernel-candidate` 和 `vla-operator-gate`，随后走模型级 campaign。

## 最小路径

1. 用 `token_usage start` 记录当前 session/model 的 onboarding 段。
2. `onboard init --model MODEL --head flow_matching --root DIR` 生成模板。
   token 模型使用 `--head autoregressive_tokens`。可选回归模板使用
   `--head other`，公共声明为 `type=other, kind=regression`。
3. 查原作者仓库、固定修订、实际权重配置和许可证。填公共 ModelSpec
   的 identity、inputs、head、actions、stages/profiling、capabilities、
   adapter 和 author。无法确认的事实保留带来源的 `unknown`。
4. 替换所有模板假设。用实际 shape/dtype、prompt、state、图像和固定噪声
   建立至少四组 fixture。合成 fixture 仅支持固定输入数值检查。
5. 实现生成的项目 adapter/backend、独立原作者 module 与测试骨架：
   `spec["author"]["adapter"]` 的 `load(spec, torch)` 返回
   `(author_model, infer(request), metadata)`；metadata 包含
   `implementation` 和 `expected_shape`。统一 producer 继续负责来源、
   权重、输入、输出域、实际 argv 和硬件控制。
   骨架的未实现标记/抛错分支未清理时，静态检查不能成为 GPU-ready。
6. 测量前确认数值预算与输入/硬件假设。运行 `onboard check --root DIR`，
   修复所有 issue，得到 `ready_for_measurement`。
7. 在实际 Slurm GPU 作业中运行原作者参考、项目 reference、
   graph/eager 切换和 bookend。每个必要路径每个 case 至少两次样本。
8. 部署完整的固定 controller 源码，核对 plan 中作者入口和 venv 解释器的实际路径。controller 完成 ingest；完整保存 ledger、plan、portable `control/`、
   result.json 和 outputs.npz，再运行 `onboard verify`。

## 实测证据

| 条件 | 验收依据 |
|---|---|
| 原作者来源 | 固定官方仓库/commit，原作者类真实文件和字节，权重内容 sha256 |
| 数值质量 | 与确认的通道、域和预算一致的 internal 与 author 两套 gate，重新从 NPZ 计算 |
| graph/eager | 同一 campaign、同一源码/配置，仅切换 use_cuda_graph；比较原始输出 |
| 实际 graph | 各样本 execution_metadata 的 use_cuda_graph、graph_replays、graph_capture_id |
| token 解码 | sampler=autoregressive，无 flow steps；实际 prefill+decoder forward 次数之和等于 NFE，并保留 generated_tokens |
| 重复执行 | 同一实际 GPU/Slurm job；每 case 至少两次，reference/bookend 漂移不超过 10% |
| profile | 已归属且可加总的非重叠区间覆盖 wall 至少 90%，或绑定原始证据的重复量化残差解释 |

低覆盖率的实测解释须绑定 result SHA、instance/attempt、job、GPU 和
workload，复现 wall/covered/residual 数值，并有至少两次观测及哈希匹配
的非空支持文件。保留原始覆盖率和限制；纯文字解释不构成通过。

token 模板使用显式 inert zero noise，不运行去噪；完成 checkpoint 对应
的 action-token 解码后再比较连续动作。它不会获得 Euler/ProbeFlow 工具。
非连续公开通道使用共享 action_channels 数值 gate，不能改成前缀切片。

## 执行与收尾

模型计算只在已分配的 Slurm GPU compute 作业执行。按当前站点约定，
用 sbatch 脚本直接运行 Python，CPU/内存由队列分配，等待同一 job
自然终态。CPU scaffold/check/verify 可以在登录节点运行。

源码候选遵守仓库的强制 snapshot checklist：独立 worktree、40 位
source_ref、内容寻址 patch、ready baseline/candidate snapshots、同一
harness、plan v3 和实际导入身份。未知或非 config 工具也要求 snapshot；
不能删改字段获得 v2。

```bash
python -m flash_rt.vla.onboard verify \
  --root "$ONBOARD" --campaign "$CAMPAIGN" --results "$RESULTS" \
  --graph-result "$GRAPH_RESULT" --eager-result "$EAGER_RESULT" \
  --decisions "$DECISIONS" --segments docs/vla_autotune_token_segments.jsonl \
  --session-id "$OPENCODE_SESSION_ID" --actor agent
```

成功路径追加 decisions note，通过 H 的 SegmentLog 结束当前
session/model 的 onboarding 段，最后发布 verification.json。
缺证据继续补齐。存在 summary、配置打开 graph、作者路径的文字说明或
单次成功输出都不能替代这个验收。

若收尾 I/O 中断，保留已经写入的 note/segment 和原始 bundle，核对
receipt 与各日志再恢复。成功 receipt 才表示整个接入收尾完成。
验收不授权提交、推送、合入、改默认配置或发布，也不推断闭环任务成功率。
