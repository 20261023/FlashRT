# 一轮完整命令

在需要真正敲命令时读这一篇。下面的命令里，尖括号外的占位符都写成 `$NAME`，按实际情况替换。

所有模块都是 `python -m flash_rt.vla.autotune.<module>`。在本机（没有 torch/CUDA）可以这样运行：

```bash
UV_CACHE_DIR=/tmp/autotune-uv PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy python -B -m flash_rt.vla.autotune.tools list
```

模型计算只能放在 Slurm 分配到的计算节点上跑，不能在登录节点上跑。

## 0. 变量

```bash
export REPO=/shared/FlashRT              # 集群上的 FlashRT 检出（harness 从这里运行）
export AT=/shared/autotune/fastwam       # 本模型的工作根目录
export LOG=$AT/decisions.jsonl
mkdir -p $AT/campaigns $AT/results $AT/snapshots
```

## 1. baseline 候选和 workload

`baseline.json` 是一个 `Candidate`。字段与 `flash_rt/vla/autotune/schema.py` 一致：

```json
{"candidate_id": "fastwam-graph-e20",
 "backend": {"model": "fastwam", "checkpoint": "/shared/ckpt/fastwam", "backend": "torch",
             "sampler": "euler", "hardware": "cuda_sm80", "framework": "torch", "precision": "auto",
             "num_steps": 20, "probeflow": null,
             "options": {"sigma_shift": 5.0, "action_horizon": 32, "model_config": "/shared/ckpt/fastwam/config.json",
                         "backbone_dir": "/shared/ckpt/wan", "use_cuda_graph": true}},
 "workload_id": "fastwam-12req", "parent_id": null, "hypothesis": "current FlashRT config",
 "origin": "human", "patch_ref": null, "source_ref": null}
```

- 用 snapshot 模式（patch lane）时，`source_ref` 必须是 40 位 commit，可以用 `snapshots resolve` 解析得到。
- workload 的格式见 `docs/vla_autotune.md` 的 “Explicit workload” 一节。输入要覆盖原始输入、换噪声、换图像、换 state 等情况；一次只测一个输入的结论不可信。

## 2. 只有 baseline 的 campaign（reference + bookend）

```bash
python -m flash_rt.vla.autotune.controller init --root $AT/campaigns/r0 --comparison-id r0 \
  --baseline baseline.json --workload /shared/workloads/fastwam-12req.json --output-root $AT/results/r0 \
  --warmup 3 --repeats 10 \
  --quality-gate '{"scope":"fastwam-normalized7-v1","action_dim":7,"max_relative_l2":0.05,"min_cosine":0.999}'
```

不传 `--proposals` 和 `--enumerate` 时，campaign 只包含 reference 和 bookend。两次测量的漂移就是噪声底线。

## 3. 生成 sbatch、提交、等待、导入

controller 不会自己生成 sbatch 脚本，用 `slurm.render_sbatch` 渲染：

```bash
python - <<'EOF'
from flash_rt.vla.autotune.slurm import render_sbatch, SlurmResources
import os
AT = os.environ['AT']; REPO = os.environ['REPO']
print(render_sbatch(
    job_name='at-r0', workdir=REPO, output=f'{AT}/campaigns/r0/slurm-%j.log',
    argv=['python', '-B', '-m', 'flash_rt.vla.autotune.batch', '--plan', f'{AT}/campaigns/r0/plan.json'],
    resources=SlurmResources(partition='gpu_a800', gpus=1, cpus=8, time_limit='00:40:00'),
    env={'PYTHONPATH': REPO}), end='')
EOF
```

把输出保存为 `$AT/campaigns/r0/run.sbatch`，然后：

```bash
python -m flash_rt.vla.autotune.controller submit --root $AT/campaigns/r0 --script $AT/campaigns/r0/run.sbatch --ssh-host $GPU_LOGIN
python -m flash_rt.vla.autotune.controller wait   --root $AT/campaigns/r0 --ssh-host $GPU_LOGIN --interval 30 --timeout 7200
python -m flash_rt.vla.autotune.controller ingest --root $AT/campaigns/r0 --results $AT/results/r0 --job-finished
```

结果在 `$AT/campaigns/r0/summary.json`；每个 attempt 的原始结果在 `$AT/results/r0/<attempt>/result.json`。

## 4. 归因到 stage

```bash
python -m flash_rt.vla.autotune.stages attribute $AT/results/r0/r0-000-reference/result.json --rank
python -m flash_rt.vla.autotune.stages list --model fastwam      # 这个模型的 stage 地图和代码位置
python -m flash_rt.vla.autotune.stages show host.launch          # playbook 症状行和默认工具
```

## 5a. config lane

```bash
python -m flash_rt.vla.autotune.tools list --model fastwam --hardware cuda_sm80 --stage denoise.schedule
python -m flash_rt.vla.autotune.tools propose --baseline baseline.json --tool sampler.euler_steps --param 'steps=[12,14,16]' --out p-steps.json
python -m flash_rt.vla.autotune.tools levers  --baseline baseline.json --lever 'graph={"options":{"use_cuda_graph":true}}' \
  --lever 'e14={"num_steps":14}' --out p-levers.json
python -m flash_rt.vla.autotune.controller init --root $AT/campaigns/r1 --comparison-id r1 --baseline baseline.json \
  --workload /shared/workloads/fastwam-12req.json --output-root $AT/results/r1 --proposals p-steps.json --quality-gate '...'
```

然后回到第 3 节，渲染 sbatch、提交、等待、导入。

## 5b. patch lane（独立 worktree + 源码快照）

```bash
export BASE=$(python -m flash_rt.vla.autotune.snapshots resolve --repo $REPO --ref HEAD | python -c 'import json,sys;print(json.load(sys.stdin)["commit"])')
git -C $REPO worktree add --detach $AT/wt-crossattn $BASE        # 在这里改代码，并 commit
python - <<'EOF'
from flash_rt.vla.autotune.tools.patches import proposal_from_worktree
import os
p = proposal_from_worktree(os.environ['AT'] + '/wt-crossattn', os.environ['BASE'], store=os.environ['AT'] + '/snapshots',
        tool_id='cache.fastwam_text_crossattn_kv', stage_id='denoise.velocity',
        hypothesis='cache text_embedding + per-layer cross-attn K/V once per request', options={})
p.save(os.environ['AT'] + '/proposal-crossattn.json'); print(p.patch_ref)
EOF
# baseline 快照（不带 patch）和 patch 快照都要建：reference 和复测也从快照运行
python -m flash_rt.vla.autotune.snapshots materialize --repo $REPO --store $AT/snapshots --base $BASE
python -m flash_rt.vla.autotune.snapshots materialize --repo $REPO --store $AT/snapshots --base $BASE --patch-ref sha256:...
# 两个快照都要就绪（输出里的 snapshot_id 分别填进来）
python -m flash_rt.vla.autotune.snapshots adopt-native --store $AT/snapshots --snapshot-id <baseline-id> --prebuilt-root $REPO --fingerprint <build-id>
python -m flash_rt.vla.autotune.snapshots adopt-native --store $AT/snapshots --snapshot-id <candidate-id> --prebuilt-root $REPO --fingerprint <build-id>
#   改了 csrc 时不能复用 .so：先 build-script --cutlass-dir ...，作为 build job 运行，再执行 record-built
#   更细的步骤见 vla-patch-candidate 第 4–5 步
```

接下来：

1. 用 `tools.patches.candidate_from_proposal(baseline, proposal)` 生成候选，写进 proposals 文件。baseline 本身的 `source_ref` 也要是 `$BASE`。
2. `controller init ... --snapshot-store $AT/snapshots`，加上需要时的 `--python /abs/python` 和 `--extra-pythonpath /abs/path`。
3. 回到第 3 节提交。batch 会先重新核对每个快照，然后每个 attempt 在自己的快照目录里运行。

## 5c. kernel lane

见 `vla-kernel-candidate`，概要如下：

1. `tools.capture.capture_calls` 在 GPU 上抓取调用点的形状和张量。
2. `tools.external.forgekernel.prepare_task` 或 `tools.external.kernelagent.prepare_problem` 生成任务。
3. `render_job` 生成作业脚本，作为独立的 allocation 运行（需经过 H3）。
4. `harvest` 收集结果，得到 `KernelArtifact`。
5. 用 `vla-operator-gate` 做算子级验收。
6. 进入 patch lane。

## 6. 记录

```bash
python -m flash_rt.vla.autotune.decisions add --log $LOG --kind decision --actor agent --action run_campaign --campaign r1 \
  --tool sampler.euler_steps --stage denoise.schedule --model fastwam --summary "Euler 步数扫描 r1" \
  --rationale "steps 12/14/16 vs E20 on 12 inputs"
python -m flash_rt.vla.autotune.decisions list --log $LOG --markdown
python -m flash_rt.vla.autotune.decisions summary --log $LOG --markdown
```
