# VLA 新模型接入与验收

`python -m flash_rt.vla.onboard` 提供 CPU 模板生成、静态检查和已完成实验的证据验收。模型声明统一使用 `flash_rt.vla.model_spec`；此入口不维护第二套模型注册表，不加载模型、不下载权重、不提交 GPU 作业。

模型源码仍遵循 [adding_new_model.md](adding_new_model.md) 的 frontend/pipeline 硬件分文件约定。通过 onboarding 表示指定输入、权重、执行路径和数值预算已经验证；任务成功率和优化加速比需要各自的实验。

## 1. 建立工作目录

用 H 的接口记录接入工作段，随后生成模板：

```bash
python -m flash_rt.vla.autotune.token_usage start \
  --model myvla --phase onboarding --item D --summary "接入 myvla"
python -m flash_rt.vla.onboard init \
  --model myvla --head flow_matching --root devlocal/myvla-onboarding
```

CLI 的两类主要模板及可选回归模板：

| `--head` | 公共 ModelSpec | 默认执行语义 |
|---|---|---|
| `flow_matching` | `head.type="flow_matching"` | 固定噪声，Euler 网络调用预算 |
| `autoregressive_tokens` | `head.type="autoregressive_tokens"` | 自回归 action-token 解码，`sampler="autoregressive"` |
| `other` | `head.type="other", head.kind="regression"` | 单次直接回归，`sampler="none"` |

公共 schema 还认识 `diffusion`；当前 CLI 没有该类的执行模板，不能直接套用 flow matching 模板作出验收结论。

新目录包含：

```text
onboarding.json                 接入约束、显式确认和文件引用
specs/myvla.json                 公共 ModelSpec 模板
fixtures/workload.json           普通 NPZWorkload manifest
fixtures/case-000.npz ...        四组可复现、不同输入的 CPU 数据
myvla_onboarding/
  project_adapter.py            项目 binding/validate 骨架
  project_backend.py            项目 frontend、infer/close 骨架
  author_adapter.py             独立原作者入口骨架
tests/test_myvla_contract.py     可运行的 CPU scaffold 合同测试骨架
author_baseline.json             produce/saved 作者参考配置骨架
```

`init` 拒绝覆盖已有目录。模板中的两台 RGB 相机、224²、state8、horizon50、model_dim32、effective_dim7、BF16/SM80 和数值预算均是**待确认假设**。它们不是对新模型的事实判断。先阅读权重配置、官方源码和许可证，替换这些值及其来源，然后再生成实际 fixture。

## 2. 填写公共声明和 fixture

只通过公共接口读取声明：

```python
from flash_rt.vla.model_spec import get_spec

spec = get_spec("myvla", spec_dir="/absolute/onboarding/specs")
```

`identity.source` 必须固定官方仓库 URL 和完整 40 位 commit；`identity.weights` 固定权重修订及来源，`identity.license` 记录源码和权重许可证。还应将统一的内容标识写入 `identity.weights.sha256`：

```python
from flash_rt.vla.autotune.author import content_identity

digest = content_identity("/local/checkpoint")["sha256"]
```

这是 checkpoint 目录中实际文件的内容标识；它不是简单对 revision 字符串求哈希。验收会与原作者 producer 在运行时保留的 checkpoint 标识核对。

`inputs` 声明实际 shape/dtype。模板 recipe `onboarding_numpy_v1` 的附加数据为：

| 字段 | 含义 |
|---|---|
| `fixture.fields` | observation 键到 input 名或有序 input 名列表 |
| `fixture.noise_key` | 固定噪声的 input 名 |
| `fixture.dimensions` | 将符号维度解析为正整数 |
| `fixture.seed` / `cases` | NumPy 随机种子和至少四组输入 |
| `fixture.prompts` | 非空提示词列表，按顺序循环 |

修改声明后生成一个新的 fixture 目录，不覆盖旧样本：

```python
from flash_rt.vla.model_spec import get_spec
from flash_rt.vla.onboard import generate_fixtures

directory = "/absolute/onboarding/specs"
spec = get_spec("myvla", spec_dir=directory)
generate_fixtures(spec, "/absolute/onboarding/fixtures-v2", spec_dir=directory)
```

将 `onboarding.json.workload` 指向新 manifest。也可以使用真实数据集生成的普通 NPZWorkload；保留采样来源、预处理、顺序、prompt 和固定噪声，并确保 `fixture.fields`、shape、dtype 与模型声明一致。移动目录后同步 workload 的 `spec_dir`；外部声明的发现作用域始终显式传递。

生成器只做 CPU NumPy 运算。合成样本不能证明数据集覆盖或闭环任务质量。回归和 token 头保留 `[1]` 全零 noise 字段以满足文件 envelope，声明 `fixture.noise_role="unused"`；实现忽略它，不产生 denoising/NFE 优化假设。

`actions` 分别声明内部宽度、有效通道、horizon、公开动作域和 worker 比较域。连续前缀通道保留原 `action_dim` gate；非前缀通道使用共同的 `action_dim` 加 `action_channels` 显式索引，重新从原始数组选取声明的通道进行比较。

把实际输入、精度、硬件、执行范围等限制写入 `onboarding.json.assumptions`。在测量前固定 `quality.thresholds`，明确确认 `quality_budget_confirmed` 和 `template_assumptions_confirmed`。这些确认用于保存已经确定的约束；看完候选结果以后不能改预算来制造通过。

## 3. 实现项目入口与原作者入口

`adapter.hooks.binding` 指向运行时适配器。它按公共约定 `binding(config, hardware)` 返回 `(module, class_name, kwargs)`；模型计算仍放在各硬件的 frontend/pipeline。生成的 `myvla_onboarding.project_adapter` 已提供可导入的 binding/validate 签名，`project_backend` 提供 frontend 的 infer/close 签名，但默认拒绝计算。缺少该入口时，声明发现仍可用，静态接入检查会报告尚未就绪。

三个骨架默认 `ONBOARDING_IMPLEMENTED=False`，未实现分支明确抛出 `NotImplementedError`。填写实现并替换原作者/项目入口后，更新 `onboarding.json.implementation_files` 指向实际三个源文件。静态检查以 AST 检查文件和标记/未实现分支，不通过 import 执行代码；残留骨架不能进入 GPU-ready 状态。生成的 CPU 测试骨架验证默认拒绝与 fixture 完整性；实现时将其替换/扩展为实际绑定、数值与资源生命周期测试。

独立原作者入口由 `spec["author"]["adapter"]` 指定一个 Python module，其协议是：

```python
def load(spec, torch):
    # 从固定的、干净的原作者源码和 checkpoint 构造模型。
    # infer(request) 使用 request.observation/prompt/noise。
    return author_model, infer, {
        "implementation": "描述实际原作者 API",
        "expected_shape": [actual_horizon, actual_worker_action_dim],
    }
```

统一 author producer 负责检查原作者类的真实文件、源码字节、权重、fixture、输出域、输出样本、实际 argv 和 Slurm/GPU 身份。适配器可以提供公共 author 机制支持的可选预检 hook；不要在适配器中绕开这些控制。`myvla_onboarding/author_adapter.py` 是待实现骨架，需放到明确的可导入 module 路径并更新 ModelSpec。

项目 Torch 或 FlashRT graph 不能标为“官方 Native”。该称呼仅用于作者原执行栈。saved 作者结果可以作为已绑定的数值参考；跨作业或不同硬件的 saved 延迟不自动获得可比加速比。

每次 `infer` 还需返回实际的逐请求 graph 计数：

```python
result["execution_metadata"] = {
    "use_cuda_graph": True,          # eager 为 False
    "graph_replays": actual_replays, # eager 为 0；记录实际行为
    "graph_capture_id": capture_id,  # graph 的已捕获资源标识；eager 可为 None
}
```

worker 会保留各次样本的完整对象。只设置 `options.use_cuda_graph=True` 而没有实际 replay 证据不能通过验收。

### 自回归 token 头的计数

使用 `init --head autoregressive_tokens`，核对并替换 `head.vocab_size`、
`head.max_tokens`，实现 checkpoint 对应的 tokenizer 和 action-token decoder。
输出仍是指定动作域中的连续动作数组，不能把 token ID 当作连续动作比较。
项目 BackendConfig 使用 `sampler="autoregressive"`，不传 `num_steps` 或 ProbeFlow。
生成的外部声明不会获得 Euler/ProbeFlow 工具路由。

逐请求记录实际观察到的 neural forward 调用，而不是把 max_tokens、
author envelope 的 num_steps 或配置中的预算写成 NFE：

```python
result["sampler_metadata"] = {
    "sampler": "autoregressive", "nfe": prefill_calls + decode_calls,
    "generated_tokens": token_count,
}
result["execution_metadata"].update(
    prefill_forward_calls=prefill_calls, decoder_forward_calls=decode_calls,
    generated_tokens=token_count, nfe_provenance="observed_forward_calls",
)
```

prefill 和 decoder 是实际神经网络 forward 调用次数，允许一个 forward
产生多个 token。验收核对各次计数之和、实际输出 token 数和最大 token
约束；它不从所请求的预算推断执行次数。原作者 producer 也必须保留实际
decoder 调用证据，不能对 token 头沿用 `nfe == Euler num_steps` 假设。

## 4. 静态检查

```bash
python -m flash_rt.vla.onboard check --root devlocal/myvla-onboarding
```

退出码 0、`status="ready_for_measurement"` 表示声明、fixture、原作者配置和显式确认齐备。check 中模板未完成、解析失败或缺证据返回退出码 1 和 `issues`；init/verify 的调用或证据错误返回 2。静态检查不导入运行时或原作者适配器，不声称模型已经在 GPU 上运行。

## 5. 运行有权威记录的 campaign

先读 `vla-autotune` 与适用 lane 的 skill，在独立 worktree 实现源码，再建立同一硬件、同一输入和相同数值预算的 campaign。对接入的最小证据组是：

1. 项目 reference；
2. 同一模型、源码、权重、precision、sampler/NFE 的 graph/eager 切换候选；
3. 重复相同 reference 的 bookend；
4. 必须绑定 `--author-baseline` 的原作者 producer 或已验证 saved bundle。

graph 和 eager 结果必须属于**同一个**已 ingest 的 campaign。每条必要路径、reference 和 bookend 均需覆盖每个 fixture 至少两次：四个 case 时 `repeats` 至少为 8。warmup 不算重复证据。保留原始 `outputs.npz`、每次 latency、实际 sampler/NFE、runtime、GPU UUID/型号/capability、Slurm job ID 和 compute host。

源码候选遵循 [AGENTS.md](../AGENTS.md) 的强制 checklist：非 config 或未知工具、不同 source_ref、任何非空 patch_ref 都要求 ready baseline/candidate snapshots、`--snapshot-store` 和 plan v3；不能删改 provenance 字段来获得 plan v2。相同源码下已注册 config 切换可以使用 plan v2。

按当前站点约定使用 `sbatch`，脚本内部直接运行 Python batch 入口；CPU 和内存由队列分配，不加固定 CPU、绑核或自行规定的模型/等待截止。模型运行与 profile 仅在实际分配的 Slurm GPU compute 作业中进行，等待同一 job ID 自然终态。onboard CLI 本身不会提交或执行这些计算。

在 controller 中完成 ingest，再把完整 `control/`、各 attempt 的 JSON/NPZ 和原始 campaign ledger 保存到验收能读取的位置。`summary.json` 是便于阅读的派生报告，不能代替 ledger 或原始 bundle。

## 6. Profile 覆盖与实测解释

graph 路径的诊断 profile 必须提供同一请求的正 wall time、原始 interval 样本、样本和、ModelSpec 中的 stage 归属，以及 `interval_provenance.additive=True`。加总的非重叠区间覆盖 wall time 至少 90% 即通过覆盖要求。区间之和超过 wall time 会被拒绝，不能用嵌套计时重复加总来提高覆盖率。重新捕获的诊断 graph 还需保留 `capture_parity.status="bit_exact"`。

覆盖不足时，`--residual-explanation /path/explanation.json` 必须提供实测解释：

```json
{
  "schema_version": 1,
  "kind": "onboarding_residual_explanation",
  "binding": {
    "campaign_instance_id": "实际 instance",
    "attempt_id": "实际 graph attempt",
    "job_id": "实际 job",
    "gpu_uuid": "实际 UUID",
    "workload_digest": "实际 workload sha256",
    "result_sha256": "graph result.json 文件 sha256"
  },
  "wall_ms": 20.0,
  "covered_ms": 9.5,
  "unattributed_ms": 10.5,
  "reason": "说明实测残差来自哪一段",
  "limitations": "诊断边界和适用范围",
  "observations": [{
    "stage": "host.launch",
    "clock": "host_wall",
    "samples_ms": [10.4, 10.6],
    "artifact": {"path": "retained-raw-evidence.json", "sha256": "该原始证据文件 sha256"}
  }]
}
```

数值必须来自实际观测；这里的毫秒数只是格式示例。每项至少两次有限非负观测，clock 为 `host_wall`、`cuda_event` 或 `nsys`，并引用不同、非空、哈希匹配的原始支持文件。各项观测中位数之和需在 wall time 的 10% 误差内解释残差。所有绑定字段和残差数值均重新核对。纯文字理由不能通过；验收保留原始低覆盖率，不将解释改写为 90% 覆盖。

## 7. 验收与结束接入段

```bash
python -m flash_rt.vla.onboard verify \
  --root devlocal/myvla-onboarding \
  --campaign /runs/myvla/campaign \
  --results /runs/myvla/results \
  --graph-result /runs/myvla/results/round-000-reference/result.json \
  --eager-result /runs/myvla/results/round-001-candidate/result.json \
  --decisions /runs/myvla/decisions.jsonl \
  --segments docs/vla_autotune_token_segments.jsonl \
  --session-id "$OPENCODE_SESSION_ID" --actor agent
```

验证会检查公共声明、当前 fixture 身份、完整 campaign/ledger/portable 作者证据和源码身份，重新计算 internal 与 author gate，再比较 graph/eager 原始动作。实际 GPU/作业必须一致，首尾中位数漂移不得超过 10%，每个必要路径都要有重复样本和实际 graph/eager 执行证据。

成功时追加 `kind=note` 的决策记录，通过 H 的 `SegmentLog` 关闭同一 session/model 的 `phase=onboarding` 段，最后发布 `verification.json`。缺失作者、输出、真实设备字段、数值预算或足够 profile 证据都会失败，不结束工作段、不生成成功 receipt。已有 receipt 不覆盖，新的接入验证使用新目录。

决策日志、段日志和 receipt 是三个 append/发布步骤。进程或文件系统若在收尾途中失败，可能留下“证据已核对、完成待 receipt”的 note 或已关闭段；只有成功 receipt 表示整个收尾已完成。保留部分记录以便核对恢复，不手动删改旧日志来隐藏中断。

## CPU 回归

```bash
python -m pytest -q tests/test_vla_onboard.py tests/test_vla_agent_skills.py
```

测试用临时目录中的合成原作者/worker bundle 和虚构设备元数据验证证据检查逻辑，不运行模型，也不构成任何模型在真实 GPU 上通过 onboarding 的证据。
