# VLA autotune 清理映射

A 阶段由 Codex 执行。审计起点是
`199376281801427abccbb307056b27e74eab1429`；仓库不是 shallow clone。
以下“删除”只指当前工作树入口，旧源码、测试及其 Git 历史仍可按提交取回。
历史报告、token 原始计数、动作结果及快照身份不在本阶段裁剪。

以上是 **A 阶段的审计边界**。J 随后只整合并撤下重复的报告正文，
没有删除机器数据、token 快照或物理实验结果；见本文末尾“J 报告整合”。

## 删除前同时核对的三方面

1. **用途与当前路径**：资源已经退出本轮目标，或已有现行实现接替；现行推理、
   作者基线、质量契约及当前 best 不依赖它。
2. **引用闭合**：搜索完整路径、文件名、动态 sibling 加载及 Python 测试导入；
   专用测试和入口说明一起处理，保留测试不能依赖已删除的辅助模块。
3. **来源与证据**：核对每个文件的新增提交、最后修改及历史报告；保留来源 pin、
   原始许可证/NOTICE、当前 best 和尚待汇总的证据。来源或用途不清就保留。

引用检查覆盖 tracked 源码、文档和隐藏技能目录。Hy 旧诊断的调用均在该诊断集合、
其专用测试或历史说明内；作者计时入口 `official_hyvla_latency.py` 独立使用
`validate_hyvla.py`，没有加载旧观察器。专用测试之外未发现对被删测试模块的导入。
保留的 GPU 功能测试仍从现行库及其现行测试辅助函数导入。
保留的作者输入转换功能测试将函数内裸 `import torch` 改为
`pytest.importorskip('torch')`：无 Torch 的 CPU 环境仅跳过该测试，其他证据标签、
计时边界和来源测试继续执行；安装 Torch 时原有全部断言仍执行。
同一模块的错误框架负例接受现行注册器明确的 `frontend/framework` 拒绝信息，
仍要求抛出 `ValueError`，没有删除该负例。

清理前报告记载的 Hy 配置是 semantic BF16、Euler8、Graph、SM80 RoPE/QK/KV
融合，见[整合报告中的配置扫描和融合结果](reports/2026-11-24-vla-autotune-final.zh-CN.md#hy-history)。FastWAM 报告记载
Euler20、compiled prefill/denoise 和请求内文本 K/V 缓存，见
[同轮缓存比较](reports/2026-11-24-vla-autotune-final.zh-CN.md#fastwam-history)。
这些实现、选项和历史证据均保留；旧报告正文已由 J 整合，最终 best 以本轮后续决策为准。

## 删除的 Hy 旧诊断：逐文件来源和测试

下表脚本路径统一位于清理前的 `examples/vla_autotune/hy_trace/`；
测试路径统一位于清理前的 `tests/`。表中的 A 是 `git log --diff-filter=A -1`
的新增提交，C 是清理前最后修改提交；均为 Git 中可唯一解析的提交前缀。
这些文件是 FlashRT 研究代码；观察官方符号的部分按 semantic-review-only
核对，未把它们误认成上游 vendored 模型实现。

所有行都退出当前目标：Hy 已改为 public20 的延迟—误差预算，旧 EXACT、
固定 Gold 轨迹、逐层 RMS/attention/LSE 闭环不再是速度搜索前提。
历史负结果仍完整留在
[完整历史逐轮记录](reports/2026-11-24-vla-autotune-final.zh-CN.md#historical-record)和
[研究过程整合](reports/2026-11-24-vla-autotune-final.zh-CN.md#research-history)。

| 删除脚本 | 旧用途 | A / C | 同时删除的专用测试 |
|---|---|---|---|
| `diagnose.py` | old/native 边界、固定 KV、轨迹诊断 | `73b381bc` / `a0ff10db` | `test_hyvla_boundary_example.py` |
| `reference_observer.py` | 官方 revision 绑定的观察 hooks | `73b381bc` / 同 A | 随 boundary 测试 |
| `eager_attention_capture.py` | 实际 eager Q/K/V/context 观察 | `78208988` / 同 A | `test_hyvla_eager_attention_capture_example.py` |
| `native_controls.py` | 固定原生输入及诊断清理控制 | `2959bd87` / `50b86d7f` | `test_hyvla_native_controls_example.py` |
| `frozen_ops.py` | patch/position/PROJ1 冻结控制 | `d91ef23b` / 同 A | `test_hyvla_frozen_ops_example.py` |
| `frozen_attention.py` | block0 spatial attention 后端控制 | `ddbb5faa` / `496d784a` | `test_hyvla_frozen_attention_example.py` |
| `proj1_camera_ablation.py` | HWD/per-camera PROJ1 临时覆盖 | `aa3da4b3` / `2959bd87` | `test_hyvla_proj1_ablation_example.py` |
| `frozen_suffix.py` | suffix/Expert0 query 冻结控制 | `c7e21b1b` / 同 A | `test_hyvla_frozen_suffix_example.py` |
| `frozen_eager_attention.py` | 固定 Q/K/V 的布局与顺序控制 | `78208988` / 同 A | `test_hyvla_frozen_eager_attention_example.py` |
| `suffix_key_order.py` | 固定 Expert 前缀顺序干预 | `50b86d7f` / 同 A | `test_hyvla_suffix_key_order_example.py` |
| `expert_mlp.py` | 记录 Expert0 MLP/GU 控制 | `179218ec` / 同 A | `test_hyvla_expert_mlp_example.py` |
| `expert_layers_capture.py` | 32 层 hidden/residual 观察 | `a0ff10db` / 同 A | `test_hyvla_expert_layers_capture_example.py` |
| `expert_layers.py` | GoldKV 下 teacher/propagated 重放 | `fe66081a` / `0c9a21aa` | `test_hyvla_expert_layers_example.py`、`test_hyvla_expert_layer_core_example.py` |
| `expert_order_replay.py` | separate-QKV 与 key order 交互 | `5248e14b` / 同 A | `test_hyvla_expert_order_replay_example.py` |
| `expert_rms_stages.py` | 固定 RMS 逐阶段差异 | `b522e8fd` / 同 A | `test_hyvla_expert_rms_stages_example.py` |
| `expert_rms_handoff.py` | RMS 中间值切点注入 | `af8e74df` / 同 A | `test_hyvla_expert_rms_handoff_example.py` |
| `expert_rms_block_handoff.py` | 四个实际 block 的方差注入 | `0aeea139` / 同 A | `test_hyvla_expert_rms_block_handoff_example.py` |
| `mean_launch.py` | 固定 FP32 mean 的 launch 观察 | `a9e3acfa` / `071ea364` | `test_hyvla_mean_launch_example.py` |
| `mean_tree.py` | 显式 FP32 加法树原型 | `19e87bcc` / 同 A | `test_hyvla_mean_tree_example.py` |
| `expert_mean_tree.py` | GoldKV 条件 Expert 的 RMS 树 | `d8cdfec9` / 同 A | `test_hyvla_expert_mean_tree_example.py` |
| `expert_policy_scope.py` | eager NFE 内的 Expert 临时覆盖 | `a1af00a3` / `7c828801` | `test_hyvla_expert_policy_scope_example.py` |
| `native_policy_tree.py` | 七例 free-running 旧逐元素诊断 | `7c828801` / `54b84033` | `test_hyvla_native_policy_tree_example.py` |
| `merger_stages.py` | 完整 merger 阶段重放 | `c990f271` / 同 A | `test_hyvla_merger_stages_example.py` |
| `proj2_calls.py` | PROJ2 几何及 bias 表达式控制 | `32ea141e` / 同 A | `test_hyvla_proj2_calls_example.py` |
| `proj2_policy.py` | PROJ2 临时 policy 干预 | `a5078137` / 同 A | `test_hyvla_proj2_policy_example.py` |
| `policy_factorial.py` | Expert × PROJ2 旧 factorial 矩阵 | `e97f22e7` / 同 A | `test_hyvla_policy_factorial_example.py` |
| `frozen_vit_prefix.py` | ViT block0–3 历史观察 | `65646f6f` / `311c66d0` | `test_hyvla_frozen_vit_prefix_example.py` |
| `frozen_vit_attention_bridge.py` | block0 context/projection 切点 | `2705333a` / `e058161a` | `test_hyvla_frozen_vit_attention_bridge_example.py` |
| `frozen_vit_lse.py` | 无权重 Flash row-LSE 观察 | `e175358e` / `cf4d9741` | `test_hyvla_frozen_vit_lse_example.py` |

合计删除 29 个 Hy 脚本和 29 个专用测试模块。官方观察接口原 pin 为
`af57e7507ec5964b52fdf6296741e553cbcd3288`；原文及观察代码仍在上面的
Git 水位。`validate_hyvla.py`、作者计时脚本、tradeoff 契约与
`test_hyvla_tradeoff_objective_example.py` 均保留。

## FastWAM pilot、profiler 与通用工具

| 资源 | 处理 | 来源、引用和理由 |
|---|---|---|
| `examples/vla_autotune/fast_graph/velocity_graph.py` | 删除 | A=C=`73b381bc`；仅旧 Euler2 pilot 和专用测试使用。现行 `flash_rt/models/fastwam/velocity_graph.py` 已承担库内图执行，且有独立生命周期、刷新及失败回收测试。未假定两文件存在未记录的上游 copy pair。 |
| `examples/vla_autotune/fast_graph/run_velocity_graph.py` | 删除 | A=C=`73b381bc`；只封装旧 pilot，当前 `validate_fastwam.py --use-cuda-graph` 已提供库内验证入口。 |
| `tests/test_fastwam_velocity_graph_example.py` | 删除 | 只加载被删 Euler2 pilot；保留 `test_fastwam_velocity_graph.py` 与真实 CUDA 功能测试。 |
| `tests/test_fastwam_graph_driver_example.py` | 删除 | 只测旧 driver 的 refresh pairs；库内 validator 的刷新验证及测试保留。 |
| `examples/vla_autotune/fast_profile/profile_fastwam.py` | 保留 | A=C=`73b381bc`；`vla-stage-profile` 仍明确引用，提供独立 CPU/CUDA trace、Linear ledger 和 kernel busy union。通用 stage intervals 不能替代其 host/launch 诊断。 |
| `tests/test_fastwam_profile_example.py` | 保留 | 仍验证上述诊断入口、缺失 CUDA 事件及结果标记。 |
| Hy `official_local_fp32`、`precision.hy_numerical_profile` | 保留已有路径 | 选项仍在 ModelSpec、SM80/Thor frontend、作者差异验证 CLI 和数值门槛指引中使用。默认性能对照使用 `semantic_bf16`。该能力仍可显式调用，因此保留现行配置和回归测试。 |
| `flash_rt/vla/autotune/{profiling,stages,dataflow}.py` | 保留，交 F 维护 | worker、stage 归因和 `profile.*` 注册项仍使用；A 未改这些 harness 文件。 |
| `flash_rt/vla/autotune/tools/**` | 保留，交工具负责人维护 | builtin 配置、外部适配器、capture、patch/artifact 身份均在注册链上；`needs_setup`/`blocked` 表示能力边界，不能据此判为无用。 |
| `examples/vla_autotune/fastwam-source/**` | 保留 | pin=`7faa71108368fbb3b6885649f112af607427a2d4`，是现行加载/作者基线契约；sidecar 明说无独立内容 hash，未把它升级成完整内容证明。 |
| `examples/vla_autotune/apxinf/**` | 保留 | 有工具注册和外部基线用途，未因当前 A800 不直接采用而删除。 |

## 第三方技能与来源

`third_party/agent-skills/manifest.json` 的 91 个文件逐项通过 blob/字节数、
NOTICE 和 VENDOR 表一致性检查，新增水位均为 `3e1c77aa`。
这证明 pin 完整性，不证明上游默认分支此刻没有新变化。本次没有联网更新 pin。

| 来源 | 文件数 | pin | 处理 |
|---|---:|---|---|
| TensorRT-LLM | 51 | `ef83ce246944131336e657681a951fa1c786bc6d` | 保留 graph、sync、profiling、Triton 技能及许可证 |
| ForgeKernel | 14 | `b2c8ae8510e02c28576808910a6b8827721ab7dc` | 保留启动技能、参考归档及许可/NOTICE |
| ApxInf | 9 | `b05b40990387e8c48e7d6267a790664981a665e4` | 保留四个改编技能的原始出处和许可/NOTICE |
| torchtitan | 4 | `631d544b1f6794ecfeca734a7bba861e03ddd5e3` | 保留数值校验和 hill-climb 参考来源、BSD 许可 |
| AITune | 13 | `1d3a4236ad8f458669d5633715b6956125f2289c` | 保留四个现行技能及 LICENSE/COPYRIGHT |

21 个现行技能与 `.claude/skills/` 的对应 symlink 保留；未发现需要删除的
失效 symlink。只修正 FlashRT 改编技能对旧 Hy/graph pilot 的当前路径描述，
并在各自 NOTICE 追加修改说明；verbatim 原件、manifest 和上游许可未改。

## 报告、token、快照

| 资源 | 处理与边界 |
|---|---|
| `docs/reports/**` | A 检查点全部保留；J 后仅撤下下表八份重复 Markdown。11-14 延迟 JSON/CSV、全部旧 token JSON/CSV、专门架构与移植来源说明继续保留。 |
| `docs/vla_autotune_experiments.md` | J 将完整历史正文迁入整合报告附录。迁移最后一处主 skill 引用后删除重定向，原文可由指定基线提交取回。 |
| `docs/vla_autotune_token_segments.jsonl` | 只追加本阶段 start/end，不重写其他会话段；计数由 J/H 汇总。 |
| `docs/vla_autotune_decisions.jsonl` | 通过 `decisions add --actor agent` 追加 Codex 决策，不改已有记录。 |
| `flash_rt/vla/autotune/snapshots.py` 及 campaign snapshot 测试 | 保留；是 plan v3、native identity、harness pin 的现行实现，A 未编辑。 |
| 旧源码快照、候选 worktree、Slurm 结果 | 不删除。当前 Git worktree 列表包含其他任务拥有的候选，且 A 没有得到完整的 manifest/决策/当前 best 绑定，不能推断为过期。 |
| 被删诊断测试内的临时 `snapshots/round*` fixtures | 随专用测试退出；只删除测试源码，没有清除磁盘上的实验快照。 |

恢复旧源码示例：

```bash
git show 199376281801427abccbb307056b27e74eab1429:examples/vla_autotune/hy_trace/frozen_vit_lse.py
git show 199376281801427abccbb307056b27e74eab1429:tests/test_hyvla_frozen_vit_lse_example.py
```

旧报告内绑定不可变 commit 的 GitHub 链接仍指向原实验版本，保留原链接。
当前示例 README 已撤下旧可运行入口，改为清理映射和 Git 查阅方式。

## 验证与待处理项

默认 `/usr/bin/python` 未安装 pytest；实际测试使用隔离的
`uv run --no-project --with pytest --with numpy --with pyyaml python` 运行。
原始要求的套件范围如下：

```bash
python -m pytest -q tests/test_vla_campaign_*.py tests/test_vla_experiment_*.py tests/test_vla_autotune_*.py tests/test_vla_agent_skills.py
python third_party/agent-skills/verify_vendor.py
```

| 检查 | 实际结果 |
|---|---|
| AGENTS CPU 套件，第二次集成检查 | 690 passed、7 skipped、21 failed（14.81s）。失败集中于其他负责人正在改动的 profile/ModelSpec/tool/token 接口；A 未修改对应实现。不能据此宣称共享框架验收通过。 |
| 保留的 Hy fixture/tradeoff、FastWAM fixture/profile/graph/adapter、延迟证据与 agent-skills 测试 | 231 passed、45 skipped、10 subtests passed（1.24s）。Torch 可选功能在当前未安装 Torch 的 CPU 环境跳过。 |
| vendor verifier | 91 个文件、5 个来源全部通过；与清理前 pin 相同。 |
| 链接和导入审计 | 801 个当前 Markdown 本地链接、21 个 tracked symlink；新增断链 0、失效 symlink 0、旧模块 Python 引用 0。 |
| Git 来源与删除映射 | 32 个审计脚本（含保留 profiler）的 A/C 水位有效；62 个删除路径全部在本文映射；31 个旧脚本均可从固定 Git 基线读取。 |
| 历史报告及第三方归档保全 | `docs/reports/` 与 `third_party/agent-skills/` 的 58 个原有文件逐字节保留；活跃 vendored 技能另由 verifier 校验。 |
| `git diff --check` | 通过。 |

共享套件第一次集成检查为 677 passed、7 skipped、22 failed；上述第二次检查
是在其他接口继续变化后执行，保留实际结果，未将失败重写成通过。原始命令输出、
失败名称和跨负责人交接放在本地清理验证日志中。主代理应在所有负责人收口后
重新运行完整套件。

链接审计覆盖当前 Markdown 本地链接、所有 tracked symlink、被删模块的剩余
Python 引用，以及当前文档内引用的 Git 提交/历史路径。上游原样归档中已有的
16 处相对链接缺项单列保留；另有 1 处无关训练文档链接缺项。
仓库已有一个带 Python 占位语法的模板，对其执行文本引用检查；
没有忽略可执行源码的意外语法错误。无关产品和其他平台内容不在本轮顺手清理。

待主代理/F/其他负责人处理：

- 核心 controller、stage/profiler、模型适配器与工具表由其负责人演进；A 不修改。
- 实现地图的完整内容注明是 2026-11-15 的阅读起点，其旧 SM80 融合限制、
  文本 K/V 重算说明不能作为当前能力结论。完成新 model spec/工具/归因接口后，
  主代理再同步完整地图；A 只清理本次失效的资源入口。
- `validate_fastwam.py` 的 saved-output compare 退出码仍是既有诊断语义；
  README 保留明确说明，A 不改当前功能入口。
- 上游 ApxInf 归档缺失的相对链接保留原字节。训练文档中指向
  `flashvla-ft/` 的既有缺失链接属于无关范围。
- J/H 收齐最终报告、token 统计、快照 manifests、当前 best 及未结束作业状态后，
  才能重新判断物理快照或结果目录是否满足删除条件。

## J 报告整合（2026-11-24，Codex）

现行结果入口统一为
[五模型整合报告](reports/2026-11-24-vla-autotune-final.zh-CN.md)。
该报告分开列项目、partial FlashRT、作者官方实现和 ApxInf，包含历史计时、
actor 缺口、原 schema／源码身份、数值失败、NR、研究认识修正及新验收待证表。
G／SmolVLA 的 GPU 作者／项目双门槛、重复计时和优化结果尚不可用，明确为 NR，
没有把已声明模型或 CPU 通过写成 GPU 资格。

**顺序：先迁移内容与引用，再删除冗余正文。** 本节不改写上方 A 阶段的测试事实；
其他负责人正在修改的框架、token 及来源文件不由 J 接管。决策用
`decisions add --actor agent` 追加，summary 明确写 Codex，无提交或推送。

| 旧入口（位于 `docs/`） | 本轮处理 | 保留位置与删除理由 |
|---|---|---|
| `reports/2026-11-09-vla-autotune-report.zh-CN.md` | 删除重复 Markdown | 早期四模型执行、graph、筛选和后续反例并入整合报告 §4—6；旧摘要的 Euler13、Hy EXACT 状态有历史范围，避免继续被误读为当前 best。原文仍在固定 Git 基线。 |
| `reports/2026-11-12-vla-autotune-research-process.zh-CN.md` | 删除重复 Markdown | “问题—区分实验—反例—认识修正”合并到 §6；R34—40 结果前预测通过固定 Git 原文保留，不以事后总结替代其预注册来源。逐轮数值在完整附录保留。 |
| `reports/2026-11-14-vla-latency-comparison.zh-CN.md` | 仅删除 Markdown | 历史大表、边界、同卡比较、失败与文献并入 §2／5／9；29 点原始 JSON/CSV 逐字节保留，不回填新测量。 |
| `reports/2026-11-23-dm05-author-loop.zh-CN.md` | 删除重复 Markdown | 全部首测／候选／复测、项目与作者质量、诊断数值及整批超时边界并入 §5.5。 |
| `reports/2026-11-23-fastwam-text-kv-campaign.zh-CN.md` | 删除重复 Markdown | SIGSEGV 的三点 NR、后续 eager／compiled 六点、144 次正式请求、质量与漂移并入 §5.3；热缓存不冒充冷缓存复验。 |
| `reports/2026-11-23-hy-config-loop.zh-CN.md` | 删除重复 Markdown | 八个配置、280 次正式请求、七输入双侧质量及 drift 并入 §5.4；不追认为本轮 schema 3。 |
| `reports/2026-11-23-hy-sm80-fusion.zh-CN.md` | 删除重复 Markdown | 三点延迟、全部诊断值、项目／作者误差及 4.43% drift 并入 §5.4；源码身份限制另列。 |
| `reports/2026-11-23-pi05-sm80-int8.zh-CN.md` | 删除重复 Markdown | 两轮共八点的延迟、P95、质量、最坏输入、86／176 GEMM 覆盖和未定位首层的限制并入 §5.2，两个失败版本都保留。 |
| `vla_autotune_experiments.md` | 撤下入口 | 原完整 1524 行实验记录迁入整合报告附录，调整了相对链接和标题。 |

被删除 Markdown 的完整原文均可从公开固定提交
[`199376281801427abccbb307056b27e74eab1429`](https://github.com/20261023/FlashRT/tree/199376281801427abccbb307056b27e74eab1429/docs/reports)
按原路径读取；整合报告 §9 提供逐文件固定链接。
append-only 决策日志的旧 `evidence_refs` 仍是当时路径，不反写历史；本节及上述固定
Git 映射负责解析它们。旧报告提交与实际运行 source_ref 明确分开，缺少的 manifest
不由整理者补造。

### 引用迁移与不删范围

- 当前 autotune／backend／学习文档、工具流程、FastWAM 缓存说明和 examples
  README 均转到新报告；旧固定 Git 链接保留，不改写成当前实现。
- 保留 FastWAM 专门架构／ApxInf 来源说明，但增加“历史阅读边界”，明确缓存的
  “尚未测量”属于旧日期。保留 DM0.5 组合优化的内容身份和冷成本细节。
- J 检查点保留 11-14 延迟 `.json/.csv`、11-15／16 token 原始 JSON/CSV 与说明、
  当前 H 四表；H 后续确认复现并封装九个重叠入口，见下方 [H 历史保留](#h-history-retention)。
- J 未删除 token segment／decision append-only 账本、作者计时入口、外部工具源增量、
  LICENSE／NOTICE、snapshot 实现、物理源码快照、候选 worktree 或原始结果目录。
  作者入口由 C 的声明适配迁移另行维护，不由报告清理认定其运行资格。
  **报告合并不自动授权物理快照清理。**
- 本轮框架／G 新实测留待主代理填入公开聚合证据；本次只有文档迁移，不把
  历史测试数字或现有文档检查冒充精确 staged tree 的最终 CPU／GPU 验收。

### J 已执行的保全检查

| 检查 | 实际结果与边界 |
|---|---|
| 旧实验正文迁移 | 迁移前 1524 行正文除标题和列明的链接适配外完全相同；从 `Experimental boundary` 开始的全部数值正文还与初始固定基线逐字节相同。 |
| 历史机器数据 | 12 份旧报告 JSON/CSV 与 `19937628` 字节相同；11-14 保留 29 点，其中 28 个完成计时、1 个原始 NR。 |
| 11-23 五报告 | 各文件原有小数观测值均可在整合正文找到；八份撤下报告都有逐文件固定 Git 恢复链接，旧来源可解析。 |
| 当前文档链接 | 检查时共 809 个 Markdown 本地链接，新增断链 0；22 个 tracked symlink 无失效；16 处上游归档与 1 处训练文档既有缺项未改。整合报告显式锚点亦逐项核对。 |
| 全量审计的非文档发现 | 同次审计仍报告 `flash_rt/vla/author_adapters/dm05.py:192` 指向已删除作者脚本，已交主代理/C 核实；J 未修改该适配器，**不宣称全量审计退出成功**。 |
| `git diff --check` | 通过。最终精确 staged tree 的 CPU／GPU 验收由主代理另跑，不以文档保全检查替代。 |

<a id="h-history-retention"></a>

## H 历史 Token 保留与重叠入口收口（2026-11-24，Codex）

H 采用后只保留一个当前历史输入
[vla-token-history.json](reports/vla-token-history.json)，附
[原发布分组、截点、归属和复现说明](reports/vla-token-history.zh-CN.md)。
包内嵌入 11-15 完整阶段来源及 11-16 三份完整原 JSON，原始 JSON 序列化摘要逐项
对应 `199376281801427abccbb307056b27e74eab1429`。阶段原名、模型分组、计数、
时间、缺失记录、归属依据和迟到结算说明不变；不把原发布分组重写成事后新分组。

`previous_latest_cumulative_source` 绑定 11-16 编译优化后旧最新基线：
**1,741,698,368 总处理／51,414,720 非缓存／1,690,283,648 cache-read**。
11-15 原发布与 11-16 两个中间快照仍各有原身份及原值。只对互不重叠的窗口求和，
不累加多个同起点累计快照，也不重复相加三份完整新增窗口。旧 14 条缺失保持未知；
DM0.5 的历史接入阶段保持 unknown。旧最新基线不会随本轮四表重新生成而变动。

先验证内容和复现、迁移当前引用，再从当前工作树撤下以下九个重叠入口。
表中链接均固定到原提交；包内 `archived_artifacts` 保存每个原路径、字节数和
SHA-256，可逐文件复核。11-15 的五份原发布文件继续保留；其他非 Token 数据不在此
清理范围。J 对初次加列 Markdown 所做的现行延迟入口迁移由新历史说明承接。

| 撤下路径（相对 `docs/reports/`，链接为固定 Git 原文） | 内容保留与删除原因 |
|---|---|
| [2026-11-16-vla-development-token-usage.json](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage.json) | 完整嵌入 `snapshots[1].data`，摘要 `948b7bb8…`；初次五模型中间累计窗口，不再维护单独统计输入。 |
| [2026-11-16-vla-development-token-usage.csv](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage.csv) | `snapshots[1].data.models` 可重建相同 LF 字节和原摘要；与 JSON 分组重复，撤下单独副本。 |
| [2026-11-16-vla-development-token-usage.zh-CN.md](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage.zh-CN.md) | 原数、窗口、排除刚开始的 DM 补接入和活动估算限制归入历史说明，延迟链接转到 J 报告；原完整措辞由固定 Git 保留。 |
| [2026-11-16-vla-development-token-usage-post-dm05.json](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-post-dm05.json) | 完整嵌入 `snapshots[2].data`，摘要 `d862d263…`；保留补接入中间窗口及一条迟到结算，不与初次窗口相加。 |
| [2026-11-16-vla-development-token-usage-post-dm05.csv](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-post-dm05.csv) | 从 `snapshots[2].data.models` 重建相同字节；作为重复导出撤下。 |
| [2026-11-16-vla-development-token-usage-post-dm05.zh-CN.md](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-post-dm05.zh-CN.md) | 计时边界、76 会话范围、迟到结算、不把活动差额解释为纯编码成本等保留于历史说明；撤下重复报告正文。 |
| [2026-11-16-vla-development-token-usage-dm05-optimized.json](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-dm05-optimized.json) | 完整嵌入 `snapshots[3].data`，摘要 `c492024f…`，由旧最新指针固定；当前包的历史总量读取该对象，撤下独立重叠输入。 |
| [2026-11-16-vla-development-token-usage-dm05-optimized.csv](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-dm05-optimized.csv) | 可从 `snapshots[3].data.models` 按原顺序、原 LF 换行重建并校验摘要；旧延迟 JSON 提供固定原 CSV 地址。 |
| [2026-11-16-vla-development-token-usage-dm05-optimized.zh-CN.md](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-16-vla-development-token-usage-dm05-optimized.zh-CN.md) | 原“最新”限定为历史旧最新；77 会话、799 条增量记录、最后完成 step 后 1 ms 截点和原累计均转入历史说明，避免继续当当前账单入口。 |

### H 当前引用与验证边界

- `token_usage.py` 的现行命令例与 CPU 历史接入测试均改传单一历史包。读取器保留
  原四份 `source` 与 SHA-256，拒绝摘要错误、同名不同内容及错误旧最新指针，
  与同一原文件重复传参也不会重复保留或计入。
- 11-14 延迟 JSON 只精确迁移 token 的 `source`、CSV 恢复地址和中间窗口指针；
  历史模型计数、29 个延迟点、质量数据及原 NR 均不改。J 整合报告只替换两条
  历史数据引用，不重写其结果正文。
- 生成报告的 CSV writer 显式使用 `lineterminator='\n'`；四份当前公开 CSV 的
  LF 归一化由主代理负责。本次没有重写归档原字节、摘要或现有账本记录。
- 冻结历史的三份 JSON／CSV、四个原截点、14 条缺失、各模型旧总量及阶段代理
  均有 CPU 行为回归；新模型 DM 接入阶段未知继续验证。H 03:57 六份生成输出在
  四个旧输入改为一个包后已实测相同；CSV 换行固定后的复核另列最终回执。
- 本段只追加自身 start/end 和 `decisions add --actor agent` 的 Codex 记录；
  不提交、不推送，不清理物理结果、源码快照、候选 worktree 或无关数据。
