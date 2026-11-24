# VLA autotune 清理映射

A 阶段由 Codex 执行。审计起点是
`199376281801427abccbb307056b27e74eab1429`；仓库不是 shallow clone。
以下“删除”只指当前工作树入口，旧源码、测试及其 Git 历史仍可按提交取回。
历史报告、token 原始计数、动作结果及快照身份不在本阶段裁剪。

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
融合，见 [配置扫描](reports/2026-11-23-hy-config-loop.zh-CN.md) 和
[融合报告](reports/2026-11-23-hy-sm80-fusion.zh-CN.md)。FastWAM 报告记载
Euler20、compiled prefill/denoise 和请求内文本 K/V 缓存，见
[同轮缓存比较](reports/2026-11-23-fastwam-text-kv-campaign.zh-CN.md)。
这些实现、选项和报告均保留；最终 best 以本轮后续决策为准。

## 删除的 Hy 旧诊断：逐文件来源和测试

下表脚本路径统一位于清理前的 `examples/vla_autotune/hy_trace/`；
测试路径统一位于清理前的 `tests/`。表中的 A 是 `git log --diff-filter=A -1`
的新增提交，C 是清理前最后修改提交；均为 Git 中可唯一解析的提交前缀。
这些文件是 FlashRT 研究代码；观察官方符号的部分按 semantic-review-only
核对，未把它们误认成上游 vendored 模型实现。

所有行都退出当前目标：Hy 已改为 public20 的延迟—误差预算，旧 EXACT、
固定 Gold 轨迹、逐层 RMS/attention/LSE 闭环不再是速度搜索前提。
历史负结果仍完整留在
[实验记录](vla_autotune_experiments.md) 和
[研究过程](reports/2026-11-12-vla-autotune-research-process.zh-CN.md)。

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
| `docs/reports/**` | 全部保留，包括旧失败、修订、延迟比较及原始 token JSON/CSV；等 J/H 汇总后再判冗余。 |
| `docs/vla_autotune_experiments.md` | 完整历史正文保留，只将入口说明改成当前脚本与历史 Git 源码的区分。 |
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

执行记录将在下列检查结束后填入实际结果：

```bash
python -m pytest -q tests/test_vla_campaign_*.py tests/test_vla_experiment_*.py tests/test_vla_autotune_*.py tests/test_vla_agent_skills.py
python third_party/agent-skills/verify_vendor.py
```

链接审计覆盖当前 Markdown 本地链接、所有 tracked symlink、被删模块的剩余
Python 引用，以及当前文档内引用的 Git 提交/历史路径。上游原样归档中已有的
相对链接缺项单列保留；无关产品和其他平台内容不在本轮顺手清理。

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
