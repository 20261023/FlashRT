# 按“游”选工具

先选模型声明，再看 stage 的工具覆盖和不可用原因。注册表读取
`flash_rt.vla.model_spec.get_spec`；新增本地 JSON 声明用 `--spec-dir`，
不需要往工具代码追加模型名单。

```bash
python -m flash_rt.vla.autotune.tools coverage --model "$M" --hardware cuda_sm80 --markdown
python -m flash_rt.vla.autotune.tools list --model "$M" --hardware cuda_sm80 --stage "$S" --all
python -m flash_rt.vla.autotune.tools show compile.inductor --model "$M"
python -m flash_rt.vla.autotune.stages show "$S"
# 外部声明示例：上面的 coverage/list/show 均可追加 --spec-dir /abs/model-specs
```

`coverage` 输出一张表：每个 stage 同时列出 ready 工具、缺失的模型特征、
硬件限制、未完成实现和外部环境要求。它是声明级路由，不能证明安装成功、
设备适配完成或实验通过。`ready` 按 lane 解读：config 可以生成已有选项的候选，
analysis/baseline 走声明的诊断或作者生产器；它们不一定有 `propose`。
`needs_setup` 表示还缺构建、环境或凭据；`needs_implementation` 表示还缺产品接线；
`blocked` 表示当前契约或设备无法使用。未知事实保留 unknown。

| stage（游） | 先补的证据 | 通用工具与匹配条件 | 不可用时看什么 |
|---|---|---|---|
| `host.input` | 输入处理、H2D 和完整请求边界 | `profile.stage_intervals`；`baseline.official_impl` 读取 `author.adapter` 和 `author.options` | 缺少声明的诊断 seam 或作者适配器时，补适配器；synthetic fixture 不能代替真实输入验收 |
| `vision.encode` | encoder 区间、输入分辨率、token 数、热点形状 | `compile.inductor` 需要已有 `tools.inductor` 配方；`compile.aitune` 需要 Torch module 或可包装的 functional Torch 区域；kernel 工具需要真实调用 capture | AITune 没有现成 FlashRT wrapper/artifact-loader 接线，仍为 `needs_implementation`；ncu 和外部 kernel 工具还需环境及 GPU 分配 |
| `vision.project` | 投影/merger 形状、prefix 行数 | `prefix.vision_pool` 仅匹配已声明 `vision_pool_factor` 的产品选项；stage profiler | pooling 是近似方法，减少 prefix 行数，不能假定减少 ViT 计算；不存在产品选项时先走 port/patch |
| `prefix.prefill` | prefix 区间、行数、每请求重复工作 | `compile.inductor`、`prefix.vision_pool`、`prefix.state_prompt_mode`、`fusion.hy_rope_qknorm_kvwrite` 都按声明配方/选项筛选；`profile.pi05_dataflow` 按 `profiling.dataflow` 筛选 | 固定 state prompt 只有实现该选项的前端可用；SM80 fusion 需要对应扩展构建；`engine.apxinf` 必须被声明为兼容外部 baseline，且计时边界对齐 |
| `denoise.velocity` | 区间/NFE、每次调用成本、重复投影和 cache 刷新 | `launch.cuda_graph`、`compile.inductor`；声明了对应选项才显示 linear backend、text K/V cache 或 fusion；kernel 工具 | compile 与 graph 的互斥/后端要求来自模型配方，不能静默组合；每请求 cache 必须刷新，准备开销计入比较 |
| `denoise.schedule` | 实际 NFE、prefix 占比、质量分布 | `sampler.euler_steps` 需要 flow-matching head、Euler 能力和可变步数声明；`sampler.probeflow` 还需要 ProbeFlow sampler 能力 | 固定步数、回归 head、autoregressive-token head 不应收到这些候选；默认步数未知则显式传参数，不能猜测 |
| `denoise.routing` | probe/check 时间、`.item()`/`.cpu()` 同步 | `sampler.probeflow` 只匹配声明支持的 flow-matching sampler；`profile.stage_intervals` | 跳过的计算若仍执行，或检查开销抵消收益，不能按 NFE 推断加速；没有 sampler 能力时先实现产品路径 |
| `host.launch` | eager/graph 对照、kernel 间隙、重编译 | `launch.cuda_graph` 需要 `use_cuda_graph`；`compile.inductor` 需要已实现配方；固定 prompt 和 cached linear 需要对应选项 | 不为原生 buffer/fvk 路径臆造 torch.compile 能力；记录冷编译成本与 break-even 调用次数，遵守配方的 framework、sampler、hardware 约束 |
| `host.output` | D2H、同步、动作解码和端到端延迟 | `profile.stage_intervals`；作者生产器的同域输出 | GPU-only 区间不能代替完整请求，未覆盖的 host 时间应明确标出 |
| `kernel.gemm` | 少数热点的形状、dtype、ncu 指标 | `kernel.forgekernel`、`kernel.kernelagent`；已实现 linear selector；`compile.inductor`/`compile.aitune` 的可编译区域 | 外部工具为 `needs_setup`，局部 kernel 通过后仍需算子 gate 与源码快照 campaign；first-valid 不等于最快 |
| `kernel.attention` | 每个 attention site 的 mask、head_dim、Q/KV heads、GQA、dtype、实际 backend | `attention.sdpa_backend`、`attention.flash_attention` 明确为 port；也可对真实 capture 使用 kernel 工具 | 两个 attention port 都是 `needs_implementation`。已有 FA2 调用或 PyTorch 支持某 API，不能证明当前产品已有可选择、已验收的替换路径；未知 mask/head 事实不可视为兼容 |
| `kernel.norm_elementwise` | norm/RoPE/QK-norm/KV-write 的 launch 与舍入 | 声明了 `fused_rope_qknorm_kvwrite` 的 config selector；kernel 工具 | 对应扩展未构建为 `needs_setup`；新增融合必须先过独立 oracle、同 dtype baseline 与 capture/replay gate |
| `kernel.conv` | VAE/patch embedding 热点、布局与 kernel 计数 | ncu、kernel 工具；可包装 Torch module 区域可考虑 AITune | 原生 kernel 不自动成为 AITune 可包装 module；没有调用 capture 或产品接线时保持不可用 |
| `precision.weights` | 权重字节、设备能力、精度余量 | `quant.int8_pi05` 按 `int8_mode` 能力、Torch/Euler/SM80 筛选；numerical profile 按选项筛选 | FP8 需要 SM≥89，NVFP4 需要 SM120 路径，且统一 backend 的量化禁用约束仍有效；SM80 不显示为可用。数值对齐 profile 不代表提速 |

工具的 `model_predicate` 是可检查的 JSON：`all`/`any`/`not` 组合
`path + equals/contains/exists/in`。它读取 head、capabilities、features 或 tools
元数据；缺失和显式 unknown 都不能满足正向能力要求。
`tools.unsupported` 优先拒绝路由。选项存在只表示产品暴露该选项，具体值、
互斥关系和模型质量仍由配置验证和 campaign gate 检查。

`compile.inductor` 统一应用模型声明的 `tools.inductor`：目标 backend、
options、remove_options、明确的 stages 和 framework/sampler/hardware 约束。
缺少 stage 声明的配方不会列为可用；候选归因到配方的首个 stage。它只生成
BackendConfig，不在 measurement harness 包装或编译模型。历史
`compile.fastwam_inductor`、`compile.dm05_inductor` 仍可解析；新列表和候选
使用 canonical ID。外部 AITune 没有这个现成 config 入口，必须先在独立
worktree 完成 wrapper 和固定 artifact 的加载接线。

正式时延请求不加诊断 hook；区间归因来自独立诊断请求。交叉覆盖的 stage、
host 时间和 kernel 类别不能相加成总时延。若当前策略只有 `full` 或合并区间，
先查看 coverage，再补能区分假设的观测，不把未拆开的总区间称为单游实测。

选工具前确认四件事：减少哪段关键路径及其占比；增加什么 probe、拷贝、同步、
编译或重录成本；方法是否 approximate；有没有更便宜的观测先否定假设。
步数/近似方法必须扩大输入集检查质量，不能从少量 fixture 推断任务成功率。
非 config 工具和任何源码变化按 source-candidate checklist 使用 plan v3
及双侧快照；报告 ready 或外部 baseline 从不豁免这个规则。
