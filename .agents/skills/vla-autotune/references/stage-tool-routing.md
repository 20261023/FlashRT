# 按“游”选工具

选 stage 和工具时读这一篇。这里列的是“先看什么、先试什么”。工具的当前状态以注册表为准：

```bash
python -m flash_rt.vla.autotune.tools list --model $M --hardware cuda_sm80 --stage $S --all
python -m flash_rt.vla.autotune.tools show $TOOL
python -m flash_rt.vla.autotune.stages show $S       # playbook 症状行和默认工具
```

状态的含义：

| 状态 | 含义 |
|---|---|
| `ready` | 可以直接 `tools propose` |
| `needs_setup` | 现成项目，但要先装环境或拿到凭据 |
| `needs_implementation` | 这一游还没有现成方法，要写 patch 或 port |
| `blocked` | 在这台硬件上或在当前接口下做不了 |

## stage 对照表

| stage（游） | 典型症状（playbook） | 先补的证据 | 先看的工具 | 对应 skill |
|---|---|---|---|---|
| `host.input` | 图片/帧数一多就慢；model_ms 好看但整体仍卡 | 计时边界内各步的时间戳 | `profile.stage_intervals`、`baseline.official_impl` | `vla-stage-profile` |
| `vision.encode` | 图片/帧数一多就慢 | stage 区间；视觉 token 数 | `profile.ncu_kernel`、`kernel.forgekernel`、`kernel.kernelagent`、`compile.aitune` | `vla-stage-profile`、`vla-kernel-candidate` |
| `vision.project` | 同上 | 投影或 merger 的形状 | `prefix.vision_pool`（π0.5，近似方法） | `vla-config-sweep` |
| `prefix.prefill` | prefix 占比大，减步数省得很少 | `profile.pi05_dataflow`；prefix 行数 | `prefix.vision_pool`、`prefix.state_prompt_mode`、`fusion.hy_rope_qknorm_kvwrite`；`engine.apxinf` 只作 π0.5 的外部对照，不是候选 | `vla-config-sweep`、`vla-port-workflow` |
| `denoise.velocity` | expert 很小却不快；去噪总耗时高 | 每次调用的耗时（区间总和除以 NFE）；每步重复做的工作 | `launch.cuda_graph`、`gemm.fastwam_linear_backend`、`cache.fastwam_text_crossattn_kv`、kernel 工具 | `vla-config-sweep`、`vla-patch-candidate`、`vla-kernel-candidate` |
| `denoise.schedule` | 步数少了，端到端却几乎没变；probe 方案没有收益 | `per_call_ms`、NFE、prefix 占比 | `sampler.euler_steps`、`sampler.probeflow` | `vla-config-sweep` |
| `denoise.routing` | 小 tensor 的 `.item()` / `.cpu()` 很贵 | 同步点的位置 | `sampler.probeflow` | `perf-torch-sync-free` |
| `host.launch` | GPU 时间线有长空洞；很多短 kernel | eager 和 graph 对比；kernel 计算时间之和与总耗时对比 | `launch.cuda_graph`、`prefix.state_prompt_mode`（避免重录 graph） | `perf-torch-cuda-graphs`、`perf-nsight-systems` |
| `host.output` | 读回和解码慢 | 拷贝和同步 | `profile.stage_intervals` | `perf-workload-profiling` |
| `kernel.gemm` / `kernel.attention` / `kernel.norm_elementwise` / `kernel.conv` | 已经定位到少数几个热点 kernel | 这几个 kernel 的 ncu 指标 | `kernel.forgekernel`、`kernel.kernelagent`、`gemm.fastwam_linear_backend`、`compile.aitune` | `vla-kernel-candidate`、`perf-nsight-compute-analysis`、`kernel-triton-writing`、`forge-start` |
| `precision.weights` | 带宽受限，且精度有余量 | 权重字节数，硬件是否支持该精度 | `quant.int8_pi05`（port）；`quant.fp8` 和 `quant.nvfp4` 在 sm_80 上不可用 | `vla-port-workflow` |

## 各模型的已知事实

这些是从代码里读到的事实，不是 winner；数字见 `docs/vla_autotune_experiments.md`。

- **π0 / π0.5**：Euler + CUDA Graph 时，profile 只有一个 `full` 区间，没法拆。要拆分时，做一次 ProbeFlow 或 eager 诊断请求。π0.5 的 prefix 是大头，先看 prefix 再考虑减步数。
- **Hy**：现在有 `vision`（ViT + merger 的 graph）、`prefix`、`velocity` 三个区间。旧结果没有 `vision`，会报 `partial`。当前接口固定走纯 BF16，FP8/FP4/fused 开关都被禁用。fused RoPE/QK-norm/KV-write 在 SM80 上没有编译，属于 port 或 patch 工作。
- **FastWAM**：`condition_vae_video_prefill`（VAE + video prefill，每次请求一次）加 `velocity`。文本 context 和 proprio 通过 cross-attention 进入模型，每一步每一层都重新投影；KV cache 里只有首帧视频。所以 `cache.fastwam_text_crossattn_kv` 是 exact 的 patch 机会，尚未实现、尚未测量。
- **步数类结论**：用少量输入选出的步数，可能在更多输入上失败。接受前一定要扩大输入集。
- **ProbeFlow**：默认 ε 可能把所有输入都判成“直线”，只走 2 步，误差很大。用之前先看 cosine 的分布。

## 选工具前回答

1. 它减少的是关键路径上的哪一段？这一段占比多少？用 `amdahl_ceiling(share, s)` 算上限。
2. 它新增什么开销？比如 probe、拷贝、同步、编译、graph 重录，以及冷启动的 break-even 调用次数。
3. 它是 exact 还是 approximate？如果是 approximate，要走 H2。
4. 有没有一个更便宜的观测能先否定它？有就先做那个观测。
