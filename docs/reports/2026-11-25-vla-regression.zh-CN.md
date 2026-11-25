# A800 回归与 SmolVLA 接入复核

2026-11-25 从归档 plan、ledger、worker 和作者动作数组复算。各点按同一请求 ID、公共通道和输出域对齐，取最坏 relative L2 和最低 cosine。两道门均要求 L2 ≤ 0.05 且 cosine ≥ 0.999。时延取完整请求中位数。阶段覆盖取独立 profile 中可相加区间对诊断 wall 的占比。

| 模型、作业 | 执行顺序 | 作者 Native，ms | 来源 |
|---|---|---:|---|
| π0 359987 | E10 / E2 / E10 | 69.0 | 同一作业 |
| π0.5 360012 | E10 / E8 / E10 | 59.4 | 同一作业 |
| Hy 360007 | 融合 E8 / E7 / E9 / E8 | 787.0 | 同一作业 |
| DM0.5 360006 | compiled + graph / 关 compile / 关 graph / 全关 / 首配置复测 | 277.7 | 已绑定的历史保存动作 |
| FastWAM 360054 | 编译与文本 KV 的 E20 / E16 / E18 / E20 | 141.9 | 同一作业 |
| SmolVLA 361896 | 原源码 eager 首尾 | 377.8 | 同一作业 |

| 模型 | 作业测点 | 时延 ms | 项目 L2 / cosine | 作者 L2 / cosine | 覆盖 | 结论 | 主导 |
|---|---|---:|---|---|---:|---|---|
| π0 | pi0-f-project-fp16-euler10 | 40.1 | 0.000000 / 1.000000 | 0.002476 / 0.999997 | 100% | 双门通过 | agent |
| π0 | pi0-f-project-fp16-euler10-euler-2 | 26.1 | 0.038418 / 0.999420 | 0.039363 / 0.999377 | 100% | 双门通过 | agent |
| π0 | pi0-f-project-fp16-euler10 | 40.2 | 0.000000 / 1.000000 | 0.002476 / 0.999997 | 100% | 双门通过 | agent |
| π0.5 | pi05-f-project-bf16-euler10 | 41.9 | 0.000000 / 1.000000 | 1.658008 / -0.397237 | 100% | NR | agent |
| π0.5 | pi05-f-project-bf16-euler10-euler-8 | 38.4 | 0.036150 / 0.999482 | 1.650275 / -0.407734 | 100% | NR | agent |
| π0.5 | pi05-f-project-bf16-euler10 | 41.9 | 0.013640 / 0.999911 | 1.662127 / -0.394989 | 100% | NR | agent |
| Hy | hy-bf16-e8-graph-fused | 97.4 | 0.000000 / 1.000000 | 0.036192 / 0.999364 | 100% | 双门通过 | agent |
| Hy | hy-bf16-e8-graph-fused-euler-7 | 95.3 | 0.019738 / 0.999829 | 0.051769 / 0.998728 | 100% | NR | agent |
| Hy | hy-bf16-e8-graph-fused-euler-9 | 102.8 | 0.017974 / 0.999845 | 0.019765 / 0.999810 | 100% | 双门通过 | agent |
| Hy | hy-bf16-e8-graph-fused | 96.7 | 0.000000 / 1.000000 | 0.036192 / 0.999364 | 100% | 双门通过 | agent |
| DM0.5 | dm05-compiled-graph-e10-frozen | 96.8 | 0.000000 / 1.000000 | 0.009138 / 0.999961 | 100% | 双门通过 | agent |
| DM0.5 | dm05-compiled-graph-e10-frozen-lv-compile_off | 196.2 | 0.010069 / 0.999950 | 0.007773 / 0.999973 | 100% | 双门通过 | agent |
| DM0.5 | dm05-compiled-graph-e10-frozen-lv-graph_off | 147.2 | 0.006752 / 0.999979 | 0.012669 / 0.999921 | 100% | 双门通过 | agent |
| DM0.5 | dm05-compiled-graph-e10-frozen-lv-compile_off+graph_off | 434.7 | 0.010069 / 0.999950 | 0.007773 / 0.999973 | 100% | 双门通过 | agent |
| DM0.5 | dm05-compiled-graph-e10-frozen | 96.3 | 0.006752 / 0.999979 | 0.012669 / 0.999921 | 100% | 双门通过 | agent |
| FastWAM | fw-regression-compiled-textkv-e20 | 128.1 | 0.000000 / 1.000000 | 0.003945 / 0.999992 | 99% | 双门通过 | agent |
| FastWAM | fw-regression-compiled-textkv-e20-euler-16 | 107.6 | 0.190669 / 0.981708 | 0.188056 / 0.982208 | 99% | NR | agent |
| FastWAM | fw-regression-compiled-textkv-e20-euler-18 | 117.7 | 0.063478 / 0.997984 | 0.060801 / 0.998151 | 99% | NR | agent |
| FastWAM | fw-regression-compiled-textkv-e20 | 127.9 | 0.000000 / 1.000000 | 0.003945 / 0.999992 | 99% | 双门通过 | agent |
| SmolVLA | smolvla-onboard-eager | 383.1 | 0.000000 / 1.000000 | 0.000000 / 1.000000 | 98% | 双门通过 | agent |
| SmolVLA | smolvla-onboard-eager | 378.3 | 0.000000 / 1.000000 | 0.000000 / 1.000000 | 99% | 双门通过 | agent |

π0 七请求 E2 相对 E10 节省 14.0 ms（35%）。π0.5 保持 NR。Hy E7 的时延低于融合 E8，但两项作者误差均越线。DM0.5 的三个消融均通过且更慢。FastWAM E16 和 E18 均同时超出两道门。以上结果按各自 workload 解释。

SmolVLA 整策略 graph 在视觉掩码复制处失败。两次原源码 eager 的逐请求动作与作者一致，覆盖均超过 98%。新作业 364125 使用同候选源码 graph/eager 对照和原源码 eager 首尾。
