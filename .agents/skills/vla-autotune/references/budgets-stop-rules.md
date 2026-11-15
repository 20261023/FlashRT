# 预算和停止规则

规划一轮实验、决定是否继续或换方向时读这一篇。规则来自之前的设计文档（ForgeKernel、KernelAgent、TVM、Optuna 等调研的结论），以及本仓库 controller 的实际行为。

## 预算（H1 时和人一起定）

| 预算 | 默认建议 | 为什么 |
|---|---|---|
| 每个 campaign 的候选数 | ≤ 8 | 一次 allocation 串行跑完，首尾复测仍然有意义 |
| 同时在测的源码变体（patch / port） | ≤ 2 | worktree 只能隔离源码，不能隔离 GPU 计时；变体多了难以对比 |
| 外部 kernel 任务 | 每个算子一个独立 allocation，预算单独批（H3） | ForgeKernel 每个算子要数小时，外加 Claude agent token；KernelAgent 按 LLM 调用计费 |
| 总 GPU 小时 | 由人定 | 达到上限就停，并汇报 |

研究 agent 和编码 agent 最多 8 个并行；GPU 测量始终串行。同一块卡上不能同时跑两个计时任务。

## 测量规则

- 每个 campaign 的顺序都是 reference → candidates → bookend。首尾两次中位数漂移超过 10% 时，controller 不选 winner。
- 噪声底线：取 max(1.5%, reference 与 bookend 的漂移)。低于这个值的提升都算“没有变化”。
- 质量门槛在跑候选前固定。在每个请求上分别算，取最差的那个请求。
- 选出 winner 之后，要用更多输入确认，然后才能接受。像步数这类近似方法尤其要这样做。
- 冷启动成本：break_even_calls = ceil(额外准备时间 / (baseline 稳态耗时 − 候选稳态耗时))。分母不大于 0 或测不准时，记为 unknown。

## 两层循环

- **内层**：在同一个 stage 上，换工具或换参数。
- **外层**：内层停滞时，重新 profile 新的 baseline，换到下一个 stage。不要在同一个方法上机械地加大搜索次数。

内层停止条件，满足任意一条就停：

1. 连续 2 轮，最好的候选都没超过噪声底线。
2. 这个 stage 的 Amdahl 上限已经低于 5%。
3. 剩下的工具都是 `needs_implementation` 或 `blocked`。这时要由人决定是否投入实现（H2）。
4. 预算用完。

外层停止条件：所有占比明显的 stage 都已经停了，或者总预算用完。最后汇报“还剩多少空间、卡在哪”。

## 记账规则

- **KEEP 和 best 分开**：合法但没变快的中间结果可以 keep，但它不是 winner。
- **first-valid 和 fastest 分开**：一个工具生成成功，不等于它在性能上胜出。
- **诊断和候选分开**：消融、探针这类诊断请求不进入正式比较。
- **exact 和 approximate 分开**：近似方法除了要过门槛，还要扩大输入集确认。
- **局部和端到端分开**：kernel 局部变快、跳过了几次 expert、剪掉了多少 token，都只是机制上的指标。只有模型级 campaign 的结果算数。
- **失败也是结果**：OOM、编译失败、不支持、质量不过，都要记成 reject 或 note，写清原因和适用范围，避免下次重复尝试同一组已知不兼容的组合。
