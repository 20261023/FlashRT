# FlashRT 多模型 VLA Agentic Autotune：实现、实测与数值一致性报告

**迭代标识：**2026-11-09 轮次，包含该轮后续排队完成的实验。

**覆盖模型：**π0、π0.5、FastWAM、Hy-Embodied-0.5-VLA。

**报告依据：**截至代码与结果记录提交 `63676648`；关键实验使用各自冻结的源码快照，而非事后统一替换版本。

**报告性质：**工程实验总结，不是模型排行榜、机器人任务成功率报告或全硬件认证。

## 1. 摘要与最终判断

本轮完成了从统一模型执行接口，到可恢复实验账本、隔离 worker、实际 GPU 测量、质量门禁和候选选择的闭环。优化决策由实际执行、完整动作输出与预先声明的验收范围决定，没有预设 FlashRT 原生路径必然快于 Torch，也没有放宽质量门槛保留更快的失败点。

| 模型／工作项 | 本轮可支持的结论 | 不能据此宣称的结论 |
|---|---|---|
| π0 | 在已测试的两步比较中，ProbeFlow2 数值代理通过但比 Euler2 慢，保留 Euler2 对照 | 两步策略已达到默认策略的机器人质量；完成了 π0 全参数搜索 |
| π0.5 | 四请求严格 v2 campaign 选择 Euler8：38.4342 ms，相对同轮 Euler10 的41.6211 ms 降低7.66% | 单一输入上的 Euler7 结果可泛化；数值代理等同任务成功率 |
| FastWAM 执行层 | Torch 与部分原生 cached 路径的可选 raw-velocity graph 均通过受测 graph/eager 保持性验证；Torch graph 更快 | 部分 Linear 原生替换就是完整原生移植；CPU enqueue 减少意味着 GPU 算术消失 |
| FastWAM 参数筛选 | 四请求、固定质量代理下选择 Torch graph Euler13：179.7011 ms，相对同轮 graph Euler20 的251.6074 ms 降低28.58% | 全局最优、held-out 质量合格或可自动替换生产默认 |
| Hy 数值一致性 | HWD PROJ1 原型恢复了冻结算子的逐位一致性，原七 fixture gate 从4/7变为6/7 | Hy 已修复、已达到完整官方一致性；失败的 state 案例可以忽略 |

**最重要的保留意见：Hy 的候选仍未通过完整原 gate，作业 `309681` 正确保持 `FAILED / 1:0`。本轮没有推广该原型，也没有修改模型默认、attention backend 或原容差。**

## 2. 目标、边界与实验方法

### 2.1 目标

建立可审计的优化流程：

```text
源码／数据流假设 + 明确候选配置
  → 冻结源码与工作负载
  → 同一 allocation 内 reference → candidates → bookend
  → 保存延迟、实际 NFE、完整动作与独立 profiling
  → 质量门禁、身份核验、漂移检查
  → 保留／拒绝／下一步定位
```

Agent 负责分析、提出候选、修改代码和调度实验；不在正式计时进程里递归调用 LLM，也不把自然语言推测当作通过证据。本轮首先落地 FlashRT + Euler/ProbeFlow，没有同时扩展大量其他算法。

### 2.2 测量边界

- GPU 为 NVIDIA A800-SXM4-80GB；每个成组对照在单 GPU allocation 内使用串行、独立进程，避免候选之间保留模型或 graph 状态。
- 原生执行环境主要为 Python3.11、Torch2.11.0+cu128；Hy 官方参考单独使用 Torch2.7.1+cu128、外部 FlashAttention2 2.7.4.post1。
- 正式请求计时覆盖 frontend 处理、必要传输、模型执行、采样与完成后的 CPU 输出；checkpoint 加载、prepare/capture、warmup 和产物核查另行记录。
- **FastWAM 特定边界：**仅测试 first-frame 条件下的动作生成，使用确定性合成双相机输入、normalized proprio 和预先准备的 T5 context。正式计时包含输入上传、VAE、每请求 video KV prefill、采样和 CPU 动作输出，**不包含 T5 文本编码，也未评估未来视频 rollout**。其毫秒数不是包含文本编码的完整请求延迟。
- CUDA-event 或 profiler 请求独立于正式延迟样本。事件区间可能包含提交／依赖间隙，不能冒充 kernel 时长总和。
- 保留原始延迟样本、完整动作数组、实际 NFE 和输出域；不能只保留均值或“passed”标签。
- 不跨 allocation 拼接一条“累计加速比”，也不将不同模型、输入或采样预算的毫秒数直接作为横向排行榜。

### 2.3 数值质量不等于机器人质量

π0.5 和 FastWAM 的四请求筛选预先使用 normalized public7／normalized7 的工程代理：所有请求均需 relative L2≤0.05 且 cosine≥0.999。该阈值不是通用 VLA 容差，也不是用户确认的机器人验收标准。

Hy 保留既有独立门槛：padded `40×32` 的每个元素都满足

```text
abs(candidate - reference) <= 0.01 + 0.01 * abs(reference)
```

同时 cosine≥0.999。Public `40×20` 单独报告，不能代替原 gate。一个 cosine 很高、RMSE 较低的结果，仍可能因少数元素超差被拒绝。

## 3. 已实现的框架与关键契约

### 3.1 执行层、采样层与实现层分离

统一边界包含候选配置、backend 构造、prepare、infer 和显式清理；observation 仍服从各模型自己的相机、状态和历史契约，不猜测通用机器人预处理。

- π0 使用 FP16 路径；π0.5 使用 BF16 路径。不能把 π0 的 FP16 action buffer 当成 BF16 重解释。
- Pi 的 `framework="jax"` 在这里表示使用既有 Orbax 权重加载接缝驱动原生 CUDA 数学，不代表新增了一套独立 JAX 推理引擎。
- Hy 输出是 normalized padded model actions；FastWAM 输出是 normalized action chunk。它们不能直接作为已完成单位换算的机器人控制命令。
- π0.5 动作解码遵从 checkpoint 声明的 `ACTION=MEAN_STD`，不能因为 checkpoint 同时存储 quantiles 就擅自改用另一种反归一化。

Euler 与 ProbeFlow 共用真实的非积分 raw-velocity evaluator，而不是调用一个已经完成采样的接口来伪造速度。FastWAM 保留 shifted sigma：

```text
sigma(u) = shift * u / (1 + (shift - 1) * u)
model_time = time_scale * sigma
action_update = (sigma_next - sigma) * raw_velocity
```

模型时间乘1000不意味着积分增量也乘1000。实际模型时间还受既有 dtype 表示影响；报告不把理想公式误称为未舍入的 runtime 值。

### 3.2 严格 v2 实验控制

SQLite 账本由单控制器写入，worker 只生成结果包。v2 结果绑定完整 Candidate、campaign-instance 与已确认 job，而不仅是一个可重复使用的名字。

入账同时核验工作负载、动作域、质量 scope／通道／阈值、GPU 和完整身份。旧 v1 实验仍作为历史证据保留，不能事后补填 v2 身份冒充新验收。

调度层完成不等于结果通过：缺包、不完整覆盖、未知质量和失败质量都不能胜出。Bookend 漂移检查与质量检查同样约束参考点。

一次真实集成问题也由此暴露：累计 graph replay 计数被误当成 backend 身份变化。修复后，动态 graph／native enqueue 计数完整保存在执行证据中；engine、graph mode、native implementation 与已安装 module coverage 仍做静态身份核验，没有关闭身份检查来“救”实验。

### 3.3 FastWAM graph 的所有权与安全边界

Graph 默认关闭，只捕获 raw-velocity body；VAE、每请求 video KV prefill、采样路由和积分均在图外。

- 每请求首次 evaluation 刷新条件，即使输入指针复用；latent/time 每次 evaluation 刷新。
- 单几何契约包含 tensor tree、shape、stride、dtype、device。变化直接拒绝，不静默 recapture 或 eager fallback。
- 独立 stream 串行管理工作；eager caller streams 也链接依赖并在 close 前 drain，避免独立 velocity 尚在执行时释放内部 timestep／模型存储。
- 使用 raw `CUDAGraph.capture_begin/end` 保留 global capture safety，避开 Torch2.11 convenience context 的隐式全局 allocator cache flush。
- 输出 scratch 的借用生命周期、失败时 graph owner 保留及 reset 重试由契约测试覆盖。首次 capture 仍要求进程 CUDA quiescent；实例锁不能协调任意外部 GPU 线程。
- `backend="torch"` 显式标识共同 pipeline 的 Torch 路径；部分原生 graph 要求 cached Linear，legacy+graph 提前拒绝。

## 4. 数据流和成本分析：哪些优化真的减少了工作

### 4.1 π0.5：prefix 与反复 action evaluation 不能混为一谈

一组实际 fixture 的 prefix 为911行，其中768行 vision、143行 prompt/state。独立诊断观察到 prefix 区间约24 ms，每次 Action Expert 区间约1.65–1.67 ms。

对应源码／shape 账本中，17个 prefix FFN 在 `M=911, K=2048, N=16384` 下约3.11794 TFLOPs；每次 Action Expert 的 projections/FFNs 约31.1427 GFLOPs，attention products 约7.08526 GFLOPs。

因此减少 NFE 只减少重复 action evaluations，不能消除 prefix 成本。Post-ViT pool2 将 prefix 行数911降到335，却不减少已经完成的 ViT；该候选虽然更快，数值误差仍超过预算而被拒绝。

这些是算术／逻辑权重账本，不是实测 HBM 流量，不能据此独自判定 compute-bound 或 memory-bound。

### 4.2 FastWAM：缓存规划和 CUDA Graph 主要改变提交路径

Legacy BF16 Linear 每次 enqueue 重建设定并查询多个 heuristic；cached 路径复用计划，减少重复 host 工作。但两个实现的 workspace 预算不同，不能把整个延迟差全部归因于某一项 host planning 开销。

**Euler2／shift5、实际2NFE 的独立诊断 job `308460`** 观察到每请求920个 Linear 调用、约1.22055 TFLOPs；cached 路径中919个为 native、另一个为 Torch proprio。约14.0704 GB 的逻辑权重访问量也仅属于这份两步账本，不是 HBM 实测字节数，更不是后文 Euler20 的固定请求成本。

Graph replay 会绕过 Python hook 和 CPU enqueue：cached Euler20 的 steady host enqueue 只有305，但20次 raw graph replay 仍包含6140次逻辑 native Linear 调用。**不是只剩305次 GPU Linear，也不是零 FLOPs。** Kernel 数、逻辑算子数和 CPU 调用数必须分开记录。

## 5. π0 与 π0.5 的实测筛选

### 5.1 π0：代理通过，但没有速度收益

Job `307979` 在两步配置下记录10个样本／点：

| 点 | 实际 NFE | Median ms | 结论 |
|---|---:|---:|---|
| Euler2 | 2 | 26.1060 | 保留对照 |
| ProbeFlow2 | 2 | 26.7000 | public7 L2≈0.000375709、cos≈0.999999945，通过该次代理但更慢 |
| Euler2 bookend | 2 | 26.3815 | 对照 |

该实验没有证明两步策略替代默认策略的质量，也没有证明 ProbeFlow 在所有模型上能加速。

### 5.2 π0.5：多请求推翻了单输入结论

早期单输入上 Euler7 曾通过数值代理；严格 v2 job `308244` 加入两组新固定噪声与 RGB brightness+8，保持 prompt/state 和几何不变。每点8个 warmup、40个总测量请求，即每 case 10次：

| 点 | 实际 NFE | Median ms | 最坏 relative L2 | 最低 cosine | 结论 |
|---|---|---:|---:|---:|---|
| Euler10 reference | 10 | 41.6211 | 0 | ≈1 | 对照 |
| Euler7 | 7 | 36.8776 | 0.665398 | 0.767542 | 拒绝 |
| **Euler8** | **8** | **38.4342** | **0.0361504** | **0.999482** | **代理范围内选择** |
| PF ε=.012,max10 | 7/9 | 38.8245 | 0.129385 | 0.992178 | 拒绝 |
| PF ε=.016,max10 | 5/7 | 37.0234 | 0.129385 | 0.992178 | 拒绝 |

Bookend 漂移约+0.155%。结论是受控四请求范围内 Euler8 相对该轮 Euler10 降低7.66%中位延迟，而不是一次单输入筛选已经足够。

## 6. FastWAM：执行保持性与参数近似分别验收

### 6.1 Library graph 保持性

本节延迟均遵守第2.2节的 first-frame/action-only、预计算 T5 context 边界，不涵盖文本编码或未来视频 rollout。

Job `309097` 测试 Euler10；job `309138` 测试真实 dense restart；job `309187` 测试20-step budget。正式点使用3个额外整请求 warmup、10个样本；graph 点另做 original/noise/proprio/image/context/mask 六组控制，控制不混入正式计时。

| Job／配置 | 实际 NFE | Eager median ms | Graph median ms | 验收意义 |
|---|---:|---:|---:|---|
| 309097 Torch Euler10 | 10 | 389.6625 | 146.6673 | 同 engine 输出保持 |
| 309097 cached native Euler10 | 10 | 515.0250 | 158.2449 | 同 engine 输出保持 |
| 309138 Torch PF max10,ε=.0001，原输入 | 9 | 390.1538 | 140.6916 | 实际 dense restart；扰动还出现11NFE |
| 309138 cached PF max10,ε=.0001，原输入 | 9 | 489.7465 | 151.5507 | 同上 |
| 309187 Torch Euler20 | 20 | 804.6938 | 255.0711 | 原输入还精确匹配独立上游 Euler20 |
| 309187 cached native Euler20 | 20 | 971.2947 | 262.6851 | 同 engine 保持；不冒称上游 exact |

以上同 engine graph/eager 动作及对应刷新对照均精确一致。真实 CUDA 的无全局 cache flush、eager 在途资源清理回归两项通过。

但“graph 保持了 Euler10”不等于“Euler10 可替代 Euler20”：原输入 Torch Euler10 相对 Euler20 仍有约6.55% relative L2。ε=.008,max10 的 PF 在所有受测输入上实际只选择2NFE，不能把它叫 dense PF 验证；其约66 ms 的 graph 延迟也不能直接晋升为质量合格策略。

### 6.2 严格 v2 四请求参数筛选

Job `309300` 与 `309386` 使用同一冻结 `round014/baef55ff` 源码、同一四请求工作负载和同一门槛，但各自有新的 campaign-instance 与 confirmed job。四请求是原输入、两组 CPU-Torch 新噪声、归一化图像 brightness+.1；不是机器人任务集。

每点8 warmup、40个总测量请求。第一轮11点，严格入账9 validated／2 rejected，选择 Euler14，191.7414 ms。第二轮细查10步失败到14步通过之间的区间，12点入账7 validated／5 rejected：

| 第二轮 graph 点 | 实际 NFE | Median ms | P95 ms | 最坏 relative L2 | 最低 cosine | 结论 |
|---|---|---:|---:|---:|---:|---|
| Torch Euler20 reference | 20 | 251.6074 | 252.3159 | 0 | ≈1 | 对照 |
| Torch Euler11 | 11 | 159.8610 | 160.8566 | .0784776 | .997424 | 拒绝 |
| Torch Euler12 | 12 | 170.1411 | 171.7190 | .0560137 | .998713 | 拒绝 |
| **Torch Euler13** | **13** | **179.7011** | **180.0663** | **.0453660** | **.999179** | **代理范围内选择** |
| Torch Euler14 | 14 | 190.6152 | 190.9841 | .0372622 | .999441 | 通过但更慢 |
| Torch PF ε=.00006,max20 | 13/15/21 | 200.5839 | 262.2253 | .0461508 | .999132 | 通过但更慢 |
| Torch PF ε=.00007,max20 | 11/13/21 | 180.5588 | 262.7861 | .0702523 | .998064 | 拒绝 |
| Cached native Euler13 | 13 | 190.8152 | 191.5830 | .0428495 | .999268 | 通过但更慢 |
| Cached native Euler14 | 14 | 201.5966 | 202.1167 | .0340776 | .999540 | 通过但更慢 |

Cached native 的11/12步同样失败。第二轮 bookend 漂移+0.39614%；选择点相对该轮 reference 的中位延迟降低28.58%。PF 的21NFE请求体现在 P95 中，未用 requested max_steps 掩盖实际工作，也没有在看见结果后另造一个尾延迟约束。

该选择只适用于有限四请求代理；没有修改默认 sampler、自动降低生产步数或宣称全局最优。

## 7. Hy：从冻结算子定位到完整 gate 的反例

### 7.1 原问题与排除实验

Job `308140` 在 `official_local_fp32` 实验性 profile 下，原七 fixture 只有4/7通过；history/state/noise 各有4/2/6个 public 元素超差。

后续以精确复现官方 oracle 的 `308690` trace 为依据定位：输入、文本及特殊 token 可精确匹配，vision prefix 和 merger 输入已经不同。因此不能把所有残差归因于后面的排序或状态分支。

冻结实验 `309016` 证明，在受测输入与几何下：

- PatchConv camera6 与 batch18 均精确匹配官方；位置插值直接 BF16 与 FP32后转BF16也精确匹配。排除这两个局部假设，不盲改 dtype。
- PROJ1 官方逐相机视图精确匹配；native B3 布局有314260/1204224个元素不同，relative L2≈.002465559。Contiguous control 更差，不能把 `.contiguous()` 当成自动修复。

### 7.2 Attention backend 不是现成的精确替代

`309182`、`309409` 使用同一冻结 QKV，分别比较 camera6/batch18 和 Torch2.7.1/2.11。Norm1、QKV 及对应分组输出精确一致，但 Torch AUTO/Flash 与外部 FA2 存在局部差异：

| 强制 Torch SDPA backend | 相对外部 FA2 的不同元素数 | Maxabs | Relative L2 |
|---|---:|---:|---:|
| Flash | 559 / 4064256 | .00390625 | 2.94742066e-5 |
| Efficient | 853921 | .015625 | .000971597 |
| cuDNN | 905372 | .015625 | .001023550 |
| Math | 925829 | .015625 | .001049684 |

两 runtime 的所有16个 forced arms 均实际执行；15个共同输出逐位相等。没有一个 backend 达到外部 FA2 exact，AUTO/Flash 反而是最近的受测选项。

独立 profiler 观测到 AUTO 的实际 Flash aten op 与 CUDA `flash_fwd_kernel`，没有仅凭 API 名称推断 kernel。原摘要混入 GPU `record_function` annotation 的分类错误已通过回归修复；旧产物未重写，真实 kernel 与注释分开计数。

这些是重建 block0 输入上的算子控制，不是已记录 block0 oracle，也不能据一个559元素的局部差异量化完整 ViT 的误差放大。本轮不强切 attention，也不把旧 FA2 二进制导入不兼容的新 Torch runtime。

### 7.3 Native B1 假设先被拒绝

`309625` 将 native B3 改为逐相机 B1 调用，但保留实际 singleton leading stride225792。它在冻结 EXACT 前置门仍有314261个元素不同，正确退出1:0；没有加载任何完整模型。

Torch2.11 的 `should_fold` 检查 leading stride 等式时并不忽略 singleton 维：官方 leading stride196满足14×14，native225792不满足。于是下一候选仅去掉这个 singleton 维，保留 HWD 非连续视图和原 offset，Linear 后恢复 B1，不复制输入或改成2D+bias融合。

### 7.4 HWD 原型：冻结 exact，但完整策略仍失败

`309681` 的 example-only HWD 原型先通过冻结 EXACT：全部1204224元素精确匹配。实际 lowering 为3个 `mm` 加3个独立 `add_`；保留的 B1/B3 控制仍失败。

随后14个独立完整模型进程运行完毕。所有 baseline 精确复现旧 `308140` 的输入／oracle／配置／runtime／checkpoint／动作；源码路径差异明确记录。所有 candidate graph/eager 和 replay 精确相等，实际10NFE，清理通过。

| Fixture | Baseline 超差元素数 | HWD candidate 超差元素数 | Candidate gate |
|---|---:|---:|---|
| baseline | 0 | 0 | 通过 |
| episode_start | 0 | 0 | 通过 |
| current | 0 | 0 | 通过 |
| history | 4 | 0 | 通过 |
| prompt | 0 | 0 | 通过 |
| state | 2 | **3** | **拒绝** |
| noise | 6 | 0 | 通过 |

以上超差均在 public20 中。State 的三个失败位于零起始 `(step, channel)`：

| 坐标 | 绝对误差 | 原门槛允许值 |
|---|---:|---:|
| (14,9) | .01342773 | .01331171 |
| (15,9) | .01663208 | .01191901 |
| (35,9) | .01406860 | .01136251 |

State cosine≈.99998731、RMSE由约.00217262降至.00163435，但 maxabs 变大；几个原本通过案例的 RMSE 也增加。因此4/7→6/7只能称作部分因果消融改善，不能称为单调精度提升或完整修复。

**最终处置：保留诊断原型，拒绝作为完整七 fixture 修复推广。`309681` 及旧 `308140`、`309625` 均保持真实 FAILED 状态。**

## 8. 验证、可复核性与已知缺口

### 8.1 实际验证证据

- 本轮 fold 实现的专项 CPU 回归：45 passed。
- 所选回归与独立暂存源码导出：均为614 passed、41 subtests passed、2个明确 module skip。
- 两个 module skip 分别涉及本地原生扩展依赖和需要显式启用的 allocated CUDA 测试，不冒充 GPU 通过。
- 实际 A800 job `309097` 中，graph cache-flush 与 eager 异步资源清理回归2项通过。
- 裸全仓 `pytest` 仍退出2：20个可选／原生依赖收集错误、25 skipped，涉及 CUDA/native 扩展、可选模型／工具依赖及其他子项目环境。**没有全仓绿色结论。**
- 报告只整理已经记录并复核的证据；没有为了撰写报告重跑 GPU、补算机器人质量或将不同快照追认为同一实验。

### 8.2 已知缺口

1. 没有真实机器人或代表性任务成功率验收；四请求代理不是 held-out 数据集。
2. Hy state 剩余残差与已知 pre-merger ViT 差异尚未完整定位，不能通过修改容差绕过。
3. FastWAM 旧 profiler/pilot 对新 library-graph 参数的提前拒绝仍待补齐；库级验证使用新 validator，不把旧包装器当成受支持入口。
4. 非单位 attention head scale 的 CPU 解析 fixture，以及直接驱动 `main(argv)` 的完整 fold 模式传递测试仍是已记录的非阻断测试缺口。
5. 固定几何 graph、受测 checkpoint、runtime 与 A800 之外的组合没有自动获得本轮资格。
6. 路径／stat、源码快照和完整身份绑定不是内容哈希证明；本轮遵守不新增内容哈希工作的约束，依赖实验期间快照与产物不变。

## 9. 后续优先级

以下是建议，不是已执行结果或已批准的新默认：

1. **Hy：先定位剩余误差，再改数学。** 对 state 的失败链路和 pre-merger vision/prefix 边界继续做固定输入对照，避免“再试一种 cast/backend”替代因果证据。
2. **FastWAM：先扩大质量覆盖，再扩大搜索。** 在代表性任务／更多几何与输入上验证当前候选；保留完整实际 NFE、尾延迟与质量失败，不继续围绕四个 case 无限精调。
3. **π0.5：做独立泛化验收。** 当前 Euler8 是受控代理选择；新增噪声曾推翻 Euler7，说明后续必须使用未参与选择的数据。
4. **执行框架：补齐环境与接口保护。** 逐项明确全仓收集失败所需依赖与验收环境，补齐已记录的旧 wrapper／模式链路测试，不把 skip 数下降当成模型质量改善。

## 10. 证据索引与复核入口

原始权重、输入、trace、完整动作数组、环境和本地账本不随报告公开上传。以下链接固定到已发布 Git 提交，方便区分实现和结果记录；它们不是原始私有产物下载入口。

| 内容 | 入口 |
|---|---|
| 逐轮实测、失败与范围声明 | [实测记录，固定到63676648](https://github.com/20261023/FlashRT/blob/636766482d77993f968f0850c9c72e6f4eec0258/docs/vla_autotune_experiments.md) |
| 执行接口与模型／动作域契约 | [执行层文档](https://github.com/20261023/FlashRT/blob/636766482d77993f968f0850c9c72e6f4eec0258/docs/vla_backend.md) |
| v2 控制器、账本与质量入账 | [Autotune 文档](https://github.com/20261023/FlashRT/blob/636766482d77993f968f0850c9c72e6f4eec0258/docs/vla_autotune.md) |
| 验证器与 example 范围 | [Examples README](https://github.com/20261023/FlashRT/blob/636766482d77993f968f0850c9c72e6f4eec0258/examples/vla_autotune/README.md) |
| HWD 实验实际源码 | [97218269 的 PROJ1 原型](https://github.com/20261023/FlashRT/blob/97218269348a31467a96fcb11612e6f9244bb5b2/examples/vla_autotune/hy_trace/proj1_camera_ablation.py) |

关键实验索引：π0 `307979`；π0.5 严格 v2 `308244`；FastWAM library graph `309097`、dense `309138`、Euler20 `309187`、v2搜索 `309300/309386`；Hy 原 gate `308140`、accepted trace `308690`、冻结算子 `309016`、attention控制 `309182/309409`、被拒绝B1前置门 `309625`、HWD完整消融 `309681`。

### 一句话结论

**本轮验证了“受约束的 agentic 实验闭环”能够筛出受测数值代理范围内更快的配置，也能够诚实拒绝更快但越界或仅部分修复的候选；尚未证明这些配置具有通用机器人任务质量。**
