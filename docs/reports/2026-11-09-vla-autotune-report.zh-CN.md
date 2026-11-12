# FlashRT 多模型 VLA Agentic Autotune：实现、实测与数值一致性报告

**迭代标识：**2026-11-09 轮次，包含该轮后续排队完成的实验。

**覆盖模型：**π0、π0.5、FastWAM、Hy-Embodied-0.5-VLA。

**报告依据：**主体截至代码与结果记录提交 `63676648`；第11节持续补充后续实测证据。关键实验使用各自冻结的源码快照，而非事后统一替换版本。

**报告性质：**工程实验总结，不是模型排行榜、机器人任务成功率报告或全硬件认证。

**研究过程：**另见[现象、假设、验证与自我修正记录](2026-11-12-vla-autotune-research-process.zh-CN.md)，按轮次解释为什么做实验、哪些猜测被否定、证据如何改变下一步，而不只列最终结果。

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

## 11. 后续证据更新：扩大范围后不再接受粗步数候选

报告上传后迭代继续，没有将文档交付当作实验终点。

### 11.1 FastWAM：新增8案例拒绝 Euler13／14／16

Job `313779` 保持冻结实现、原 L2/cosine 数值预算，新增4组噪声与 image/proprio/context/mask 控制。每点80次请求、每 case 10次；候选预先声明，不在看到新数据后继续调参。

| Torch graph 配置 | Median ms | 最坏 relative L2 | 最低 cosine | 新范围处置 |
|---|---:|---:|---:|---|
| Euler20 reference | 249.4247 | 0 | ≈1 | 保留参考 |
| Euler13 | 178.7498 | .194629 | .980923 | 拒绝 |
| Euler14 | 189.9475 | .117659 | .993055 | 拒绝 |
| Euler16 | 209.1321 | .179380 | .983885 | 拒绝 |

三者仅在直接扰动预计算 T5 context 的案例中失败，其余7例通过。该扰动也改变原来为零的 padding 向量，属于**分布外数值压力测试，不是自然语言／机器人任务样本**。不能据此声称真实任务失败；但它明确阻止将原四案例 Euler13 选择当成通用数值质量保证，也说明增加步数不保证误差单调降低。

独立原始数组审计后严格入账2 validated／3 rejected，bookend 漂移+0.20146%。新增范围没有合格的加速候选，保留 Euler20；没有改阈值或围绕这些新案例继续调参。原四案例实验保持原历史结论，不回填或篡改。

### 11.2 Hy：剩余误差不能全部归到 vision/KV

Job `313766` 精确复现 HWD candidate 的 baseline/state 动作、逐步 raw velocity 与积分，并验证实际 time embedding 和 state embedding。冻结相同官方 latent/time 后替换成官方 KV，仍留下约.0031–.0054 relative L2 的 raw-velocity 残差；所有20个官方-KV输出还与旧诊断的对应注入输出逐位一致。

这定位了下游仍有残差，而不是确定某个 Expert 算子就是根因。它没有执行新的七 fixture gate，也没有将历史6/7失败改成成功。继续工作将先冻结核查 suffix／首层 query 边界，不盲改 attention、cast 或容差。

后续详细证据见[更新后的实测记录](https://github.com/20261023/FlashRT/blob/1c1a2fa9/docs/vla_autotune_experiments.md)。诊断实现 `2959bd87` 的所选回归及独立导出为651 passed、41 subtests passed、2个明确 module skip；裸全仓仍为20个收集错误／25 skipped，不是全仓绿色。

### 11.3 Hy：存在局部顺序效应，但全 Expert 恢复原序不是充分修复

- `313996`：step0 的 suffix、time／RoPE 和输入 RMSNorm 均 EXACT。分开／融合 QKV 仅9个 BF16 值不同（Q 0、K 5、V 4），两臂在同一 runtime 的 query-norm 相同；旧 runtime 精确复现记录，新 runtime 有 `1.43e-6` 量级 FP32 微差。没有据此改 cast 或 query 投影。
- `314150`：真实官方 attention 输出及 probabilities 能精确重放；原生匹配布局和连续 KV 原顺序均 EXACT。只同步改变 prefix K/V/mask 顺序后，baseline 每步2–20个、state 每步3–71个 BF16 context 值不同。这证明局部顺序效应，不等于量化最终动作误差份额。
- `314305`：两项新 EXACT 前置门、完整 native trace、每例全部50个旧 sorted frozen tensors、mask eligibility、自回放和 cleanup 均经原始数组复核。全32层 Expert 恢复原序后，固定官方轨迹上的残差没有一致改善：

| case／prefix KV | sorted 顺序 raw-velocity relative L2 | 恢复原序 |
|---|---:|---:|
| baseline／native | .00422149–.00696753 | .00406991–.00700935 |
| baseline／official | .00329106–.00537082 | .00340313–.00548461 |
| state／native | .00399431–.00657527 | .00406906–.00669422 |
| state／official | .00313123–.00510265 | .00325715–.00515597 |

因此不能把局部 attention 顺序差异提升为已经找到主因或已经修复。adapter 本身不改变 Q 参数，但前层变化会传递到后层 hidden/Q；这不是32层都固定 Q 的实验。每例640次 eager attention 调用也不代表640 NFE 或640个 GPU kernel。

本项没有产生新的 free-running 原序策略动作、七 fixture gate 或机器人成功率结果；Hy 仍为历史6/7失败。详细证据固定在[后续实测记录 a6543218](https://github.com/20261023/FlashRT/blob/a6543218/docs/vla_autotune_experiments.md)。后续转向记录实际 MLP 边界，而不是直接更改生产顺序或精度。

### 11.4 Hy：实际32层边界区分局部误差与传播误差

`314364` 先记录 Expert0 MLP 的真实输入及 gate/up/activation/output：两例十步的原 API、分开与融合 GU 均 EXACT，两个 runtime 每例90个消费者张量也逐位一致。该局部假设被否定，但不能外推其余31层。

随后 `314485` 记录全部32层的实际 hidden、第一 residual、MLP 输出和最终 norm 输入。每例970个有序事件、320条 residual 到下一实际输入的链均 EXACT。全部20步 native H0 通过运行时 EXACT 准入；H0 重建数组未另行导出，因此不把该准入冒充独立权重重算。独立原始数组审计确认：2.11传播模式的20个最终 raw velocities 全部逐位复现 `313766` 官方-KV条件，逐层重放确实对应已有受测 native 路径。

| case | 固定官方每层输入的 layer31 hidden relative L2 | native hidden 逐层传播 |
|---|---:|---:|
| baseline | .0000517155–.0004999602 | .0023364851–.0035240699 |
| state | 0–.0003188937 | .0024410601–.0035831674 |

局部 residual／output 差异从第0层已经出现，不能只归因于某个晚层；传播后的误差更大，但这些 hidden 范数不是动作容差，也不能相加成“解释误差百分比”。固定官方 H32 的输出头对照在旧 runtime 为20/20 EXACT，在2.11为19/20：state step9仅两个 BF16 velocity 值不同，maxabs `.001220703125`、relative L2 `.0000340360`，该微差仍保留在结果中。

本项未改变历史6/7失败。详细证据见[逐层实测记录159f7368](https://github.com/20261023/FlashRT/blob/159f7368/docs/vla_autotune_experiments.md)。下一步仅干预 QKV 投影几何，并先要求不变 fused 对照的全部1300个张量精确复现；尚不把该待测干预当成修复或生产默认。

### 11.5 Hy：QKV 投影几何单独改变仍不是充分修复

`314589` 完成四个零模型消费者后，完整18文件已回传并独立复核：四组 fused 对照各1300个张量 EXACT，native-fused 的20个速度 EXACT 复现已有基线，40个固定-Gold输出头负对照不变。只将 Expert QKV 改为分开的三个 GEMM，其他运算保持固定，2.11的结果仍为：

| case | fused raw-velocity relative L2 | separate Q/K/V |
|---|---:|---:|
| baseline | .00329106–.00537082 | .00327744–.00533305 |
| state | .00313123–.00510265 | .00315046–.00503325 |

baseline仅6/10步、state仅7/10步 L2 改善，其余步反而变差；没有一步变为速度 EXACT，也没有任一 local layer output 在十步上全部 EXACT。因此，这不是七 fixture 通过或默认配置改进。旧2.7对照同样保留非零残差，完整数值见[实测记录e4103fe1](https://github.com/20261023/FlashRT/blob/e4103fe1/docs/vla_autotune_experiments.md)。

后续若检验顺序与投影的交互，应在已拆分投影条件下先精确复现全部1300个 sorted 基线，再只恢复 prefix 顺序；尚无该组合结果。当前 Hy 仍为历史6/7失败，没有改容差、生产数学或默认值。

### 11.6 Hy：旧 runtime 的投影＋顺序联合条件精确，新 runtime 仍有差异

`315024` 已完成，并在暂停／恢复后取回全部27文件独立审计：四组 sorted 控制各1300个张量 EXACT，80个零权重 Gold context 前置门及40个固定-Gold head 负对照均保持 EXACT。separate QKV 固定后，只恢复 prefix K/V/mask 原序，得到：

| runtime／case | separate＋sorted raw-velocity relative L2 | separate＋original |
|---|---:|---:|
| 2.7／baseline | .00299707–.00529732 | **0，10/10 EXACT** |
| 2.7／state | .00297147–.00517791 | **0，10/10 EXACT** |
| 2.11／baseline | .00327744–.00533305 | .00307602–.00494808 |
| 2.11／state | .00315046–.00503325 | .00316721–.00527045 |

2.7下不只是最终速度：全部32层、两种 hidden 输入条件、两例十步的每个 residual 与 output 都逐位 EXACT Gold。这定位了真实联合效应，不能再用“两项单独都不够”推断组合效应为零。但它仍是**官方 KV／固定外部轨迹下的 Expert 条件等价**，没有原生 vision KV、free-running 策略或机器人质量验收。

实际 Q 布局在所有 runtime／arm 相同，不能仅靠“布局不同”解释版本差异；也不能因此保证任意布局无影响。2.11 的最早分歧因 step／layer 而异，部分位置 residual EXACT、FFN output 不同，后续将固定捕获的 RMS 输入和权重，拆查阶段首个差异，不直接改 cast、归约或 `rsqrt`。

详细证据见[联合条件实测记录1d4345b9](https://github.com/20261023/FlashRT/blob/1d4345b9/docs/vla_autotune_experiments.md)。[PyTorch2.7](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html)及[2.11官方数值说明](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)均不承诺跨版本逐位一致；当前不把该兼容性差异直接叫作库 bug。生产 runtime／默认未更换，历史6/7失败及原门槛保持不变。

### 11.7 Hy：固定相同输入后，RMS 首个分歧在 FP32 mean

`315256` 已完成且完整原始数组独立审计。预先固定 baseline/state 各 `(step0,layer1)`、`(step2,layer0)`，两 runtime 的8次 block 执行都精确复现各自 `315024` residual/output oracle。32个 own RMS 端点和16个 fixed-old native 端点也逐位重放实际 RMS，并非只核对阶段实现的自回放。

| 固定旧输入下的观测 | 独立审计结果 |
|---|---|
| FP32 转换、平方 | 16/16 EXACT |
| 首个分歧阶段 | 16/16 为 `square.mean(-1, keepdim=True)` |
| 128项跨 runtime 阶段比较 | 82项不同、46项 EXACT |
| FP32 Q/K weighted 端点 | 8/8 不同 |
| BF16 weighted 端点 | 7/8 EXACT，1项不同 |

唯一 BF16 端点差异是 baseline `(step2,layer0)` postnorm：输入／residual 仍 EXACT，17个 variance 值不同，5个 cast-back 值不同，最终4个 weighted BF16 值不同，maxabs `.0078125`。这在 FFN 之前直接锚定了归一化输出差异，不能把后续 FFN 差异都归到 GU 融合。另一处 own postnorm 输入已受上游 residual 影响；fixed-old 臂排除了这个输入混杂。

**首个分歧不等于因果充分性。** 尚未证明相同 variance 下 `rsqrt`／下游算术全部一致，也没有量化其对全部32层或最终策略残差的解释份额。下一步只交接旧中间量，先复现已有2.11基线，再检查下游；没有更换生产归约、精度、runtime 或阈值。历史6/7失败不变。详细证据固定在[实测记录52e03612](https://github.com/20261023/FlashRT/blob/52e03612/docs/vla_autotune_experiments.md)。

### 11.8 Hy：旧 variance 对受测局部 RMS 下游已具充分性

`315323` 复用上述16组固定旧输入，先要求144个2.11基线数组逐位复现 `315256`，再分别交接旧 variance、variance＋epsilon、inverse、scaled 和 cast-back。交接使用**当前 native 中间量布局**；旧中间量 live strides 没有记录，不伪造旧布局一致性。

完整784张量独立审计通过：144个历史基线、320个自值交接控制 EXACT；320项旧值交接阶段比较全部 EXACT。五种交接各16/16恢复旧 weighted 端点，且每种交接的**整个受测下游 suffix**均 EXACT。因此在这16组输入／当前布局下，旧 variance 已足以消除局部 RMS 差异，不只是让 BF16 末端恰好舍入相同。

这轮没有执行 live block，也未读取 checkpoint 张量或运行新旧模型；不能据局部充分性宣称完整 Expert／策略已经修复。后续四个实际 block 的干预会先核对每次实际 norm 输入、权重与布局 EXACT，任何不匹配都停止该 probe，不能偷偷使用无关旧 variance。历史6/7失败、生产数学／精度／runtime／默认及原门槛均不变。详细证据见[实测记录df83cb94](https://github.com/20261023/FlashRT/blob/df83cb94/docs/vla_autotune_experiments.md)。

### 11.9 Hy：受保护的旧 variance 交接恢复四个真实 block

`315395` 完成后，全部144个原始张量独立审计通过：56个 native 基线角色 EXACT 复现 `315256`；干预路径16次 norm 的实际输入、权重及 live layout 全部匹配旧捕获，零输入拒绝；16个未干预 native 观测端点复现已有 fixed-old native 结果。仅将旧 variance 交给原下游算术后，四个真实 block 的 residual 和 output 均 EXACT 旧 oracle。

| Probe | 原生 residual／output 不同元素数 | 交接后 |
|---|---:|---|
| baseline.s0.l1 | 1／2 | 全部 EXACT |
| baseline.s2.l0 | 0／102 | 全部 EXACT |
| state.s0.l1 | 0／0 | 保持 EXACT |
| state.s2.l0 | 0／0 | 保持 EXACT |

因此“两处原有差异被恢复”和“两处原已相同的控制不变”应分开描述，不能把四处都叫新增修复。实验没有粘贴 Gold block 输出、改 GU／attention，或在输入变化时套用旧 variance；只读取22个唯一 BF16 权重张量（44次读取事件），没有构造完整模型或替换生产 norm。

范围仍是四个 teacher-input／Gold KV block，不是32层传播、全部 step、native vision KV、free-running 策略或机器人资格。下一步核查实际 runtime 的归约源码／安装头文件，再决定最小 dispatch 观测；不据四个点盲写兼容 mean。历史6/7失败和原阈值／生产默认保持不变。详细证据见[实测记录71406637](https://github.com/20261023/FlashRT/blob/71406637/docs/vla_autotune_experiments.md)。

### 11.10 Hy：实际 mean launch 相同，不等于内部加法树相同

精确版本源码与实际安装头文件均发现：CUDA warp 内 shuffle 加法顺序从升序 offset 改为降序，128宽输入的向量化门槛从 `>128` 改为 `>=128`；FP32 `MeanOps` 仍为 FP32 求和后乘 factor，新版16位输入的8元素向量化分支不适用于这里的 FP32 平方张量。这是机制证据，不是库 bug 或实际内部分支证明。

首轮 `315596` 因解析器把同名 CPU 范围与 GPU 投影重复计数而 FAILED；仅留下首个 trace，没有 final raw/report，native 进程未运行。失败及原快照保留。实际格式回归 RED→GREEN 后，新 round 的 `315622` 完成，全部37文件、32个完整 trace、128个数组独立审计通过：profile 前后 mean 精确复现各自历史值，FP32转换／平方 EXACT旧值。

| 实际 mean 输入 | 平方张量 live stride | 两 runtime 的 grid／block |
|---|---|---|
| [1,41,1024] | [41984,1024,1] | [3,1,1]／[32,16,1] |
| Q [1,16,41,128] | [2048,128,2048,1] | [41,1,1]／[32,16,1] |
| K [1,4,41,128] | [512,128,512,1] | [11,1,1]／[32,16,1] |

32个窗口的平方输入均为 FP32、storage offset0、16字节对齐；两版实际 kernel 完整符号也相同，为 FP32 `reduce_kernel<512,1,…,4,4>`。这确认了直接输入／布局／launch 预测，**没有证明 runtime `ReduceConfig` 的向量化字段或编译后的 shuffle 路径相同**，更不否定此前已测得的不同 mean 数组。

下一步只测试从实际输入计算的显式 FP32 加法树，不能把保存的旧 variance 当答案；没有启用生产兼容归约或将 profiler 时长当作候选延迟。完整策略、机器人及历史6/7失败边界不变。详细证据见[launch 实测f04a95e3](https://github.com/20261023/FlashRT/blob/f04a95e3/docs/vla_autotune_experiments.md)。

### 11.11 Hy：输入驱动的显式加法树重现两版 mean 端点

`316521` 直接接收实际 FP32 平方输入，预先固定四种“标量／向量4分组 × 升序／降序 shuffle”模型，不读取保存的 variance 作为答案，也不调用 native `sum/mean`。全部96个数组、128项旧／新对照独立审计通过，64个 GPU 树输出均逐位等于独立 NumPy FP32、逐 lane／component 索引的重算。

| 显式 FP32 树 | EXACT旧 mean | EXACT新 mean |
|---|---:|---:|
| 标量分组＋升序 | 8/16 | 0/16 |
| 标量分组＋降序 | 0/16 | 0/16 |
| 向量4分组＋升序 | 8/16 | 0/16 |
| 向量4分组＋降序 | 0/16 | 16/16 |

按旧源码为128宽选择标量＋升序、1024宽选择向量＋升序，可由实际输入恢复16/16旧 mean；新式向量＋降序恢复16/16新 mean。这比交接保存的旧 variance 更进一步，但仍不是对内部编译分支的直接观测，也不是任意形状／数据／布局或完整策略的保证。所有否定交叉组合都保留，没有据结果继续调参。

该模型刻意采用多个 eager FP32 加法 kernel，只作机制诊断，不是加速候选。后续完整 Expert 原型必须由 LIVE input／weight 计算 RMS，并先通过全部 native 历史对照；最终 norm／head 参与干预，不能宣称 Gold-head 是不变负对照。生产默认、原质量阈值和历史6/7失败不变。详细证据见[源码树实测398bdecb](https://github.com/20261023/FlashRT/blob/398bdecb/docs/vla_autotune_experiments.md)。

### 11.12 Hy：32层条件 Expert 的跨 runtime 差异已闭合

`316722` 保持 separate QKV、prefix 原序、Gold KV 和固定官方外部轨迹，先要求全部2600个未改动 native 对照张量精确复现历史。随后只用每次 LIVE input／checkpoint weight 计算 source-old RMS，没有注入保存的 norm 输入、variance 或输出。

| case | native 对照与旧2.7 EXACT | live-tree 与旧2.7 EXACT | 传播 raw velocity relative L2 |
|---|---:|---:|---|
| baseline | 509/1300 | **1300/1300** | .00307602–.00494808 → **0** |
| state | 510/1300 | **1300/1300** | .00316721–.00527045 → **0** |

全部5200个原始数组及旧 oracle 对照均独立审计。原型两例的32层、两种 hidden 输入模式、十步的每个 residual／output 都 EXACT；20个传播速度和20个固定-Gold head 速度也全部 EXACT。最终 norm／head 参与干预，不能把它当作不变负对照；state 的 control head 为9/10 EXACT，原型为10/10。额外 BF16→int16 比较连 signed zero 的位型也一致，未添加内容哈希。

每臂观测5160次 norm，作用域／H0／attention-order 清理通过；仅加载363个唯一 checkpoint key／臂，726次读取事件，没有构造完整模型或构建新 kernel。登录节点只读加载预检的 SSH 超时没有被记为通过；相同来源／输入门在 allocation 内先于 owner／weights 执行，实际完成后再由原始数组审计确认。

**这是联合固定投影／顺序条件下的完整 Expert 等价，不是完整 native policy 修复。** Vision/prefill KV 在该实验中来自 Gold，latent/time 来自固定外部轨迹；native vision KV、free-running、原七 fixture 和机器人资格尚未重测。多-kernel 原型不是加速候选，历史6/7失败、生产默认和门限保持不变。下一轮只在实际 `evaluate_velocity` 阶段接入，VLM/prefill、积分和原质量门均不改。详细证据见[完整 Expert 实测200e6560](https://github.com/20261023/FlashRT/blob/200e6560/docs/vla_autotune_experiments.md)。

### 11.13 Hy：完整 native-policy 候选仍6/7，拒绝晋升

`316975` 将 separate QKV、prefix attention 列逆序和 LIVE source-old RMS 三项改动限制在实际 `evaluate_velocity` 内，最终 norm／head 包含在内；vision、VLM/prefill、native KV、Euler10 和固定 noise 不变。先让七例 HWD 控制各运行两次并精确复现历史 actions／6/7，再让候选各运行两次。没有注入 Gold KV、latent、norm 或 velocity。

原作业曾因 SSH 观测失败而终态未知，但没有被判为 Slurm 失败或重复提交；退避恢复后确认同一回执／源码的作业 COMPLETED `0:0`。完整60文件、739398628字节回传，1680个请求张量、28次请求／280 NFE、两次回放、native-prefix／conditions 不变性、实际129 RMS／32 block／32 attention 每 NFE、Euler 链及清理均独立审计通过。

**机械验证通过，数值候选仍被拒绝：padded40×32 和 public40×20 都是 control6/7、candidate6/7，state 仍失败。** 原门保持逐元素 `abs(delta)<=.01+.01*abs(ref)` 且 cosine≥`.999`，public 切片不能替代 padded 门。

| state／padded40×32 | HWD control | Expert 三项改动 candidate |
|---|---:|---:|
| 超限元素数 | 3 | **4** |
| maxabs | .016632080078125 | .015106201171875 |
| RMSE | .0016343457521882086 | **.0017574000621361032** |
| cosine | .9999873074562833 | .9999853985877847 |

六个原本通过的 case 保持通过，但只有 episode_start／history／noise 的 RMSE 改善，baseline／current／prompt／state 变差。不能用更小的 state maxabs 救门限，也不能把 R32 的条件 Expert EXACT 外推成完整策略单调改善。七例历史 actions 都精确复现；baseline/state 另有各47个历史字段 EXACT，其余五例旧 NPZ 没有逐步轨迹，明确只作本轮新控制的双回放验证，没有伪造历史 velocity 证据。

CPU 边界复核进一步发现：baseline/state 与 Gold 的266738个 prefix embedding 差异全部在 image tokens（relative L2 `.01257768617059049`、maxabs `.75`），非图像 tokens、pad／modality masks EXACT；K/V 从 layer0 已不同。历史 native Merger 输入在进入 Merger 前也已有476775个值不同（relative L2 `.006573505861752344`、maxabs `16`）。R33 image-token 输出 EXACT该历史 native `vision.merged`，但**没有捕获 R33 实际 vision 输入，输出桥接不能冒充同一次输入捕获**。

因此下一轮先用相同 frozen 输入比较完整 Merger 的官方 per-camera API 和 native HWD 调用，并检查阶段端点，不直接归因于 VLM RMS、不扩大支持宽度或围绕失败 state 调参。历史失败、原阈值、生产默认／runtime／精度均保持，未形成机器人或加速资格。R33 实测与否定结论见[固定记录5f084d43](https://github.com/20261023/FlashRT/blob/5f084d43/docs/vla_autotune_experiments.md)。

### 11.14 Hy：完整 frozen Merger 跨版本相同，调用形式差异首现在 PROJ2

`317360`／`c990f271` 已完成，完整120个数组独立审计，8个阶段重建末端与各自实际API EXACT，旧runtime官方Gold锚点、新runtime native历史输出／R33输出桥均通过。只加载8个BF16权重，未构造模型、运行策略或测候选延迟。

固定同origin／form后，两runtime的全部14阶段与实际端点都EXACT，60对BF16位型含signed zero也一致。因此本轮没有发现受测完整Merger的跨版本差异。固定相同输入／runtime、比较两种调用形式，则PROJ2之前所有阶段包括activation均EXACT，首差异在PROJ2：Gold输入254值不同，历史native输入287值不同，maxabs均`.015625`。

这支持“最后Linear调用组织仍有局部数值影响”，而不是PROJ1局部EXACT就代表完整Merger等价。实际PROJ2输入为每camera `[1,7,7,2048]` 对native batched `[3,7,7,2048]`；源码预示flattened M49对M147，但具体GEMM算法／bias epilogue未观测。上游输入差异也仍在：同form换origin的merged relative L2约`.012578`，不能把PROJ2发现当作整个策略误差的解释比例。

下一步先从已获准activation孤立最后Linear，验证两个原端点再判别调用组织；没有新增七例通过或生产默认变更，历史6/7失败保持。研究过程中的M1—M4预测及其支持／否定边界见[过程记录第8节](2026-11-12-vla-autotune-research-process.zh-CN.md#8-r34在结果前写下下一轮如何判别原因)。

### 11.15 Hy：隔离PROJ2后，biased Linear的M49／M147效应被直接重现

`317555`／`32ea141e` 的30数组、14个CPU trace、4个历史端点和28项对照独立审计通过。在两个冻结activation来源上，per-camera B1／HWD／flat49三个有bias调用都EXACT官方端点；B3／flat147都EXACT native端点。实际CPU侧aten记录分别是3次M49 `addmm` 对1次M147 `addmm`，计算使用A800，但没有采CUDA算法／kernel时长。

两种显式BF16 `mm→add`均不匹配任一原端点，对official参考分别76773／77608个值不同、maxabs`.5`。保存输出的追加CPU比较还发现，这两种显式表达式彼此在两个来源上EXACT。因此这是**biased执行路径条件下**的行分组效应，不是“M不同就总会不同”的普遍规律，也没有证明某个具体cuBLAS epilogue原因。

已追加到[研究记录第8.7节](2026-11-12-vla-autotune-research-process.zh-CN.md)，保留第8.6节结果前七臂预测。下一步须先验证examples-only完整Merger中的实际端点，再重测单一PROJ2改动的native-policy；当前没有新七例通过、速度或机器人结论，原6/7失败和默认／门限保持。

### 11.16 Hy：单改PROJ2使state过门，但noise新失败，统一候选仍拒绝

`317665`／`a5078137` 完成并通过20前置＋1792请求数组、28请求／280NFE独立审计。完整Merger候选恢复受测official端点；两臂实际输入／activation、非视觉prefix、固定条件一致，Expert均为原生，所有回放与清理通过。七例control actions精确复现309681；baseline/state47字段各EXACT313766，七例旧60字段又全部EXACT R33 control。

原padded与public都仍6/7，但**失败从state迁移到noise**。state超限3→0、maxabs`.01663208→.0078125`，RMSE却`.00163435→.00171723`；noise超限0→6、maxabs`.01318359→.03903198`、RMSE`.00172436→.00297720`。四例RMSE改善、三例变差，没有七例统一通过，候选被拒绝，不放宽门限或按case挑实现。

当前真实vision输入复核仍见上游差异。Merger输出端点的附加向量分析在一个共享视觉context上验证了局部L2净抵消，但不是完整策略误差归因／比例。具体预测、失败迁移、公式证据及限制见[过程记录第8.8—8.9节](2026-11-12-vla-autotune-research-process.zh-CN.md)。下一步拟预注册一次E×P交互验证而非无限组合；目前无新默认、机器人、泛化或性能资格。

### 11.17 Hy：E×P交互非零，组合5/7，停止组合搜索转上游ViT

`317924`／`e97f22e7` 的20前置＋3584请求＋14交互数组、56请求／560NFE完成独立审计。三个已知格先完整复现R33/R36 raw再执行both；Eonly新增视觉字段只与当前base比较，没有补造R33历史输入。两臂作用域、固定条件／布局、固定P的prefix／KV不变量及所有回放／cleanup均通过。

原padded与public均base6／Eonly6／Ponly6／**both5**。both保持state通过，但noise仍6个超限，maxabs`.0421142578125`；新增history1个超限，maxabs`.0107421875`，即使其RMSE比另三格都小仍失败。组合统一候选拒绝，不调容差、不删case。

预注册的`I=(both-Ponly)-(Eonly-base)`由实际数组CPU FP64独立重算。七例首步velocity与最终actions交互均非零，前者每例747–797个非零元素、maxabs均`.03125`；后者1096–1119个、maxabs约`.00537109–.00695801`。共享首步条件下已出现非加性响应，但非零交互并未带来统一质量资格，也不识别E内三个子改动。

已执行结果前声明的停止规则：**结束E×P组合搜索，下一步做上游frozen ViT定位**。这不是独立泛化、机器人成功率或性能结果，不改生产默认／runtime／精度／sampler／原门。完整预测与结果后修订见[研究记录第8.10—8.11节](2026-11-12-vla-autotune-research-process.zh-CN.md)。
