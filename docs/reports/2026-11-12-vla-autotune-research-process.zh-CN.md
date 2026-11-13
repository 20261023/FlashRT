# VLA Autotune 研究过程记录：现象、假设、验证与自我修正

**迭代起点：**2026-11-09；**本记录整理日期：**2026-11-12。

**覆盖：**π0、π0.5、FastWAM，以及重点展开的 Hy 数值一致性定位。

**记录边界：**已核验结果更新至 R39。保留各轮结果前预测，包括 `e97f22e7` 的R37四格停止规则、`311c66d0` 的R38四块前缀与 `e058161a` 的R39实际context／投影切口；各轮结果另节追加，不改写先前假设。

**性质：**可审计的研究／工程实验记录，不是机器人成功率、模型排行榜或生产默认变更说明。

## 1. 为什么另写一份“过程记录”

[结果报告](2026-11-09-vla-autotune-report.zh-CN.md)回答“实现了什么、测到什么”；本文回答另外几个问题：

1. 当时看到的异常究竟是什么？
2. 有哪些相互竞争的原因解释，哪些只是猜测？
3. 怎样设计对照，才能区分这些原因，而不是只看最后误差变大或变小？
4. 哪些实验否定了先前想法，为什么下一轮改查另一个位置？
5. 已证实的局部机制，为什么没有自动变成完整策略的修复？

本文依据冻结计划、已发布实测记录及保留的原始数组审计整理。**“假设／分析”表示可检验的命题或实验动机，不等于已验证的原因。** 对早期轮次的衔接性解释是事后整理，不伪装成当时已有的逐条预测；R34 的判别条件则在结果前写下。

历史实验按作业与源码版本区分。今天知道了后来的结果，不会把早先的失败改写成成功，也不会把新控制数据冒称旧数据。

## 2. 本轮自我迭代实际遵循的闭环

```text
观察现象并确认测量可信
  → 列出候选解释及其不同预测
  → 冻结输入／源码／runtime／质量范围
  → 先复现未改动控制，再只改变待研究因素
  → 保留完整数组、实际执行次数和失败
  → 独立重算，判断支持了什么、否定了什么
  → 缩小结论范围，选择下一项能区分剩余解释的实验
```

“自我迭代”不是不断换参数直到通过，而是**依据反例修订下一步问题**。过程中的主要修正包括：

- 单输入通过后，主动增加请求覆盖；新请求否定旧候选时接受否定。
- 原生实现仍比 Torch 慢时，不把“native”当成胜出理由。
- 一个局部算子恢复 EXACT 后，回到完整策略验证，不直接宣布修复。
- 两项单独干预都不充分时，仍测试有机制依据的交互，而不是认定组合无效。
- 找到首个不同阶段后，继续做中间量交接，区分“首先发生差异”与“足以解释下游”。
- 保存答案的交接实验通过后，再要求新公式从 LIVE input 计算，避免把 oracle 当算法。
- 条件 Expert 全部 EXACT、真实 native policy 仍失败时，保留两者，转查上游输入混杂。

### 2.1 三种“通过”必须分开

| 层次 | 要回答的问题 | 不能替代什么 |
|---|---|---|
| 机械／执行验证 | 身份、输入、实际 NFE、回放、graph/eager、资源清理是否可信？ | 不能替代与参考输出的数值验收 |
| 数值质量代理 | 指定 case、输出域和阈值是否通过？ | 不能替代机器人任务成功率或未测范围的泛化 |
| 性能验证 | 在明确计时边界、同 allocation 对照下是否更快？ | 不能让数值失败点胜出，也不能由 profiler 时长冒充正式延迟 |

Hy 的原门始终是 padded `40×32` 每个元素满足 `abs(delta) <= .01 + .01*abs(reference)`，且 cosine≥`.999`；public `40×20` 另报。π0.5／FastWAM 的相应筛选使用明确范围内 relative L2≤`.05`、cosine≥`.999` 的工程代理。它们不是可互换的通用 VLA 门限。

## 3. 先校正优化方向：四模型早期实验的启示

### 3.1 π0：数值近似通过，不代表有加速

**现象：**`307979` 的 Euler2 和 ProbeFlow2 都实际执行2 NFE。ProbeFlow2 的 public7 relative L2约`.000375709`、cosine约`.999999945`，通过该轮代理。

**待检验想法：**共享采样／执行接缝可以工作，但换成 ProbeFlow 是否真的减少请求成本，必须实测，不能由算法名称推断。

**验证与结果：**同轮中位延迟 Euler2 `26.1060 ms`、ProbeFlow2 `26.7000 ms`、Euler2 bookend `26.3815 ms`。没有观察到 ProbeFlow2 加速，保留 Euler2 对照。

**认识修正：**这只是两步执行比较；既没有完成 π0 全参数搜索，也没有验证“两步替代默认策略”的机器人质量。

### 3.2 π0.5：单输入结论被新噪声推翻

**现象：**早期单输入 `308132` 的 Euler7 达到约`36.9639 ms`、relative L2约`2.56%`，看起来可保留；post-ViT pool2 更快，约`28.8652 ms`，却有约`33.21%`误差。

**分析／候选解释：**

- 减少重复 Expert evaluations 可以省时，但 prefix 成本不随 NFE 消失。
- 单输入的粗时间离散误差可能恰好较小，不能据此假设新噪声也稳定。
- Pool2 改变 prefix GEMM 行数，不会撤销已经执行的 ViT；速度变化与质量变化必须同时记录。

**区分性实验：**`308244` 扩为原输入、两组新固定噪声、RGB brightness+8；prompt/state 和几何保持不变，所有请求共用原数值预算。

**结果：**Euler7 最坏 relative L2变为`.665398`、最低 cosine`.767542`，拒绝；Euler8 为`.0361504`／`.999482`，中位`38.4342 ms`，相对同轮 Euler10 `41.6211 ms`降低`7.66%`。两个 PF 点也被新请求否定。

**认识修正：**扩大覆盖会改变选择；Euler8 只在这四请求范围内获选，不能把未测输入的稳定性写成已证实。没有为了保住更快的 Euler7 放宽门限。

### 3.3 FastWAM：先分清 host 提交与 GPU 数学工作

**现象：**`308343` 中 cached native 从 legacy 的`202.5082 ms`降至`154.7789 ms`，但 Torch 为`122.5195 ms`，仍更快。

**候选解释：**重复 descriptor／heuristic planning 可能制造 host 开销；另一个可能因素是实现／workspace 算法差异。仅比较总延迟还不能分摊二者的贡献。

**验证：**`308460` 核对两引擎同样920次逻辑 Linear 调用、约`1.22055 TFLOPs`；instrumented kernel duration 总和相近，却有不同跨度／间隙。Profiling 有扰动，因此不把这些 walltime 当正式延迟，也不把逻辑权重字节当实测 HBM 流量。

**下一实验：**`308611` 同时给两引擎 raw-velocity graph，而不是只给 native。Torch graph 中位`69.0292 ms`，cached graph `75.8655 ms`；两边均通过各自 graph/eager 数值保持控制。

**认识修正：**host 提交路径确实值得优化，但 native 不是预定赢家；workspace 不同，不能把全部收益精确归给 planning。Graph 绕过 Python enqueue 不等于 GPU Linear／FLOPs 消失。

后续库级 graph 验证覆盖了普通步数及真实 dense restart；再把“执行保持性”和“减少采样步数的近似质量”分开筛选。

### 3.4 FastWAM：四请求最优点没有通过新增压力范围

**原结果：**`309386` 在四请求范围选择 Torch graph Euler13，中位`179.7011 ms`，相对同轮 Euler20 `251.6074 ms`降低`28.58%`。

**待检验想法：**这只是有限代理选择，增加噪声、图像、proprio、context／mask 压力后未必保持。增加步数也未必对每个扰动单调降低误差。

**验证：**`313779` 预先声明 Euler13／14／16，在八个新增请求上测试，不见结果后继续调参。三个候选都在 `context-minus0p0625` 失败，最坏 relative L2分别约`.194629`／`.117659`／`.179380`；新增范围保留 Euler20。

**认识修正：**旧四请求结果没有被抹掉，但其适用范围不能外扩。这里直接扰动准备好的 T5 向量，属于数值压力，不是代表性自然语言任务；它反驳“普遍满足数值预算”，不能证明真实任务成功率下降。

## 4. Hy 第一阶段：先证明残差不是一个笼统的“精度问题”

### 4.1 从4/7失败开始，而不是从“BF16应该一致”开始

**现象：**`308140` 在实验性 official-local profile 下，原七 fixture 只有4/7通过；history／state／noise 失败。固定 noise 回放和执行机制通过，并没有使官方数值门自动通过。

**竞争假设：**输入预处理／状态投影不同，vision prefix 已不同，KV 误差传入 Expert，或者 Expert 在相同输入下仍有自身算术差异。

**验证：**`308690` 先精确复现旧官方 oracle 和 native 自身路径，再比较 prepared inputs 与边界；用官方 KV 替换 native KV，只作冻结条件实验。

**结果：**输入／文本／特殊 token 可匹配，vision embedding 和 Merger 输入已经不同；官方 KV 能减少但不能消除速度差异。

**解释更新：**既有上游输入混杂，也有待定位的下游差异。不能把最终误差全归给排序、状态分支或 vision 单一因素。

### 4.2 用相同输入排除 PatchConv batching 和位置插值 cast

**问题：**vision 首部的 batch 组织、BF16 插值是否造成已见差异？

**设计：**`309016` 固定接受过的输入与权重，在两 runtime 串行比较 PatchConv per-camera／combined batch、位置插值直接 BF16／FP32后转BF16，以及不同 PROJ1 布局。

**结果：**PatchConv 两种 batch 和位置插值两种 cast 都 EXACT参考；PROJ1 官方视图 EXACT，native combined 布局有`314260/1204224`个值不同，contiguous control 反而有`440808`个值不同。

**解释更新：**在受测边界排除前两项假设；保留 PROJ1 调用／布局因素。不能把“连续”当成“数值更正确”，因此没有盲加 `.contiguous()` 或插值 cast。

### 4.3 先确认 attention backend，随后接受“没有现成 exact 替代”

**问题：**Torch SDPA backend 是否能直接恢复外部 FA2 的参考结果？

**设计：**`309182`／`309409` 固定 QKV，比较实际 backend 与两 runtime；独立 profiler 区分真实 CUDA kernel 和范围注释。

**结果：**Torch Flash 最接近，但仍有559个 BF16 output 值不同；Efficient／cuDNN／Math 的差异更多。没有一个受测 backend 达到外部 FA2 EXACT。

**解释更新：**这是一个局部 attention 差异，不是完整 ViT 误差占比。没有强切较差 backend，也没有把旧 FA2 二进制硬塞给新 runtime。

### 4.4 “逐相机 B1”先失败，HWD 才验证通过

**初始猜测：**PROJ1 差异来自 B3 batched 调用，逐相机就能消除。

**反例：**`309625` 保留 native singleton leading stride 的 B1 仍有`314261`个值不同，冻结前置门失败；未加载完整模型。

**进一步机制分析：**singleton 维没有自动消除其 stride 对 Linear folding 分支的影响。官方和 native 的 leading stride 不同；“相同逻辑 shape”不能证明相同 lowering。

**下一实验：**只移除 singleton metadata，保留非连续 HWD 视图和 offset，再恢复 B1 输出；不复制成2D，不改变 bias 融合。

**结果：**`309681` 的 HWD 原型冻结`1204224`元素 EXACT，实际 lowering 为3个 `mm`＋3个独立 `add_`；完整七例从4/7变6/7，但 state 仍失败，超限2→3。

**认识修正：**“局部机制修正成立”和“完整策略全部通过”是两个命题。保留 HWD 诊断价值，拒绝将其称作完整修复；这也是后续不能只看平均误差的第一个明显反例。

## 5. Hy 第二阶段：把 Expert 的竞争解释逐项拆开

### 5.1 先排除轨迹／时间／状态接缝，再查相同输入的 Expert

`313766` 精确复现 HWD baseline/state 的 native actions、十步轨迹和时间 embedding；实测增量符合“BF16标量×BF16速度，再加到FP32 latent”的原配方，状态 embedding 精确。固定同一官方 latent/time 后，官方 KV 条件仍有非零 raw-velocity 残差。

**被收窄的解释：**不能仅用 sampler 接缝、初始 state 投影或 native KV 解释全部残差。这里仍是条件比较，不是 free-running 官方 KV policy 验收。

### 5.2 QKV 融合、prefix 顺序、GU 融合是不同问题

| 研究问题 | 区分性实验 | 观测与对解释的影响 |
|---|---|---|
| QKV 融合是否改变投影？ | `313996` 固定 step0 suffix 输入比较 separate／fused | 每例9个 BF16 K/V投影值不同，但 normalized Q 在这两个投影臂内相同；另有跨 runtime Qnorm 微差，不能混为一个原因 |
| prefix 顺序是否改变 attention 算术？ | `314150` 捕获实际 Q/K/V、context／probabilities 后重放，仅同步排列 prefix K/V/mask，Q和suffix不改 | matching layout／contiguous 原序均 EXACT；排序引入局部输出差异，支持局部顺序效应 |
| 恢复顺序是否足以修完整 Expert？ | `314305` 全 Expert 条件轨迹干预，先复现原 sorted 控制 | 无条件变 EXACT，也非普遍改善；否定“单独恢复顺序就充分” |
| GU fusion 是否是 MLP 差异来源？ | `314364` 在真实捕获的 Expert0 normalized inputs 上比较 gate/up/activation/down | 所有受测中间量 EXACT；在这20组输入排除 GU fusion，不外推全部层／任意输入 |
| 更早层误差与局部误差怎样区分？ | `314485` 捕获32层，分 teacher hidden 与逐层 propagated hidden | 两种模式分离局部 block 误差与传播影响；没有用 hidden L2 计算“解释了动作误差的百分比” |

**方法上的修正：**重建算子先要复现实际捕获端点，否则比较的是两个“看起来合理”的实现，不是已观测路径。恢复顺序时局部 adapter 不改 Q，但早层变化会传播到后层 hidden/Q，因此不能称整个32层实验为 fixed-Q。

### 5.3 两个单独不充分的干预，组合后却在旧 runtime 完全恢复

**现象：**`314589` 只拆 QKV 没有任何 step 的速度变 EXACT，也有 step 变差；前述只恢复顺序也不充分。

**保留的假设：**两项改变可能相互作用。每项单独不充分，不等于组合效应为零。

**验证：**`315024` 在已拆分 QKV 的条件下只恢复 prefix 原序；先要求 sorted 控制全部1300张量／例／runtime精确复现历史。

**结果：**

- Torch2.7：两例、十步、两种 hidden 模式、32层 residual／output 及20个传播速度全部 EXACT Gold。
- Torch2.11：仍有残差；速度 relative L2范围约`.00307602–.00494808`和`.00316721–.00527045`。
- 实际 Q 布局一致，不能仅用布局标签解释 runtime 分歧。

**解释更新：**投影与顺序的联合条件在旧 runtime 成立，但还存在跨 runtime 算术因素。不能到此宣布整条 native policy 修复，也不能把 residual 一律归给第0层。

## 6. Hy 第三阶段：从“首差异在 mean”到输入驱动的原因模型

### 6.1 R26：首个不同阶段在 RMS mean，但此时还没证明充分性

**现象：**部分 block 的 residual EXACT、FFN output 不同；另一些 postnorm 输入已经受到上游 residual 影响。

**竞争假设：**差异可能来自输入已变、FP32转换／平方、mean、epsilon／rsqrt、scaling／cast 或最终乘权重。

**设计：**`315256` 预定 baseline/state 各两个 block 点，先复现实际 block oracle；每 block 捕获四次真实 norm。把同一旧输入／权重交给两 runtime，记录八个阶段，不在 block 内安装重建数学。

**结果：**旧184／新328张量审计中，16/16相同输入的 FP32转换与平方 EXACT，首差异全部在 `square.mean(-1)`。一处 BF16 postnorm 输入 EXACT，但17个 variance、5个 cast-back、4个 weighted output 值不同，末端 maxabs`.0078125`。

**当时能说：**差异首先出现在 FP32 mean，且能越过 BF16 rounding 影响真实 norm 端点。  
**当时不能说：**后续 primitive 在相同中间量下必然相同，或 mean 已解释全部层／策略误差。

### 6.2 R27：中间量交接检验下游充分性

**问题：**旧 variance 是否足以让新 runtime 的下游恢复？还是 `rsqrt`／后续算术另有独立差异？

**设计：**`315323` 先复现144个 native fixed-old 基线张量；在 variance、variance+epsilon、inverse、scaled、cast-back 五个独立位置交接旧值，只执行其后缀。每次先注入当前值做不变控制；使用当前阶段布局，不伪造未记录的旧中间量 stride。

**结果：**784张量审计，320个自值控制 EXACT；五种交接各16/16末端 EXACT，整个受测 suffix 也全部 EXACT。

**解释更新：**对这些固定输入／当前布局，旧 variance 已局部充分，支持 mean 变化是受测 RMS 下游差异的来源。不是任意布局／数据的定理，更不是能部署的算法：交接仍使用保存的答案。

### 6.3 R28：局部充分性是否能组合穿过真实 block？

**设计：**`315395` 在四个真实 block 内，只有实际 norm 输入、权重、live layout 全部 EXACT 旧捕获时才允许交接旧 variance。不匹配必须停止该 probe，禁止把无关 variance 套到变化后的输入。

**结果：**144张量审计，16次 norm 全部获准、无输入拒绝。两处 baseline 原有差异被恢复；两处 state 原本 EXACT 的控制保持 EXACT。四个 residual／output 都 EXACT旧 oracle。

**解释更新：**局部 RMS 充分性可在这四个实际 block 组合；但仍未测试32层传播，仍依赖保存 variance。下一步应寻找能从输入计算的机制模型，而不是继续扩大保存答案的注入范围。

### 6.4 查精确源码：提出加法树机制，而不是先换精度

两实际版本为 Torch2.7.1 `e2d141dbde55c2a4370fac5165b0561b6af4798b` 与 Torch2.11.0 `70d99e998b4955e0049d13a98d77ae1b14db1f45`。版本匹配官方源码与安装的 `Reduce.cuh` 表明：

- CUDA warp shuffle offset 从升序`1,2,4,8,16`改为降序`16,8,4,2,1`。
- 输入向量化门槛从 `dim0>128`改为`dim0>=128`，恰好涉及128宽 Q/K。
- FP32 `MeanOps` 仍为 FP32求和后乘 factor；新版16位 scalar 的 vector8 分支不适用这里的 FP32平方输入。

**形成的假设：**FP32加法分组／结合顺序改变，可能重现观察到的两版 mean；无需假设平方或 BF16 GEMM 开关改变。  
**限制：**源码／安装头文件是机制证据，不是对某次已编译 kernel 内部分支的观测；不自动定性库 bug。

### 6.5 R29失败、R30完成：相同 kernel 符号也不能证明相同内部树

`315596` 的首个 trace 实际含一个 CPU `user_annotation` 和一个同名 GPU 投影。解析器误计为两个 CPU 标签而 FAILED；没有 final raw/report，native 进程未运行。真实格式反例 RED→GREEN 后，新源码、新 round 的 `315622` 完成，旧失败保留。

**R30设计：**只 profile 单次 mean；直接观测平方张量的 dtype／stride／offset／对齐，保存完整 kernel 名称及 launch 元数据；要求 profile 前后输出复现各自历史。

**结果：**37文件、128数组、32窗口审计通过；两版均为 FP32 `reduce_kernel<512,1,…,4,4>`，block`[32,16,1]`，norm／Q／K grid为3／41／11。平方布局跨版本一致、offset0、16字节对齐。

**解释更新：**支持了输入／布局／launch 的条件预测，排除了“只凭原 Q stride 猜 mean 输入”的做法；但 kernel 符号／launch 相同仍不揭示 runtime vectorization 字段或编译后的 shuffle 树。继续测试算术预测，不把相同名字写成相同算法。

### 6.6 R31：用输入计算两版端点，不再交接保存 variance

**设计：**`316521` 预先固定四臂：标量／vector4 分组 × 升序／降序 shuffle。仅接受实际 FP32平方输入，用32 lanes、4累加器的显式加法和最后 factor 计算；不读取 variance 作为输入，不调用 native `sum/mean`，不使用 FP64／JIT／新 build。

| 显式树 | EXACT旧 mean | EXACT新 mean |
|---|---:|---:|
| 标量＋升序 | 8/16 | 0/16 |
| 标量＋降序 | 0/16 | 0/16 |
| vector4＋升序 | 8/16 | 0/16 |
| vector4＋降序 | 0/16 | 16/16 |

旧源码宽度选择“128标量升序／1024向量升序”恢复16/16旧 mean；向量降序恢复16/16新 mean。96数组、128对照及64个 GPU树输出均独立审计，后者全部 EXACT逐 lane／component 索引的 NumPy FP32重算。

**解释更新：**有了能够从输入重现两组局部端点的算术模型，强于“先看到差异，再提供旧答案”。仍不是直接读出编译分支，也不证明任意 shape／layout／数据都成立；多个 eager kernel 的原型不是加速候选。

### 6.7 R32：把输入驱动公式放回完整条件 Expert

**问题：**16组均值匹配，能否在新的 propagated LIVE input 上穿过32层和最终 head？

**设计：**`316722` 固定 separate QKV／prefix 原序／Gold KV／外部 latent-time 轨迹，共享重放核先跑未改 native，再跑仅依赖 live input／checkpoint weight 的 source-old RMS。每臂5160次 norm，包含 final norm／head；native 每例全部1300张量必须复现历史，原型匹配只是测量结果。

**结果：**5200张量独立审计，control2600 EXACT历史 native；原型两例各1300/1300 EXACT旧2.7，32层两 hidden 模式的全部 residual／output、20传播速度和20固定-Gold head 速度 EXACT。额外 BF16→int16 比较连 signed zero 位型也一致，不是内容哈希。

**已支持：**在这些联合条件下，input-derived RMS 模型闭合了已测跨 runtime Expert 差异。  
**尚不支持：**native vision KV、完整 free-running policy、七 fixture 或机器人通过。这里的 final head 参与干预，不能称其为不变负对照。

## 7. Hy 第四阶段：主动验证局部成功是否能推广，并接受否定

### 7.1 R33为什么必须跑，而不是在 R32停下宣布成功

R32使用 Gold KV 和固定外部轨迹；实际策略使用 native vision/prefill KV，latent 在每步按自己的 velocity 演化。输入分布与模块误差的耦合都变了。由条件 EXACT 外推策略恢复，是一个**待验证假设**，不是逻辑必然。

**R33设计：**`316975` 只在真实 `evaluate_velocity` 内启用三项 Expert改动，prefill 的128／2048 norm保持原生；用实际 observation 的 permutation，而非固定 Gold mapping。一个 eager backend，七例 control各两次全部先获准，再跑候选各两次，共28请求／280 NFE。保存 native prefix／conditions、十步 latent／time embedding／raw velocity／actions。

**来源限制：**七个旧 `actions.npz` 都没有逐步轨迹。七例 actions 精确复现历史；仅 baseline/state 有 `313766` 额外轨迹，各47字段 EXACT。其余五例明确是新控制双回放，不伪造历史逐步验证。

### 7.2 R33反例：Expert更接近旧算术，策略门却没有变好

**机械结果：**60文件、1680请求数组、280 NFE、native prefix／conditions 不变性、所有回放／作用域计数／Euler链／资源清理通过独立审计。

**质量结果：**padded 与 public 都是 control6/7、candidate6/7，state 仍失败。

| state／padded40×32 | HWD control | Expert candidate |
|---|---:|---:|
| 超限元素 | 3 | **4** |
| maxabs | .016632080078125 | .015106201171875 |
| RMSE | .0016343457521882086 | **.0017574000621361032** |
| cosine | .9999873074562833 | .9999853985877847 |

只有 episode_start／history／noise 的 RMSE 改善；baseline／current／prompt／state 变差。

**被否定的命题：**“条件 Expert 全 EXACT 就足够让 native policy 通过”在本次范围不成立。更小 maxabs 也不等于质量全面提升，候选被拒绝，不救容差、不改默认。

**仍可保留的解释：**不同 native prefix／trajectory 可能使组件误差非单调组合，甚至存在误差抵消；但“抵消”目前只是候选解释，**没有测出其因果份额，也没有证明它是唯一原因**。R32的条件等价与R33的完整策略失败同时为真。

### 7.3 新证据把下一问题移回 vision／Merger，而非盲扩 VLM RMS

对保留数组做 CPU 复核：

- baseline/state 的266738个 prefix embedding 差异全在 image tokens，relative L2`.01257768617059049`、maxabs`.75`；非图像 tokens 和 pad／modality masks EXACT。
- K/V 从 layer0 已不同，因此 VLM不是接收了相同 vision embedding 再自行产生全部差异。
- 历史 native Merger 输入与 Gold已有476775个值不同，relative L2`.006573505861752344`、maxabs`16`。
- R33 image-token 输出 EXACT历史 native `vision.merged`，但 R33实际 vision输入没有捕获，不能把这个输出桥接冒充同一次输入／输出观测。

**解释更新：**存在上游输入混杂，直接把 VLM RMS扩到2048或在失败 state 上调参数，会绕过尚未区分的原因。下一轮应固定相同 Merger 输入，分别看调用形式与 runtime，再决定是否继续查 ViT或Merger本体。

## 8. R34：在结果前写下下一轮如何判别原因

**冻结源码：**`c990f271`；**作业：**`317360`；单 A800、10分钟、旧2.7→新2.11串行，零完整模型，只读8个 BF16 Merger权重。**第8.1—8.3节是初版 `5d515b0a` 的结果前记录，不按后来的结果重写预测；第8.4—8.5节为结果后追加。**

### 8.1 竞争假设及可区别的预测

| 编号 | 待验证原因 | 什么结果支持／削弱它 |
|---|---|---|
| M1 | 目前输出差异主要随上游输入值变化 | 相同 form/runtime 内，Gold／native输入导致差异；若相同输入两 form/runtime均一致，会强化“已测Merger外”的解释，但仍不能给完整策略归因百分比 |
| M2 | 完整 Merger 的 per-camera／batched后续运算另有影响 | 固定相同输入及 runtime，official／native HWD之间从某阶段开始不同；PROJ1已局部验证不代表完整Merger必然一致 |
| M3 | Merger 后半段还有跨 runtime算术差异 | 固定同一输入、同一调用形式，旧／新runtime在mean、GELU、softmax、sum或Linear等某阶段出现差异 |
| M4 | 阶段重建并未忠实复现实际API | 重建末端不能 EXACT对应实际 API；此时必须拒绝解释中间阶段，不把重建误差归给模型 |

M1／M2／M3并不互斥；不能把一次局部 EXACT写成对所有输入的普遍排除。

### 8.2 冻结矩阵、硬门与结果项

- **输入值来源：**接受过的 Gold308690 per-camera PROJ1输入；历史 native313766 Merger输入。
- **调用形式：**官方 AST Tensor-forward／pooler逐camera；未改动 native HWD `camera_merger`。
- **runtime：**两实际版本串行运行。
- **观察：**14阶段＋实际API末端，共60数组/runtime、120总数组；official布局在每camera concatenate前记录，concatenate只是产物表示。
- **硬门：**每个阶段重建末端 EXACT其实际API；旧runtime官方Gold调用恢复Gold merged；新runtime native/HWD/native输入恢复历史native merged及R33输出桥。输入／runtime／来源／checkpoint准入先于权重读取。
- **结果项：**其余 form／origin／runtime的逐阶段差异、首差异与完整merged对照。不预设必须相等，也不为得到“修复”而改数学。
- **布局限制：**旧metadata没记storage offset，只能有界zero-offset重建并明确假设；不能宣称观察到了旧offset。

### 8.3 结果回来后怎样更新本文

必须先完整回传并独立审计120数组；跨runtime比较读取两端原始数组，不从各自锚点通过推断跨版本一致。若硬门失败，记录失败位置／可能的接线或重建原因，再设计反例，不覆盖旧快照。

当前已通过38项专项、1018项所选回归及独立导出、41个subtests，另有2个module skips；独立评审无阻断项。**缺口：**完整 `load_history()` 尚无临时canonical目录的端到端CPU正反例，个别helper测试不能替代实际准入。裸全仓仍有20个依赖／native扩展收集错误、25 skips，不称全仓绿色。

### 8.4 结果后追加：不是 runtime 差异，完整 Merger 另有 PROJ2 调用组织效应

原 `317360` 完成 `0:0`，无观测错误、无重提。完整6文件／227763005字节取回，独立审计120数组、8个实际 API 端点、旧 Gold／新 native历史锚点和 R33输出桥均通过。所有56项同 origin／form 的跨runtime阶段比较 EXACT，额外60对BF16位型比较含signed zero也全部相同。

**固定相同输入／runtime，改变调用形式的结果：**

| 输入来源 | 首个不同阶段 | merged 不同元素 | maxabs | relative L2 |
|---|---|---:|---:|---:|
| Gold | PROJ2 | 254 | .015625 | .00004932582823326906 |
| 历史 native | PROJ2 | 287 | .015625 | .00007520447640456106 |

在两runtime中结果一致。PROJ2之前的所有阶段，包括mean、predictor、softmax、products/sum和activation，均EXACT。因而“PROJ1前置门通过”确实不足以代表完整Merger相同；最后一层Linear还存在可独立测到的调用形式效应。

**固定调用形式，改变输入来源的结果：**

- 输入从一开始就有476775个值不同，relative L2约`.00657350586`、maxabs`16`。
- official形式 merged差异266736个值，relative L2约`.01257794642`；nativeHWD形式为266740、约`.01257768738`；两者maxabs均`.75`。
- 这些是**同form的origin对照**，与R33“native输出对Gold输出”的混合对照266738不是同一比较，不能互换数字或计算“解释了多少策略误差”。

### 8.5 假设如何修订，以及下一步还不能跳过什么

| 结果前假设 | R34后的处置 | 限制／认识修正 |
|---|---|---|
| M1：上游输入值差异 | **输入混杂继续得到支持，但不能认定为唯一／主要原因** | 同form换origin仍有完整输出差异；同时还发现M2，无法直接分摊完整policy原因 |
| M2：Merger调用形式另有影响 | **支持，首差异定位至PROJ2** | 同值通过activation仍EXACT，最后Linear不同；不再把PROJ1局部成功当完整Merger等价 |
| M3：受测Merger跨runtime算术差异 | **在这两origin／两form上未观察到** | 四对完整阶段及端点跨版本EXACT；这是限定范围的否定证据，不是任意数据／库API一致性的保证 |
| M4：阶段重建不忠实 | **这些调用的8个实际API端点验证排除了该解释** | 只保证当前source／输入／布局／端点，不取消未来新数据的同样前置门 |

**新的具体猜测：**最后Linear的batch组织改变了GEMM几何。实测activation为contiguous BF16，官方每camera `[1,7,7,2048]`，native一次 `[3,7,7,2048]`，stride同为`[100352,14336,2048,1]`、offset0；匹配Torch源码预示biased addmm的M为49对147。**尚未观测具体cuBLAS算法／epilogue／加法树，不能把这个机制猜测写成已确认内核原因。**

源码依据为精确版本的 [`2.7 Linear.cpp`](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/Linear.cpp#L57-L69) 与 [`2.11 Linear.cpp`](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/Linear.cpp#L52-L74)；这两个链接证明的是source路径，不替代实际CUDA launch／GEMM算法观测。

**下一项最小判别实验：**只取已获准的PROJ2输入activation与同一weight/bias，先复现两个历史form端点，再改变最后Linear的调用组织，分清call geometry、bias表达式与布局因素。若不能复现原端点，先拒绝该重建。它不需要先跑完整策略，更不能直接围绕state重调参数。

**仍未解决：**进入Merger之前的vision输入差异、PROJ2形式调整是否改善native-policy，以及机器人质量。R33的6/7失败和所有生产默认／数值门不变。后续记录应继续保留PROJ2局部发现与未解决上游问题，而不是改写成“找到最终原因”。

### 8.6 R35结果前追加：只改变最后Linear，区分行几何与bias表达式

本节在R35提交前写下，固定于 `32ea141e`，不按后来结果改写预测。依据R34提出下列可证伪问题：**同一activation／weight／bias，只改变最后Linear的调用组织，能否直接重现两个历史端点？显式在BF16 matmul后再加bias，是否又引入另一类差异？**

预先固定七臂，不根据结果追加调参：

| 臂 | 保持的输入值 | 只改变的因素／预测 |
|---|---|---|
| batched_bias | 同一三camera activation | 原B3调用；必须先恢复R34 nativeHWD端点 |
| per_camera_b1_bias | 同一activation逐camera视图 | 原B1调用；必须恢复R34 official端点 |
| per_camera_hwd_bias | 同一camera值、去singleton维 | 若仍flatten到49行且数值等价，应接近／匹配B1；这是待验证预测 |
| per_camera_flat_bias | 同一camera值显式49行视图 | 区分外层维数与M49几何，不能先假设必然EXACT |
| flat147_bias | 同一三camera值显式147行视图 | 区分外层维数与M147几何 |
| batched_mm_add | 同一B3值与参数 | 无bias matmul先产出BF16，再加BF16 bias；与融合表达式有明确中间舍入区别 |
| per_camera_mm_add | 同一camera值与参数 | 同样显式舍入，但每次49行，用于区别几何效应与bias表达式效应 |

只读PROJ2两个权重张量，不跑完整Merger或policy；先准入R34两端完整120数组、来源／runtime／checkpoint及同值activation。每臂两次非profile控制，再做一次profile；计算在A800上执行，**仅记录CPU侧aten调用和input shapes，不采CUDA kernel／算法／时延作为证据**。所有profiled／unprofiled输出必须EXACT；两个历史form端点是硬门，其他臂是否匹配是结果，不强迫通过。

预期保留30数组、14个CPU trace及28项对两个历史端点的比较。先做TDD／独立评审／回归，再冻结新round和一次提交回执。即使隔离算子能重现差异，也不能跳过仍未解决的上游vision与完整策略质量问题。

**提交状态追加：**作业 `317555` 已由持久回执确认，冻结 `round035/32ea141e`，单A800、10分钟；26项专项、1044项所选回归及独立导出通过，另有41个subtests／2个skips，裸全仓仍有既有20个收集错误。这条提交状态不包含数值结果；后续实测另见第8.7节，不改写上述预测。

### 8.7 R35结果后追加：有bias路径按M49／M147分组，显式拆bias不是修复

`317555` 完成 `0:0`，无观测错误或重复提交。17文件／18425066字节完整取回，独立审计30数组、14个CPU trace、14次profile前后数值回放、4个历史form端点和28项对照通过。没有构造模型或运行policy，只读取两个PROJ2 checkpoint张量。

| 预先固定臂 | EXACT官方端点 | EXACT native端点 | 实际aten观测 |
|---|---:|---:|---|
| batched_bias | 0/2 | 2/2 | 1次addmm，M147 |
| per_camera_b1_bias | 2/2 | 0/2 | 3次addmm，各M49 |
| per_camera_hwd_bias | 2/2 | 0/2 | 3次addmm，各M49 |
| per_camera_flat_bias | 2/2 | 0/2 | 3次addmm，各M49 |
| flat147_bias | 0/2 | 2/2 | 1次addmm，M147 |
| batched_mm_add | 0/2 | 0/2 | M147的mm后再BF16 add |
| per_camera_mm_add | 0/2 | 0/2 | 3次M49的mm后分别BF16 add |

**认识增强：**M49／M147从源码预测变成了这几次实际调用的aten形状观测。对有bias的Linear路径，改变外层rank但保留M分组，结果仍分别落在官方／native端点；支持受测行分组效应，不再只是完整Merger上的相关位置猜测。

**否定结果同样保留：**拆成BF16 `mm→add`不匹配任一历史端点。对official参考，Gold来源76773值不同、relative L2约`.00275639`，native来源77608值不同、约`.00267861`，maxabs都`.5`。不能把“显式拆bias”当作精确替代，也不能未经内核观测把全部差异精确分配给某种epilogue。

**进一步限制了原解释：**对已保存七臂输出补做CPU比较发现，显式 `mm→add` 的B3与per-camera两臂在两origin上恰好EXACT，BF16位型也相同。这是对已有数据的追加分析，没有新GPU候选或事后调参。它反驳了“只要M49与M147不同，任何表达式都必然不同”的更强说法；行分组效应须保留**有bias执行路径**这一条件。

普通数值EXACT不自动保证正负零位型。为此另做18对BF16→int16检查（14次replay＋4个历史锚点），包含signed zero，未发现位型差异；该补充不是内容哈希，也不将未检查的比较一概称作位型证明。

**仍未验证：**实际cuBLAS算法／GPU epilogue／累加树、任意数据几何、上游ViT输入差异以及完整policy质量。这里仅采CPU侧aten事件，parent matmul与child mm不冒充两次GPU GEMM，更不据trace时长宣称加速。

**下一问题：**将per-camera biased PROJ2放回examples-only完整HWD Merger，先验证真实完整端点，然后仅改变这一项重测native-policy。若它只恢复局部而state仍失败，继续保留该反例；不把R35算子端点等价跳写成七例／机器人通过。

### 8.8 R36结果前追加：完整端点前置门与单一PROJ2策略实验

本节在R36提交前固定于 `a5078137`，不按后来结果改写预测。只改变最后PROJ2的biased Linear：B3→三个B1，保持HWD PROJ1、原生Expert QKV／attention顺序／RMS、native vision/prefill、Euler10、noise和原质量门。**不混入R33三项Expert改动。**

| 待验证命题 | 判别条件／如何接受反例 |
|---|---|
| R35的activation端点恢复能组合到真实完整Merger | 先固定R34两输入来源：原HWD与带观测control恢复native端点，带观测candidate恢复official端点；任一失败先停，不构造完整模型 |
| instrumentation没有改变control算术 | 带观测与未观测完整Merger一致；完整策略七例control各两次必须精确复现旧actions及原6/7，再允许candidate |
| 单独改变PROJ2能改善原七例数值代理 | 原padded40×32 abs/cos门不变，public20另报；改善、无变化、变差均保存，零效果不强行判错 |
| 对最终策略的影响只经由PROJ2及其下游 | 每case实际Merger输入、PROJ2 activation及布局、非视觉prefix、mask／positions／suffix条件不变；视觉embedding／KV及后续轨迹允许变化且完整量化 |

每请求将捕获真正的 `vision.merger_input`、`vision.proj2_activation`、`vision.proj2_output`、`vision.merged`，补上R33缺失当前vision输入的限制，不用阶段重建冒充实际激活。四个新增字段使每请求64个，总1792个；**仍未保存实际prepared.images，不能写成65字段或逐值重捕获图像证据**。

独立完整前置门保存20数组，包括两臂分别捕获的activation和实际PROJ1子证明，不只靠“passed”摘要。原PROJ1子证明与新完整Merger证明保留不同kind，不冒称额外attention profiling。

计划是单A800、20分钟、先零模型前置门再一次模型加载，七例两臂各双重放，共28请求／280NFE；无prepare、warmup、profiling或时延目标。每臂Expert都需观测129次original RMS／32 block／32 attention每NFE，保证本次没有悄悄开启R33候选。所有cleanup／raw／原门独立审计后才追加结果；局部成立不预告策略或机器人通过。

独立复核还将利用**已经保留的R33作业316975七例control轨迹**核对旧60字段，这是明确命名的额外历史来源，不代表更早309681的NPZ突然有了velocity数据，也不增加GPU请求。完整入口链仍缺临时目录端到端CPU测试；该覆盖缺口保留，不用1088项所选回归替代本轮实际准入和完整原始审计。

### 8.9 R36结果后追加：state达标，但失败迁移到noise

作业 `317665`／冻结 `a5078137` 完成 `0:0`，无观测错误、无重提。完整60文件／835570416字节取回，20个前置数组、1792个请求数组、28请求／280NFE及所有回放、mask／perm／visual bridge、Euler链、作用域与close清理均独立审计。两臂均为original Expert，每NFE129original RMS／32block／32attention，确认没有带入R33三项Expert改动。

**前置命题成立：**在两来源的实际完整Merger上，观察前后control恢复native端点，candidate恢复official端点；真实输入、各自捕获的activation和callback/view/read-only证据都获准。真实策略中实际Merger输入／activation及布局、非图像prefix、条件／noise／time保持不变，只有预期的PROJ2及下游可变。

**完整策略达标命题仍被否定，但不是“state一直没变”：**padded和public均control6/7、candidate6/7；state从3个超限变0而通过，noise从0变6而失败。

| padded40×32 | HWD control | 单改PROJ2 candidate |
|---|---:|---:|
| state超限 | 3 | **0** |
| state maxabs | .016632080078125 | .0078125 |
| state RMSE | .0016343457521882086 | .0017172309821804235 |
| noise超限 | 0 | **6** |
| noise maxabs | .01318359375 | .039031982421875 |
| noise RMSE | .0017243589961676209 | .0029772034588235836 |

noise cosine仍有`.9999559557532167`，不能覆盖逐元素失败。RMSE在episode_start／current／history／prompt四例改善，baseline／state／noise三例变差。**仅写“6/7没变”会掩盖失败迁移；仅写“state修好了”又会掩盖新反例。候选作为七例统一合格配置被拒绝。**没有按case选择实现、放宽门限或改默认。

控制证据也加强了：七例actions EXACT旧309681；baseline/state各47字段 EXACT313766；另有七例各60字段 EXACT已保留的R33 control，总420项。后者明确引用316975，并不是给更早309681 NPZ补造velocity。

#### 同一prefix扰动为何不能预告统一改善

对已保存数据额外做CPU复核，确认baseline／state／noise在各臂内部拥有EXACT相同的实际Merger输入、activation、merged、prefix embedding及K/V。state与baseline的差别在suffix state，noise的差别在初始noise／latent；tokens相同。相同prefix变化却产生不同最终误差响应，支持**效果依赖后缀条件／自由轨迹**，不能由局部端点更接近参考就预告所有请求改善。这不是新GPU试验，也不构成模块误差百分比。

#### 从“可能抵消”到一个可验证、但严格局部的结论

本轮真正捕获的baseline/state Merger输入 EXACT历史313766，仍与Gold输入相差476775值、relative L2约`.00657351`。candidate实际merged EXACT R34 official form在该native输入上的端点。**这是本轮当前输入证据，仍不回填R33未捕获的vision输入。**

对这一个共享vision context（baseline/state不是两个独立视觉样本），比较三个经验证端点：

```text
U = official_form(native_input) - official_form(Gold_input)
B = native_form(native_input)   - official_form(native_input)
E_control = U + B
```

实际向量验证上述恒等式；以Gold输出范数平方归一化：

```text
(||E_control||² - ||U||²) / ||Gold||²
  = 2<U,B>/||Gold||² + ||B||²/||Gold||²
  = -1.2201970422635243e-8 + 5.655137735145709e-9
  = -6.5468326874961355e-9 < 0
```

也就是说，在**这个Merger输出、这个L2指标与这个共享视觉输入context**上，旧批调用差异`B`对输入来源效应`U`有净抵消。改为official调用形式后，merged不同元素266738→266736，relative L2却略增`.01257768617→.01257794642`。这是局部净抵消的直接端点证据，不再只是猜测。

**边界必须保留：**不能据此证明最终action误差为何变化、给出因果解释比例，或解释R33 Expert修正失败的全部原因；也没有观测具体cuBLAS算法或epilogue。此前“完整策略可能存在误差抵消”的强说法仍未闭合，只有这里的有限向量关系被验证。

#### 下一问题的选择，不是下一轮成功的预告

R33的E（Expert bundle）单独未达标，R36的P（PROJ2）单独使失败迁移。下一项有依据的问题是一次**统一配置的E×P四格交互验证**：前三个已知单元格必须在当前运行中重现，再测both；不把两组旧action差值相加当作组合结果，不按state/noise路由。

这是由已见反例触发的机制交互／事后组合选择，不是独立泛化验证；E本身含三项改动，四格不能再分摊三者贡献。若both仍未统一7/7，结束这一轮组合搜索，转上游frozen ViT定位；即使7/7，也只保留原七例代理资格，不能自动推广默认、机器人或速度。具体矩阵、交互量和停止条件应在新作业前另行固定。

### 8.10 R37结果前追加：一次E×P机制交互验证与停止条件

本节在R37提交前固定于 `e97f22e7`，不按后来结果改写预测。R33的E和R36的P各自未达七例统一资格，但作用边界清楚，失败模式不同。**这不是“两个6/7应该拼成7/7”的推论**：下一步只检验Expert bundle是否改变PROJ2所引入prefix扰动的响应，不再盲扩参数／按case拼实现。

| 固定组 | E：separate QKV＋原prefix列顺序＋LIVE旧树RMS | P：PROJ2 B3→三个biased B1 |
|---|---:|---:|
| base | 关 | 关 |
| Eonly | 开 | 关 |
| Ponly | 关 | 开 |
| both | 开 | 开 |

所有七例统一应用，全部双重放。前三格必须先在当前运行中完整复现：base与Ponly各64旧字段对R36，Eonly的60旧字段对R33；Eonly四个新视觉字段只对当前base，不冒称R33原来捕获过vision输入。任何raw／dtype／来源／metadata／回放不匹配都阻止both，不以“都是6/7”替代复现。

预注册两个交互量，直接从实际数组在**CPU FP64**分析，不改变GPU数学精度：

```text
I_v0 = (v0_both - v0_Ponly) - (v0_Eonly - v0_base)
I_actions = (a_both - a_Ponly) - (a_Eonly - a_base)
```

四格共享初始latent、time／state embedding，首步速度用于检验bundle对不同prefix/KV的响应；最终actions包含自由轨迹反馈。零／非零交互、改善／恶化均是结果，不给交互量另设“必须非零”的质量门，也不从RMSE、超限计数或范数差推断交互，更不将其解释为误差百分比。

保持新鲜完整Merger前置门20数组，Expert启用前准入R32完整5200支持；四格固定输入／activation及布局、非视觉prefix、mask／segment／RoPE／条件一致，固定P切换E时全部视觉输出／prefix／KV也须一致，防止Expert作用域泄漏到prefill。P改变时允许视觉／KV及后续轨迹变化；每请求内部KV只读不变。

资源预定单A800、20分钟、两进程串行：零模型前置门，再一个backend完成56请求／560NFE／3584请求数组，另存14个完整交互数组。容量尚未实测；超时或少任一case／replay都只能是不完整诊断，不能删数据救完成。

**停止规则：**机械准入完成而both仍未统一7/7，就停止本轮组合搜索，下一步转上游frozen ViT定位，不继续state/noise特调。即使7/7，也只取得当前七例数值代理资格，不改生产默认、不宣称机器人质量／速度／泛化。这是已见失败触发的**机制交互与事后组合验证**，不是独立测试集；E包含三项修改，四格不能分配它们各自贡献。

**提交状态追加：**`317924` 已由持久回执确认，冻结 `round037/e97f22e7`，单A800、20分钟、两进程串行。59项专项、1147项所选回归及隔离导出通过，另有41个subtests／2个skips；裸全仓仍有既有20个收集错误。评审指出历史raw类型／sampler准入偏晚，已用五个“双份同错”反例RED→GREEN前移到模型构造前，保留的42份真实历史请求／2632字段也通过新guard。完整GPU入口仍无端到端CPU测试，不能从组件通过推断本轮资格；这条提交状态不含数值结论，实测另见第8.11节。

### 8.11 R37结果后追加：交互确实非零，但组合仅5/7，执行停止规则

作业 `317924`／冻结 `e97f22e7` 完成 `0:0`，无观测错误、重提或取消。117文件／1654149558字节完整取回并核对清单、大小、源码标记与两个单独获取的header。独立CPU auditor未导入producer，核验20个新鲜前置数组、3584个请求数组、14个FP64交互数组，共56请求／560NFE，以及全部实际dtype／几何、双回放、mask／排序／visual bridge、Euler链、作用域计数和cleanup。

**前三格复现先成立，才测both：**base对R36 control有896项raw字段比较、Eonly对R33 candidate有840项、Ponly对R36 candidate有896项，均包含两次回放。Eonly四个新增视觉字段对当前base EXACT，未冒称R33原来捕获过这些数据。固定P改变E时全部Vision／prefix／KV不变；四格固定输入／activation／布局、非视觉prefix、mask／segment／条件及初始latent保持一致。

#### 观察到的不是“勉强差一点7/7”，而是新的失败模式

原padded和public门都为base6／Eonly6／Ponly6／**both5**。

| case／padded超限元素 | base | Eonly | Ponly | both |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 0 |
| episode_start | 0 | 0 | 0 | 0 |
| current | 0 | 0 | 0 | 0 |
| history | 0 | 0 | 0 | **1** |
| prompt | 0 | 0 | 0 | 0 |
| state | 3 | 4 | 0 | 0 |
| noise | 0 | 0 | 6 | **6** |

both的state继续通过，但noise没有被E救回，并新增history失败：

- history maxabs`.0107421875`、RMSE`.0017224562468873739`、cosine`.9999835400631613`，逐元素超限1个。
- noise maxabs`.0421142578125`、RMSE`.0031483730505876707`、cosine`.9999509332927161`，超限6个。
- history的both RMSE比另三格都小，仍新增逐元素失败，再次说明一个聚合指标改善不能替代原门。不能删除这个case、放宽元素容差或按case选择E/P。

**“统一组合能让七例全达标”的命题被否定，组合候选拒绝。**不是前置控制漂移或漏了第二次回放造成的不完整结论；本轮机械执行与原始证据门通过，失败发生在数值质量代理。

#### 交互量支持了什么，没有支持什么

独立NumPy FP64从四格实际输出重算第8.10节的两条公式，14个完整交互数组数值EXACT计算式；不将普通EXACT冒称未检查的signed-zero位型证明。

| 交互量／七例范围 | 非零元素（每例1280） | maxabs | L2 norm |
|---|---:|---:|---:|
| 首步raw velocity `I_v0` | 747–797 | 每例均`.03125` | `.17198954905–.19135488884` |
| 最终actions `I_actions` | 1096–1119 | `.00537109375–.0069580078125` | `.04744580724–.05132326981` |

因此“只需把两个旧action差相加就能预告组合”的加性解释被实际数组反驳。四格首步共享latent／time／state条件，且固定P切换E并未改变prefill／KV，说明**首步Expert bundle对不同prefix／KV的响应已非加性**；不是只看到最终自由轨迹RMSE不同后再猜交互。最终action交互还包含后续轨迹反馈。

不过，**存在非零交互≠交互有利≠七例达标**。它也不分离E内三项修改的贡献，不证明某个cuBLAS算法／epilogue、不提供“解释了多少策略误差”的比例。当前七例是事后选择的机制检验范围，不能升级为泛化、机器人质量或性能证据。

#### 对解释和下一步的实际修订

R32的条件Expert EXACT、R35局部PROJ2端点恢复、R36失败迁移、R37组合5/7同时成立。继续拼组件或只围绕state/noise调参，已不再是本轮预先允许的判别路径。

**现执行停止规则：结束这一轮E×P组合搜索，下一步转上游frozen ViT定位。**不追加新组合、不做case-dependent路由；生产默认、runtime／精度、Euler10和原容差均不变。上游实验也须先复现真实固定端点，再区分ViT内部差异，不将Merger局部净抵消当作最终策略归因，也不回填R33未捕获的输入。

### 8.12 R38结果前追加：上游ViT最短可锚定前缀，而非再拼修复

本节在R38提交前固定于 `311c66d0`，不按后来结果改写预测。R37停止组合搜索后，下一问题固定为：**同一份实际prepared images进入ViT，差异是否已出现在block0–3；最早不同的、可验证实际边界是什么？**

选择四块而不是一块或27块，是因为Gold实际只保留了block3、block26输出：四块是能抵达第一个已保存块端点的最短前缀。主会话已核对raw角色与shape：有三份Patch输出、三份block3全六帧输出和prepared images；没有native的逐块历史轨迹。此前“patch＋position”重建过block0输入，但不能冒称真实历史边界；这轮要真正观察。

STD只读header核实51个BF16张量均存在，总161436544字节，包括Patch／position及blocks0–3所需norm／QKV／投影／MLP参数；不读VLM、Expert、Merger或后23块。视频配置核对为开启、stride4、past_drop=None，固定时间PE源默认值不改。该预检未读取权重数据，也没有使用GPU。

| 竞争解释／边界问题 | 这轮能区分的结果 | 不能跳写的结论 |
|---|---|---|
| 差异在Patch／位置或进入第一块前已产生 | 实际Patch／position／block0输入先比较，旧结果须复现已有Gold | 不因309016曾通过就省略当前实际端点验证 |
| 首个不一致已在blocks0–3某个实际阶段 | 逐块输入、norm1输入／输出、QKV、attention投影、residual、norm2、MLP、output，另捕获block3时间PE／temporal V | 后续输入已不同的阶段不等于独立局部原因；不预设attention或RMS是根因 |
| 差异只在更后层出现 | 如果四块均一致，本轮结论是此前缀未定位 | 不自动扩读4–26，不冒称整个ViT一致 |
| 观察器或重建改变了路径 | 官方三份block3全六帧输出须EXACT历史Gold；native观察前后完整四块prefix控制须一致 | 任一端点失败即停止，不拿“接近”替代硬门或回退成相似循环 |

官方旧runtime只构造原四块ViT模块，用真实视频函数与SpaceTimeBlock3逐camera调用；不是构造完整HyVLA。native新runtime保留原`vit_forward`源码循环，在实际block3返回后用独立停止信号退出，禁止block4触及权重。该端点是block3真实返回，不称完整27块API输出。两边都不改数学，只加实际module／方法／功能调用观察器，并要求输入／权重只读和异常恢复。

预定旧124、新42共166个完整BF16数组；block3的norm1输入和输出实际都为4维，保留该事实。官方新内部输入是**当前新观测**，native内部轨迹同样是新数据，不补写313766历史。prepared images的live stride有历史metadata；旧storage offset未记录，零offset重建须标注假设。

**重要限制：**本轮比较的是旧runtime／per-camera／FA2对新runtime／combined／SDPA，runtime与调用形式仍混杂。它只定位最早实际边界，不从两格比较宣称runtime单因果、具体kernel原因或策略误差份额。若要做同输入局部因果切口，必须另立后续判别实验。

计划仍是单A800≤20分钟、旧→新两进程串行、零完整VLA／backend，无prepare／warmup／profile／性能目标；166数组全部审计。官方依赖无法导入、旧端点不能复现、缺任一camera／frame／stage均只得未获准诊断；不装依赖、不建新runtime、不放宽门限。当前CPU组件测试不替代旧环境真实导入或Gold前缀资格。

**提交状态追加：**`318174` 已由持久回执确认，冻结 `round038/311c66d0`。60项专项、1207项所选回归及隔离导出通过，另有41个subtests／2个skips；裸全仓仍有原20个收集错误。评审纠正package／policy来源schema误读，并用失败注入保证观察后的锚点／同步失败仍保留未获准数组与状态，未改历史数据。一次只读预检SSH超时后，连接探测正常且无遗留进程，纯元数据准入由系统Python持久确认；实验解释器仍为旧2.7／新2.11，未重提模型作业。这条提交状态不包含数值结论；真实执行与166数组审计结果另见第8.13节。

### 8.13 R38结果后追加：首差异在block0投影后attention，尚未拆成单一原因

`318174`／冻结 `311c66d0` 完成 `0:0`，无观测错误、重提或取消。完整6文件／770439476字节取回，核对清单、大小、源码标记和两个单独取得的header。独立CPU auditor未导入producer，核验官方124／native42共166数组、41个比较边界及逐camera／frame指标。

**预先要求的实际端点硬门全部通过：**官方三份block3全六帧输出、Patch与位置输出 EXACT历史Gold；native观察前后实际block3控制 EXACT。四块调用顺序、未进入block4、51权重范围、所有dtype／几何／链连续／residual／时间PE／API当前帧桥接和cleanup均获准。这里的EXACT是数值等值，不冒称尚未另测的signed-zero位型证明。

#### 哪些解释被排除，哪些仍混在一起

同一实际输入下，images、Patch、position、block0输入、norm1输入／输出及QKV均EXACT。**第一个不同的实际捕获边界是 `block0.attention`，即输出投影之后，而不是已经隔离的attention context。**

| 边界 | 不同值数 | maxabs | RMSE |
|---|---:|---:|---:|
| block0 attention投影后 | 2139 | .03125 | .000043909956944 |
| block0 residual／norm2实际输入 | 642 | .03125 | .000053090967356 |
| block0 norm2输出 | 1406 | .03125 | .000046897356430 |
| block0 MLP输出 | 21532 | .0625 | .000157515718338 |
| block0 output | 9889 | .0625 | .000211912479530 |
| block1 output | 438831 | .125 | .001227866431582 |
| block2 output | 1432143 | .5 | .002584514271401 |
| block3 output，全六帧 | 2089847 | .5 | .003857332748271 |

首个边界的三个camera分别483／675／981值不同。后续block输入已经不同，不能把它们的norm／MLP输出差异各算成独立“根因”，也不能从数目增长换算策略误差比例。比较器relative L2以native端点范数为分母，首边界约`.00002361598`；这不是原策略质量门，EXACT／maxabs／RMSE则是对称指标。

- “Patch／位置或第一块输入已经不同”在这个context被否定。
- “首差异发生于block0 norm1／QKV”也被否定；没有依据此时盲扩LayerNorm或RMS修正。
- “差异只在更后层才开始”被否定，无需自动读取后23层。
- **attention context、输出Linear、runtime与调用组织仍未拆开。** 旧per-camera FA2对新combined SDPA是混合比较；2139个差异不能直接命名为“SDPA误差”或“GEMM batch误差”。局部PROJ2的既有机制也不能未经实验移植到这里。

#### 一个独立于hidden传播、但不取代首边界的小现象

block3固定时间PE也有一个端点差异：旧三个camera的PE彼此EXACT，各与native在零基索引frame4／feature571这一元素不同，旧值`-.0830078125`，新值`-.08251953125`，maxabs`.00048828125`。跨camera拼接后有3个差异，**其实是一个固定PE点被三个camera重复使用，不是三个独立样本**。

该PE只依赖固定帧数／宽度／base／dtype，不消费已变的hidden，所以其端点差异不是hidden传播的结果。但具体算术来源和最终策略影响仍未验证；不能看到源码里scalar表达式不同就直接写成原因已证实，也不因此跳过更早的block0问题。

#### 老实验与新实际边界终于怎样接上

对已保存数组做追加CPU复核：本轮实际block0输入、norm1、QKV与309182、309409两轮的official／native对应重建值均EXACT，共12项配对。它把此前“patch＋position重建／冻结QKV”的**数值输入**与新实际边界连接起来，使旧同QKV的attention控制在这个context有了更强的来源支撑。

边界仍要保留：这不把过去未观察的内部stride／offset倒写成历史事实；R38也未保存attention的投影前context，不能把旧保存context冒称本轮新捕获。已有TorchFlash对外部FA2的559个context差异可作为同值QKV的旧局部证据，不必再次盲扫全部SDPA backend；它仍不是最终2139个投影后差异的解释比例。

#### 下一步问题收窄

下一项最小判别应在这个已准入block0输入上，**真正捕获投影前context和投影操作数，并先复现两端实际attention API输出**，再考虑同输入context／调用形式切口。不能只复制一个“看起来一样”的attention循环、把新runtime的旧调用形式匹配强行设为必过，或把局部定位写成整策略修复。

R37的组合停止决定继续有效：不恢复Expert×PROJ2搜索、不按case路由、不改runtime／精度／sampler／门限。本轮没有新的机器人、速度、泛化或生产默认资格。

### 8.14 R39结果前追加：把实际context与输出投影拆开，不强迫旧角恢复

本节在R39提交前固定于 `e058161a`，不按后来结果改写预测。针对R38“QKV以前一致、投影后2139个差异”的边界，下一问题不是再搜backend或策略组合，而是：**实际attention context已经不同吗；在同一context值上改变投影分组又有什么影响，两者有无局部有限差分交互？**

只读block0 `attn.qkv.{weight,bias}` 与 `attn.proj.{weight,bias}` 四个BF16张量，STD header已核实合计10626048字节。旧实际 `_ViTAttention` 类也已在R38绑定的原源文件中确认；不能用一个相似循环替代。旧进程执行原API三次camera6，新进程执行原native两个方法combined18，从R38实际norm1交接开始，不重算上游norm或加载整prefix。

先捕获投影前真实context与操作数，并设以下机械硬门：本轮QKV、未观察API控制、观察API、proj输出与自身投影回放必须分别EXACT对应R38实际端点；输入／权重只读、调用次数／scope／cleanup正确。官方21字段必须先完成并在新进程准入，才允许native四权重读取。保留R38 helper自己的原snapshot路径／namespace，不放宽旧round守卫。

然后在新runtime复用**同一实际native投影输入buffer及其layout**，固定四个局部算子单元：

| context来源 | 一次combined18 biased Linear | 三次camera6 biased Linear |
|---|---|---|
| N：本轮实际native context | N18，自值硬控制 | N6，同值分组切口 |
| O：本轮实际官方context | O18，同当前layout的context值切口 | O6，旧角结果 |

N18必须先EXACT当前native端点，失败立即阻止其他切口。其余三格是结果，**尤其新runtime的O6不强迫匹配旧2.7端点**；若有残差，runtime／layout／lowering仍可能混杂，不能为了“恢复”而改算术或松门。只在诊断中显式copy context值，不向完整策略注入答案，不把旧context切口当成可部署算法。权重／bias／dtype和原biased Linear表达式保持，真实切片共享原storage，最后恢复N及投影控制。

预注册CPU FP64 `I=(O6-O18)-(N6-N18)`，直接从四份实际投影输出计算，零／非零均保留，不从norm／RMSE／通过数推断，也不解释为策略误差比例。推导行几何M1176／3528来自实际Python操作数，**不是本轮未采集的aten／cuBLAS算法观测**。

预定旧21、新17个BF16数组，加1个FP64交互数组，共39个；包含四格完整输入／输出、原context恢复值与恢复后投影。新实际context与309182／309409保存context另做数值桥接；不相等也不触发backend重扫，更不能倒写R38未捕获的投影前数据。

仍限单A800、20分钟、旧→新两进程，无warmup／profile／时延目标、无新环境／依赖／build／hash。任一自身API／QKV／自值／恢复硬门失败即未获准，保存已采API和切口partial及原异常；不丢frame／camera救完成。R37的策略组合停止规则继续生效，本轮不追PE、不扩后23层、不改生产默认／精度／runtime／sampler／原质量门。

已知未补项也保留：完整`load_inputs→legacy→run`仍无端到端CPU成功链测试；若I已在内部计算、但晚期bundle qualification抛错，失败载荷可能没有这份FP64缓存，四个BF16投影输出仍保留，可不重跑GPU而在CPU精确重算。该失败不生成complete，也不能因此免除正常成功报告所需的完整39数组审计。

**提交状态追加：**`319502` 已由持久回执确认，冻结 `round039/e058161a`，单A800、20分钟、旧→新两进程串行。84项专项、1291项所选回归及隔离导出通过，另有41个subtests／2个skips；裸全仓仍有原20个收集错误。hook半安装泄漏已用第二／第三个注册故障反例RED→GREEN修正，原异常／已有hooks／active flag及后续普通forward均验证。回归曾因`/tmp`配额写盘失败，改用本计划全新临时目录后同选集通过，未删研究数据／改门。真实纯元数据准入确认原round038 namespace与来源绑定，未导入Torch／读权重／使用GPU；这条提交状态不含数值结论，实际API／39数组审计另见第8.15节。

### 8.15 R39结果后追加：context值在该边界局部充分，投影分组无效，仍不是LIVE修复

`319502`／冻结 `e058161a` 完成 `0:0`，无观测错误、重提或取消。7文件／260161742字节完整回传并核对源码、清单／大小和两份单独取得的header。独立CPU auditor未导入producer，核验旧21、新17个BF16数组及1个完整FP64交互数组，所有自身R38 norm1／QKV／API／未观察控制／proj／自回放锚点、实际layout／切片／调用计数、输入／权重只读、同buffer值替换和恢复均通过。

#### 从投影后混合边界，推进到实际context与值交接

本轮真的捕获了投影PREhook／原始功能调用前的context。实际旧FA2与native context相差559个BF16值，maxabs`.00390625`、RMSE`5.784658939572094e-6`、relative L2约`.00002947421`（以旧实际context为参考）。新实际context分别EXACT309182及309409中保存的对应FA2／native context。旧冻结算子结果因此获得了**新的实际API context桥**，不再只是R38给出的同值QKV桥；但不把这些新数据倒写成R38当时捕获的context或旧内部layout。

在保持当前native投影input buffer／layout与原biased Linear的局部切口上：

| 单元 | 对当前native投影后端点 | 对旧实际投影后端点 |
|---|---|---|
| N18：native context＋combined18 | EXACT | 2139值不同、maxabs`.03125` |
| N6：native context＋三个camera6 | EXACT | 2139值不同、maxabs`.03125` |
| O18：旧实际context＋combined18 | 2139值不同、maxabs`.03125` | **EXACT** |
| O6：旧实际context＋三个camera6 | 2139值不同、maxabs`.03125` | **EXACT** |

O6“旧角恢复”是结果前允许失败的**结果项**，不是为了通过而设置的硬条件。本次它通过了；更关键的是 **O18已经恢复旧端点**，没有改变native投影分组。因此在这个固定输入／权重／layout边界，实际旧context值对恢复旧投影后结果**局部充分**；同值下从18改6在两份context上均无数值效果。

预注册CPU FP64 `I=(O6-O18)-(N6-N18)` 的4064256个元素全零，maxabs／L2均0，完整保存数组与独立NumPy公式数值EXACT。它测的是这个投影算子的context×分组有限差分，**与R37的Expert×PROJ2策略交互不是同一因子／边界**；这里I=0不反驳R37 I≠0，也不证明任意输入上的投影分组都无效。

#### 哪个较强说法仍被拒绝

这是显式的中间量答案交接，不是从新LIVE QKV计算旧context的实现。它没有修复native attention，也没有验证完整ViT／七例策略／机器人。不能从559个context差异到2139个投影后差异计算“解释百分比”或一个误差放大倍数。

这轮支持的因果边界比R38更强：context值切口恢复了已测下游端点；但还没有区分context计算里的QK分数、softmax归一化、概率／V累加及具体编译kernel行为。更不能把FA2／Torch的差别直接定性为库bug。

#### 用已有数据排除一个不必要的解释，不再跑backend矩阵

对已有309182／309409保存输出补做CPU比较，共16个context输出：旧Torch2.7的AUTO／强制Flash SDPA与新Torch2.11的AUTO／强制Flash，group6／18全部EXACT本轮实际native context；保存的外部FA2 group6／18全部EXACT本轮实际旧context。

于是，**Torch版本升级不是此输入559个差异的必要条件**：差异在同一2.7环境的外部FA2与TorchSDPA之间已经存在。这个有限端点证据不是任意runtime／backend一致性定理，也不保证kernel内部所有分支相同。外部包的实际版本为`flash_attn 2.7.4.post1`，不是猜测的较新版本；已有309409记录两版Torch的Flash kernel traits为`96,128,64,4`，名称仍不能代替内部算术观测。

#### 下一步只增加有鉴别力的观测

剩余问题回到context计算，而不是继续改变投影分组。下一步先做精确版本官方源码／既有API的只读研究，确认能否在同一实际QKV上取得真实row-LSE归一化输出；任何新观测都必须先让context EXACT本轮实际端点。若API不支持或观测改变路径，就停止该方案，不为此安装新环境／kernel或重扫backend。

相等LSE不能证明分数／概率相同，不等LSE也不能单独区分QK与softmax归约；它若可安全获取，也只是进一步收窄解释。当前没有新GPU作业，不恢复R37组合、不追PE、不扩后层、不改变默认／runtime／精度／sampler／原门限。

### 8.16 R40结果前追加：零权重、同实际QKV的row-LSE观测

R40尚未冻结或提交，没有LSE实测结果。R39将首个投影后差异收窄到实际context，并排除了该输入上投影分组6／18及Torch版本升级的必要性；它仍未说明context计算的哪一阶段产生差异。下一步只增加一个已有kernel可返回的行标量：**自然对数softmax LSE**，不开发或切换kernel，不重扫backend。

精确版本官方源码支持可观测性：外部`flash_attn 2.7.4.post1`公共接口的`return_attn_probs=True`返回context、FP32 LSE和空S_dmask，dropout=0时底层`return_softmax`仍为False；新Torch2.11的已有Flash算子返回九项，0为context、1为LSE，无需开启debug mask。主会话另用系统Python核对了已安装旧wrapper的签名、版本、AST分支与binary stat，未导入Torch／FA2或读权重／使用GPU。**源码可行与stat一致不等于已安装二进制及实际输出获准**。

四个预定目标仅为O6／O18／N6／N18。实际QKV值、camera／frame／head轴序、stride／offset必须保持；旧G18明确是新分组控制，不冒称R39原API形式。每个目标的plain与LSE-capture context均须EXACT对应R39实际context，才解释LSE。缺接口／schema漂移、padding或context不一致即未获准停止，不换backend或安装依赖补救。

| 可能观察到什么 | 可以支持的判断 | 不能推出什么 |
|---|---|---|
| 旧／新存储后的FP32 LSE全部数值或位型相同 | 该行归一化摘要在相应等值定义下相同 | 不证明scores、概率、内部row_sum、PV或context舍入相同，也不排除QK／softmax内部差异 |
| 旧／新部分LSE不同 | 归一化侧存在可观察的行标量差异 | 不单独区分QK累加、scale、exp、归约、log或FP32存储舍入；不证明这些行导致559个context差异 |
| group6与18的LSE相同或不同 | 该固定输入下分组对行摘要的结果 | 不能从已有context相同预设LSE必同，也不能泛化成任意shape的行分组定律 |
| 与CPU FP64数学参考有不同距离 | 相对该明确定义数学参考的端点距离 | 不排名kernel“正确性”，不当作机器人质量、政策误差解释比例或新的验收容差 |

真实LSE预期FP32 `[B,16,196]`，不把196补成256。逻辑head_dim=72用于scale，内部rounded96不改变该逻辑。每组18帧统一后有56448行；context行差异必须先按`[B,N,H,D]→[B,H,N,D]`对齐，再与LSE交叉计数，不能拿flatten后碰巧等长的数组相比较。数值EXACT与实际int32位比较分别报告，包含signed zero，不使用hash。

**零权重约束包括trace里的learned权重。** 不调用旧R39完整`load_inputs`递归，因为它会经R38／Merger reader读Gold trace权重。只复用原snapshot namespace中的纯元数据guard和非权重validators，按角色白名单读取166个R38、38个R39数组及原FP64 I。新完整磁盘CPU成功链核验205次非权重读取，额外learned role、QKV篡改、错误layout都在API consumer前拒绝。原checkpoint／配置仅核对identity／stat，不get_tensor。

计划每engine保存6个BF16＋2个FP32数组，共16个实际数组；再存一个CPU FP64数学LSE参考，共17个。参考将BF16 Q/K提升为FP64，使用实际FP32 scale再提升后的数值做稳定logsumexp，不复现融合kernel的FP32／Tensor Core次序。独立参考公式复核仅容许FP64机器舍入差（绝不用于放宽实际context的EXACT门）。所有raw、调用metadata、只读控制、scope和cleanup均须独立审计；LSE相等／不等均是结果项。

源码中Torch声明`UNFUSE_FMA`、softmax存在条件算术分支，仅动机化“FMA／exp2舍入可能有关”的假设，不证明已安装binary分支或559个差异原因；本轮不改flag、不重建。仍限单A800、20分钟、旧→新两进程，无完整模型、warmup／profile／时延目标。失败时保留已经取得的context／LSE和FP64参考，不把缺数组运行写成complete；R37组合停止决定与所有生产默认／精度／runtime／sampler／原质量门不变。

版本绑定来源：[FA2接口](https://github.com/Dao-AILab/flash-attention/blob/5231d95fe13733fb534c01895f7ea88c6a6c7793/flash_attn/flash_attn_interface.py#L812-L862)、[Torch2.11 schema](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/native_functions.yaml#L15193-L15198)、[返回LSE的softmax归一化](https://github.com/Dao-AILab/flash-attention/blob/e2743ab5b3803bb672b16437ba98a3b1d4576c50/csrc/flash_attn/src/softmax.h#L169-L185)、[Torch构建声明](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/CMakeLists.txt#L329-L334)。

## 9. 研究工具本身也要接受反例和修正

| 问题 | 容易出现的错误解释 | 实际处理与影响 |
|---|---|---|
| RMS历史准入只看张量数／summary | “数量对了就能作为oracle” | 对完整role、dtype／shape／finite、live layout、实际端点做准入；缺失真实role＋补dummy不能通过 |
| R28来源链的历史报告未绑定已记录stat | “最后block输出相同就能证明所有来源没变” | 真实report/stat、实际trace角色／路径先绑定，再允许owner／权重读取；空input-record不能绕过 |
| R29 CPU／GPU同名范围重复计数 | “GPU mean执行失败” | 保留FAILED及首个trace，实际schema反例RED→GREEN；新round只修解析类别，数值门不放宽 |
| R33 legacy checkpoint schema混用 | “checkpoint本身不兼容／缺权重” | 文件stat与目录／config完整记录分别验证；真实结构正例先RED，bytes／mtime／config等负例保持拒绝，真实headers再CPU重放 |
| SSH观测连续失败 | “Slurm作业失败，重新sbatch” | 先用无实验FS／无Slurm的SSH探测隔离传输问题；保留原回执，退避恢复同一作业，绝不因此重复提交 |
| 大数组传输超时 | “只拿summary也足够” | 有界压缩传输、完整清单／大小／source／单独header核对，再独立读全数组；没有为记录研究而重跑GPU |
| R38把Gold的package来源字段当成policy文件 | “绿色临时fixture就代表真实历史schema相容” | 核对实际Gold与producer：字段是`__init__.py`，严格绑定round007 package后再派生policy源；真实schema正例RED→GREEN，旧错误文件／错误round仍拒绝，未改历史数据 |
| R38观察完成后的Gold锚点／最终同步失败 | “拒绝complete就足够，不必保留已采数组” | 失败仍须保留未获准raw与cleanup／layout状态；注入锚点、同步及证据写盘失败，保持原异常与`qualified=False`，绝不生成成功报告；修正发生在GPU提交前 |
| R39第二／第三个hook安装失败 | “把三个注册调用放进列表赋值，finally就能统一移除” | 实际列表赋值要等全部右侧成功，先前handle可能丢失并误报cleanup成功；两种注册故障反例RED→GREEN，改为每成功一次立即记录handle，并验证已有hooks／active flag及之后普通forward不受旧observer影响 |
| R39回归写临时safetensors出现磁盘配额错误 | “剩余全局空间就说明没有存储限制，或把7项写盘失败当成模型回归” | `/tmp`挂载启用usrquota且仍有全局空间／inode；不删原始产物、不改配额、不改测试，改用本计划独立新临时目录后同选集通过。原失败名／错误与处置另留观察摘要，不冒充恢复了已被复验覆盖的原完整stdout |
| R40把ATen schema中的Tensor当成每个Python槽位都必为Tensor对象 | “九个返回的声明类型已经足以构造真实fixture” | 对照同Git CUDA实现与随包meta：dense cum_seq_q/k是未定义Tensor，Python为None；只在这两个槽位支持并记录，context／LSE／debug仍必须满足原硬门。真实meta占位贯穿parser和完整validator的正例RED→GREEN，不靠空Tensor fixture掩盖契约 |
| R40解析不支持的LSE返回时丢失刚取得的数组 | “parser抛错前没有资格，所以不必先保存返回” | 非空debug、padded／NaN LSE故障反例RED→GREEN：资格校验前按原shape／dtype快照context和LSE，保存未获准返回metadata及after-QKV；不归一化或裁剪异常数据，次级快照失败只能附注原异常，不生成complete |

这些修正提高的是证据可信度，不是模型质量。RED→GREEN说明特定回归反例被覆盖；它不是跨设备、所有输入或全部主流程的证明。独立审查提供第二视角，但主执行者仍负责核查结果和最终原始审计。

## 10. 当前可以站住的解释与仍开放的问题

### 10.1 已得到限定范围内支持

1. 减少host规划／graph提交可以省时，但不会删除逻辑GPU数学工作，native也未必最快。
2. 扩大请求范围能推翻较小范围的采样选择；数值预算不随报告范围自动泛化。
3. PROJ1调用几何影响受测算术路径；HWD局部EXACT并不保证整策略通过。
4. QKV拆分与prefix原序有非零联合效应；单项不充分不表示组合无效。
5. 在16组相同RMS输入上，首差异位于FP32 mean，旧variance对所测下游局部充分；显式输入驱动加法树可重现两版mean。
6. 该输入驱动RMS在联合固定GoldKV／外部轨迹条件下恢复完整32层Expert，但native-policy仍6/7。
7. 实际native prefix的image embedding在prefill前已不同；VLM不是一个无上游混杂的独立解释对象。
8. PROJ2的受测biased调用按M49／M147恢复不同端点，显式拆bias不复现原端点；这是条件化的执行路径效应，不是通用行数定律。
9. 单改PROJ2在本七例使state过门、noise新失败，6/7计数不变会掩盖失败迁移；数值质量改善不能只用一个聚合数字叙述。
10. 在一个共享视觉context的Merger输出L2边界，三端点向量验证了局部净抵消；该结论不等于最终action误差的机制或解释比例。
11. Expert bundle与PROJ2的首步／最终action有限差分交互在七例均非零，但both仅5/7且新增history失败；非加性不等于有利，组合搜索已按预注册规则结束。
12. 同一实际视觉输入的四块前缀已精确锚定，首不同边界定位在block0投影后attention，前置norm1／QKV一致；后续模块差异不能再被当作无输入混杂的独立原因。
13. 在该block0固定边界，旧实际context值足以恢复旧投影后端点，同值投影分组6／18无效且I=0；仍只是诊断交接，不是LIVE算法。保存数据还表明此context gap不需要Torch版本升级才出现。

### 10.2 尚未验证，不能写成结论

- 每项源代码变动对ULP／动作误差贡献多少；没有加性“解释误差百分比”。
- 某次mean调用具体读取了哪个runtime `ReduceConfig`字段／编译分支；相同kernel符号不是证明。
- R33失败是否主要来自某个vision／Merger／VLM因素或完整策略误差抵消；R34/R35已识别局部PROJ2项，R36证明单改仍未统一过门。Merger边界已测出的局部抵消不能代替这一开放问题。
- 显式旧树是否适用于任意形状、布局、数据或更宽norm；当前不扩大支持范围。
- 多kernel数值原型的加速价值；它没有作为性能候选验收。
- 未参与筛选的真实任务／机器人质量、跨硬件／checkpoint泛化、任何自动生产默认变更。

## 11. 后续每轮追加的最小记录模板

```text
轮次／job／冻结源码：
研究问题和上一轮触发它的反例：
已观测现象（明确输出域／输入／runtime／几何）：
候选原因H1/H2/...（注明猜测，写出能区分它们的预测）：
固定项与唯一干预项；如果多因素组合，明确组合条件：
未改控制的准入、实际API端点与失败时停止条件：
预先声明的候选／门限／资源界限：
实际执行与产物（NFE、调用数、完整数组、cleanup、缺失项）：
独立复核结果：
支持／否定／仍分不清的解释；不要把相关性升级为充分性：
与原猜测不同的地方和为何调整下一步：
决定：保留／拒绝／只作诊断；默认是否改变：
下一项最小判别实验与未解决风险：
证据提交／公开代码入口；原始私有产物不公开：
```

## 12. 证据入口与复核方式

- [逐轮实测记录](../vla_autotune_experiments.md)：各job、冻结源码、正负结果、数值与限制的主索引。
- [实现与结果总报告](2026-11-09-vla-autotune-report.zh-CN.md)：第11节持续补充后续轮次；早期摘要有其历史时间边界。
- [Examples README](../../examples/vla_autotune/README.md)：诊断入口、API范围及非生产性质。
- [执行层契约](../vla_backend.md)与[autotune控制层](../vla_autotune.md)：输出域、实际NFE、身份／质量入账边界。
- 固定结果提交：R26 `52e03612`、R27 `df83cb94`、R28 `71406637`、源码机制 `eb59d6a8`、失败采集 `682f9f14`、R30 `f04a95e3`、R31 `398bdecb`、R32 `200e6560`、R33 `5f084d43`、R34 `6930a674`、R35 `3b76eae8`；R36实现 `a5078137`，实测与边界分析见第8.9节。
- R34实现固定到 [`c990f271`](https://github.com/20261023/FlashRT/tree/c990f271/examples/vla_autotune/hy_trace)；实现存在／测试通过不代表作业结果已通过。

完整输入、权重、动作／中间数组、raw trace、环境和私有路径不随本文公开；公开记录给出可核对的job／源码／聚合结果，不上传凭据、个人信息或大产物。本文不新增内容哈希工作。
