# VLA Autotune 研究过程记录：现象、假设、验证与自我修正

**迭代起点：**2026-11-09；**本记录整理日期：**2026-11-12。

**覆盖：**π0、π0.5、FastWAM，以及重点展开的 Hy 数值一致性定位。

**记录边界：**已核验结果更新至 R35。首版 `5d515b0a` 的R34预测及 `32ea141e` 的R35七臂预测保留，分别在第8.4—8.5节和第8.7节追加结果与解释修正。

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

R36目前尚未冻结／提交。只改变最后PROJ2的biased Linear：B3→三个B1，保持HWD PROJ1、原生Expert QKV／attention顺序／RMS、native vision/prefill、Euler10、noise和原质量门。**不混入R33三项Expert改动。**

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

## 9. 研究工具本身也要接受反例和修正

| 问题 | 容易出现的错误解释 | 实际处理与影响 |
|---|---|---|
| RMS历史准入只看张量数／summary | “数量对了就能作为oracle” | 对完整role、dtype／shape／finite、live layout、实际端点做准入；缺失真实role＋补dummy不能通过 |
| R28来源链的历史报告未绑定已记录stat | “最后block输出相同就能证明所有来源没变” | 真实report/stat、实际trace角色／路径先绑定，再允许owner／权重读取；空input-record不能绕过 |
| R29 CPU／GPU同名范围重复计数 | “GPU mean执行失败” | 保留FAILED及首个trace，实际schema反例RED→GREEN；新round只修解析类别，数值门不放宽 |
| R33 legacy checkpoint schema混用 | “checkpoint本身不兼容／缺权重” | 文件stat与目录／config完整记录分别验证；真实结构正例先RED，bytes／mtime／config等负例保持拒绝，真实headers再CPU重放 |
| SSH观测连续失败 | “Slurm作业失败，重新sbatch” | 先用无实验FS／无Slurm的SSH探测隔离传输问题；保留原回执，退避恢复同一作业，绝不因此重复提交 |
| 大数组传输超时 | “只拿summary也足够” | 有界压缩传输、完整清单／大小／source／单独header核对，再独立读全数组；没有为记录研究而重跑GPU |

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

### 10.2 尚未验证，不能写成结论

- 每项源代码变动对ULP／动作误差贡献多少；没有加性“解释误差百分比”。
- 某次mean调用具体读取了哪个runtime `ReduceConfig`字段／编译分支；相同kernel符号不是证明。
- R33失败是否主要来自某个vision／Merger／VLM因素或误差抵消；R34专门用于区分其中部分解释。
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
- 固定结果提交：R26 `52e03612`、R27 `df83cb94`、R28 `71406637`、源码机制 `eb59d6a8`、失败采集 `682f9f14`、R30 `f04a95e3`、R31 `398bdecb`、R32 `200e6560`、R33 `5f084d43`。
- R34实现固定到 [`c990f271`](https://github.com/20261023/FlashRT/tree/c990f271/examples/vla_autotune/hy_trace)；实现存在／测试通过不代表作业结果已通过。

完整输入、权重、动作／中间数组、raw trace、环境和私有路径不随本文公开；公开记录给出可核对的job／源码／聚合结果，不上传凭据、个人信息或大产物。本文不新增内容哈希工作。
