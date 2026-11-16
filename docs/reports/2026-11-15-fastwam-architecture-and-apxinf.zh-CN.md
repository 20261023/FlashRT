# FastWAM 架构、FlashRT 适配差距与 ApxInf 接入范围

**范围：** 本文解释实际运行的 FastWAM LIBERO 配置，比较 π0／π0.5／Hy 的结构，
区分已有计时与尚未测量的优化假设。新增实跑已取得**FastWAM原生ApxInf研究实现
完整public policy约215.34 ms**、**DM0.5作者default约277.66 ms**及**DM0.5 ApxInf
完整public policy约149.35 ms**。2026-11-16已核对全部正式作业终态和保存输出；
研究实现不等于公开引擎已有支持或完成部署验收。

## 1. 先纠正“FastWAM 适配后很慢”的范围

同一 A800 作业 `327380`，BF16、Euler20、shift5、H32、cached-T5、八案例／80 个正式样本：

| 路径 | Median ms | 相对旧项目 graph |
|---|---:|---|
| 旧项目 Torch raw-denoise graph | 250.9826 | 基准 |
| **本项目 compiled prefill＋denoise** | **137.6503** | 降低 **45.1554%**，延迟比 **1.8233×** |
| 作者原始 compiled API，同为20步 | 142.4363 | 本轮新项目低3.3601% |
| partial FlashRT cached Linear＋raw graph | 263.3574 | 比旧项目 graph 高4.9305% |

这些结果来自 **FlashRT `227993d5`／round044／job327380**，不是随后拉取的
stage-aware harness 已复测结果。完整输入与计时范围见[延迟大表](2026-11-14-vla-latency-comparison.zh-CN.md)。

新路径80份保存输出与对应作者 compiled20 输出逐元素相同；相对旧 graph 的最坏
relative L2 为0.0393335742、最低 cosine 为0.9992527670，在原八案例0.05／0.999代理
门槛内。它不是闭环成功率或任意输入质量证明。冷准备约24.60秒，不混入稳态毫秒。

因此，**旧 graph 和 partial FlashRT 的性能差距确实存在，但不能再说当前已选路径
仍比作者 compiled 慢约1.8倍**。同样，旧 eager804.69→graph255.07 ms 的约3.15×
收益，也不能改写成击败作者 compiled 的收益。

## 2. 四种架构不是同一个 Transformer 换维度

以下为本轮具体配置，不是所有模型变体的通用规格。

| 结构 | FastWAM LIBERO | π0 | π0.5 | Hy semantic-BF16 |
|---|---|---|---|---|
| 视觉入口 | **WanVAE38＋Conv3d patch＋VideoDiT** | SigLIP＋PaliGemma | 同π0 | 历史时空ViT／MEM＋merger＋VLM |
| 视觉／前缀主干 | VideoDiT **30层，D3072，FFN14336** | SigLIP27层D1152；Gemma18层D2048 | 同π0 | ViT27层D1152；VLM32层D2048 |
| 动作主干 | **30层D1024，FFN4096** | 18层D1024，FFN4096 | 同π0 | 32层D1024，FFN2048 |
| 动作Q／KV | **24Q／24KV×128** | 8Q／1KV×256 | 同π0 | 16Q／4KV×128 |
| 条件注入 | 视频／动作混合注意力，**另有每层文本／本体cross-attention** | 前缀KV与动作KV共同参与注意力 | 同π0 | VLM前缀KV＋动作后缀 |
| FFN | **无gate的GELU-tanh** | GEGLU | GEGLU | SwiGLU |
| norm／时间 | LayerNorm；六路shift／scale／gate | RMSNorm；action/time拼接；连续state token | AdaRMSNorm；离散state prompt | RMSNorm；action/time拼接；state token |
| QK与RoPE | **整3072维RMSNorm，再相邻pair RoPE** | Gemma式RoPE | 同π0 | **RoPE后按head128做QK-norm** |
| 前缀缓存 | 30份K/V，每份本例`[1,98,3072]` | 固定1KV-head布局 | 同π0 | 本路径展开到16 heads并原位写动作窗口 |

关键区别是 **FastWAM 动作 hidden=1024，但每个Q/K/V的投影宽度是3072**，
不能把 attention 当作1024宽或1KV-head处理。按当前源码静态计数，每NFE约有307次
Linear调用；这说明移植面大，**不等于307个GPU kernel，更不能直接换算成耗时**。

视觉也不是替换一下SigLIP：WanVAE采用因果Conv3d、通道RMS归一化、重采样及其缓存
语义。本例只有当前帧；不能把单帧的潜在简化未经核对就推广到多帧／历史路径。

### 主要源码证据

- [FastWAM固定上游](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)：
  `src/fastwam/models/wan22/{wan_video_vae,wan_video_dit,action_dit,mot,fastwam}.py`。
  `action_dit.py:73–98,312–331` 为动作与时间入口；`wan_video_dit.py:54–60,178–219`
  为QK归一化／RoPE／cross-attention；`mot.py:474–578` 为缓存与混合注意力。
- π维度与入口：[`pi0/pipeline_rtx.py`](../../flash_rt/models/pi0/pipeline_rtx.py)、
  [`pi05/pipeline_rtx.py`](../../flash_rt/models/pi05/pipeline_rtx.py)；GEGLU语义见
  [`activation.cu`](../../csrc/kernels/activation.cu)，不沿用调用处旧“SiLU”注释。
- Hy维度／执行：[`_hyvla_thor_spec.py`](../../flash_rt/frontends/torch/_hyvla_thor_spec.py)、
  [`hyvla/pipeline_thor.py`](../../flash_rt/models/hyvla/pipeline_thor.py)。

## 3. 目前 FlashRT 到底接入了哪一部分

当前 [`registry.py`](../../flash_rt/vla/registry.py) 的 FastWAM `backend="flashrt"`
只选择 native Linear；不是完整模型原生移植。VAE、patch Conv3d、attention、norm、
RoPE、modulation等仍为上游Torch计算，本体投影也仍为Torch。

| Stage | 请求内工作 |
|---|---|
| `host.input` | 图像、context、proprio、noise校验／上传 |
| `vision.encode` | WanVAE编码当前图像 |
| `prefix.prefill` | 加入proprio条件，VideoDiT prepare＋30层prefill，生成视频KV |
| `denoise.velocity` | 每NFE的action prepare、30层混合注意力、文本cross-attention和FFN |
| `denoise.routing` | Euler multiply-then-add |
| `host.output` | 有限性检查和完整FP32 CPU动作 |

入口代码见 [`fastwam/pipeline.py`](../../flash_rt/models/fastwam/pipeline.py)。
现有condition区间合并了输入／VAE／prefill；**当前没有137.65 ms的细分占比，不能
按结构推测给出各阶段百分数**。

### 已有编译路径为什么改善很大

旧路径主要捕获raw denoise的CUDA Graph，prefill仍eager；新路径同时compile
prefill与denoise，沿用作者已有的编译思路。Graph降低host重复提交，不自动等于
编译融合。新路径的已测整体收益不能全部归给GEMM、缓存或某一个融合算子。

### 为什么不能直接搬π／Hy融合链

1. π0.5 AdaRMS不减均值，Fast的LayerNorm要减均值；不能换名字复用。
2. Hy的rotate-half→head128 QK-norm→KV写入，与Fast整3072维QK-norm→相邻pair
   RoPE在顺序、归一化范围和旋转布局上均不同。
3. Gemma GEGLU的gate×up与Fast无门控GELU FFN不同；π固定1KV-head缓存也不能
   用于Fast的24KV-head。

通用GEMM、GELU或LayerNorm原语可以复用；**模型级融合链只有语义契约吻合才能复用**。

## 4. 规划、布局和copy：哪些已确认，哪些仍是假设

- legacy GEMM每次创建descriptor／查询heuristic，而本次partial已经使用cached-plan。
  因此不能继续把所有差距解释为“每个GEMM重新规划”。
- Linear wrapper的dtype／contiguous／分配／指针调用可能有成本；`contiguous()`
  **不必然发生复制**。NT接口也避免了为了接口额外保留整塔转置权重。
- Graph replay不重新执行307次Python forward或host planning；相关设备GEMM、cast、
  concat等仍会执行。prefill在旧路径中仍是eager。
- 视频KV已按请求缓存，不是每NFE从host重传全部视频KV；但raw body每层仍concat
  视频／动作K与V。

证据见 [`native_linear.py`](../../flash_rt/models/fastwam/native_linear.py)、
[`velocity_graph.py`](../../flash_rt/models/fastwam/velocity_graph.py)、
[`gemm_runner.cu`](../../csrc/gemm/gemm_runner.cu) 和
[`bf16_nt_plan.cpp`](../../csrc/gemm/bf16_nt_plan.cpp)。

**结论：** GEMM算法／epilogue、eager包装、中间量和布局复制均是合理解释，但没有
新profiling，就不能把partial额外12.37 ms或编译节省113.33 ms全分摊给某一项。

## 5. 接下来最明确的原生接缝

以下针对FlashRT侧，是源码支持的候选，不是已经取得单项收益的方法。独立ApxInf
候选已实现请求级文本KV并完成整体计时，但没有做单项消融，不能由此证明缓存收益：

1. **请求级文本cross-attention K/V缓存。** 视频KV已有缓存；action context embedding
   以及各层context K/V和K-norm仍每NFE重算。应在append proprio之后按请求准备，
   不能仅按prompt跨请求复用。约需45 MiB BF16附加KV，实际收益尚未测量。
2. **ActionDiT固定布局和持久工作区。** 把视频KV一次放入含动作窗口的固定布局，
   每层只更新动作窗口；在加载时打包QKV，避免每层反复concat。必须保留BF16舍入
   边界与KV所有权，不能只看shape相同。
3. **保留独立prefill边界。** 当前compiled prefill已有效；是否进一步原生化应由阶段
   证据决定，不因要“全原生”而先盲目重写完整VAE。

新配置保持opt-in，不改既有默认；FlashRT候选不得修改固定的
`flash_rt/vla/autotune/`测量框架。

## 6. ApxInf：工具登记、历史移植和新模型不是一回事

公开RLinf/APXinf-robo固定版本`31d6225`所锁引擎为`7f37642`；不能混用后续引擎
`b05b409`的接口。公开family中没有FastWAM或DM0.5。当前FlashRT
[`engine.apxinf`](../../flash_rt/vla/autotune/tools/external/apxinf.py)仅登记π0.5
外部baseline、状态`needs_setup`，不证明其他模型已经实现或运行。

旧私有Hy移植已找到，源码存在未提交的CUDA／Rust／policy改动。新任务保存了这些
实际字节的独立源码快照，旧环境／源码保持只读；只checkout其Git HEAD会漏掉移植。

历史证据需单列：

| 模型／历史配置 | ApxInf历史结果 | 不能直接当作本轮同边界结果的原因 |
|---|---|---|
| π0.5，job229581 | BF16，H50/E10，143 prepared tokens；39.16 ms | prepared输入到动作，排除tokenization／state normalization／公共decoding |
| Hy，job231955／233770 | raw-policy约268–271 ms；device core133.65 ms | 旧profile／环境；core不等于完整policy；不是本轮表内重新配对 |
| π0独立matched | NR | 没有可核验独立记录 |
| FastWAM新family，job329256 | **public policy 215.3437 ms** | 新增原生研究实现；不同allocation／NHWC接口，未完成部署验收，不算纯engine加速比 |
| DM0.5新family，job329543 | **public policy 149.3522 ms** | 原生研究实现；四案例范围内，provided-noise／作者RNG及allocation不同，非纯engine比值 |

旧Hy最终七组输出与官方一致，但旧FlashRT development对照仅1/7满足同一数值gate；
不能据旧104.04 ms development点宣称公平纯engine比值。历史ApxInf约270.9 ms与
作者约804.3 ms的结果也不应冒称本轮新ApxInf实测。

真正ApxInf接入需要family独立的权重映射、模型执行、请求／workspace／graph所有权
及公共policy。仅包装Torch、只替换Linear或只测action core，必须按真实范围命名，
不能填入“完整ApxInf”列。

### 新原生FastWAM结果不是性能胜点

本轮独立快照新增WanVAE38、VideoDiT30、ActionDiT30×20及whole-model graph，
模型计算调用Rust安全CUDA接口；公共入口为AutoPolicy与现有ModelRunner binding。
首次作业329215在I8输入文件解析处失败，改用FP32磁盘mask／native I8后，329256
完成native及public policy各80次正式推理。不是Torch或TensorRT模型包装。

graph提交到完成wall中位数212.5183 ms、prepared输入上传至CPU输出212.8693 ms、
完整public policy215.3437 ms；只有最后一项进入主表。对应native／policy保存动作
逐元素相同；与作者327380已保存参考相比，最坏relative L2为0.0116095123、最低
cosine为0.9999326110，满足原八案例代理范围，不代表闭环、无损或未运行的算子门槛。

此实现目前慢于作者compiled约142 ms和项目compiled约138 ms，保持研究接口和
`deployable=False`，不替换已选方法。固定NHWC图像布局在计时前准备，历史接口为
NCHW，且不是同allocation；不给纯engine比值。已有请求级文本KV缓存也不意味着
整个执行栈一定更快。空间／attention原语目前为未调优组合，尚无新的阶段profile，
不能把剩余差距定量归给其中一项。详细边界见[大表新增结果](2026-11-14-vla-latency-comparison.zh-CN.md#新增dm05作者参考与fastwam-apxinf实跑)。

在**这份研究实现**中，源码可见的具体性能债是：

1. **不等长／masked action attention仍是标量baseline。** 同长度、无mask分支复用
   现有FA2；32-query／130-key的action self-attention及带mask文本cross-attention
   则使用FP32 score／probability和逐key PV循环，不是tiled Tensor-Core／FlashAttention。
   这描述本候选的实际选择，不代表整个ApxInf没有其他attention能力。
2. **缓存不等于没有KV拼接。** video KV和文本cross-KV均已按请求缓存，但30层×20步
   的600个action block仍分别concat K／V。仅按源码输出shape计数，
   `600 × 2 × 130 × 3072 × 2 = 958,464,000 bytes/request`。
   这是逻辑输出字节数，**不是测得的DRAM流量、峰值带宽或阶段耗时**。
3. **VAE显式im2col，许多BF16边界未融合。** 卷积为im2col→GEMM→bias，norm、
   modulation、RMS、RoPE、激活和残差保留多个操作；whole-model Graph只改变提交方式，
   不自动消除这些设备计算／访存。
4. **GEMM autotune关闭，arena没有按存活期复用。** 已记录的4,595,864,768字节是arena
   consumption，不是实测活跃峰值或总显存。公共入口还包含检查、复制、同步及输出传输，
   当前数据没有把约215 ms分摊给这些操作。

因此，“已使用原生kernel／whole-model graph／文本KV缓存”并不足以推出胜过作者compiled。
本轮不新增profiling、算子测试或消融，也不承诺某一项修改能消除全部差距。

源码增量与身份说明已整理到
[`examples/vla_autotune/apxinf/fastwam-sm80/`](../../examples/vla_autotune/apxinf/fastwam-sm80/README.md)。
公开补丁中24份运行时代码与实测字节相同；7份后补文档／helper未执行。它仍依赖
含未提交改动的私有Hy foundation，**不是从干净上游提交可独立复建的完整源码包**。
公开增量SHA为`79baaac2…fc6b450`，与实测私有增量包身份分开；没有打包权重、环境、
二进制或原始动作，也没有为打包新增测试／模型运行。

## 7. 新加入的DM0.5具体是什么

本任务固定为作者[OpenDM源码`03f7991`](https://github.com/dexmal/opendm/tree/03f799120d425ead51d9482bb1da950b6dabade9)
与[`Dexmal/DM05`基础权重`5cd1873`](https://huggingface.co/Dexmal/DM05/tree/5cd18734814abb075a9ccfd9ad6d16777b5cf10e)：

- 三相机，448×448，无历史，chunk50，公共14维／内部32维，10步。
- SigLIP27层／D1152；Gemma3前缀34层／D2560／FFN10240；动作expert34层／D1024／FFN4096。
- 8Q／4KV×256；Gemma3 QK-norm、交替sliding/full attention、额外post norms；
  动作采用时间条件AdaRMS及残差gate，不能直接视为π0.5的18层执行图。
- 基础checkpoint约5.83B参数、11.66 GB BF16权重。**DM0.5不是0.5B参数模型**，
  也不能与LIBERO两图／H10／公共7维版本混写。
- 源码Apache-2.0，权重使用Gemma许可。A800无原生FP8，作者TensorRT fast／FP8宣传
  路径不能未经实现和测量直接套用。

作者default／vision SDPA／无Liger参考已由329331取得：四case／40份完整`[50,14]`
FP32 CPU动作，policy中位数277.6579 ms，内部模型267.1751 ms，加载准备17.0973 s；
实际使用两个suffix graph profile、无禁用项。源码／权重身份核对后才运行，首轮
329232的PyArrow／NumPy导入失败保留，不改写为模型或质量失败。

原生ApxInf的329543现已完成完整vision／prefix／10步action和公共policy，沿作者实际
热态BF16 suffix算术分支。40份完整public14输出相对作者参考的最坏relative L2为
0.0093243955、最低cosine为0.9999608552；全体有限且同case重复一致。完整policy
中位数149.3522 ms、P95 150.1001 ms。相对作者277.6579 ms的名义比值为1.8591×，
但ApxInf使用预先保存的准确noise、作者在请求内生成CUDA noise，且为不同allocation，
不称严格纯engine加速。ApxInf公共setup45.1512 s，比作者17.0973 s慢，不能只报稳态优势。
这些结果没有新增测试、算子门槛或闭环评测，仅代表四组正式输入作用域。

DM0.5外部ApxInf源码增量见
[`examples/vla_autotune/apxinf/dm05-sm80/`](../../examples/vla_autotune/apxinf/dm05-sm80/README.md)。
它的23份运行时代码保留实测身份；后补helper未执行，且仍依赖私有Hy foundation。
这是外部引擎的研究源码，不能与下面单独完成的本项目DM0.5／FlashRT身份混用。

### DM0.5项目与FlashRT基线另已完成

补接入后的正式作业334487通过项目公共VLA API取得Torch全图194.2449 ms、FlashRT
混合算子206.0562 ms及Torch bookend194.1798 ms，各40份完整CPU public14动作。
最坏relative L2分别0.0077728340／0.0086970135，最低cosine分别0.9999727883／
0.9999650775；原代理门槛未变，project/bookend输出逐元素相同、延迟漂移约0.03354%。

新FlashRT路径实际使用cached BF16 GEMM、FA2和strict GELU，但norm、RoPE、
masked prefix attention等仍为Torch，明确是partial native，不是完整native迁移。
同allocation下它比项目Torch慢6.0806%，不是winner。作者277.6579 ms与外部ApxInf
149.3522 ms仍属于之前独立allocation和原接口，不混用其计时／源码身份。
项目源码与用法见[`docs/dm05_usage.md`](../dm05_usage.md)，全部边界及Token更新见大表。

后续只针对项目方法做编译／数据流优化，335362同allocation配对结果已达到用户目标：
项目compiled96.1659 ms，未改动FlashRT206.3909 ms，未改动ApxInf151.1550 ms／
bookend150.7088 ms；对同作业更快APX端仍低36.19095%。历史基线不改写。
方法和独立阶段观测见[配对优化报告](2026-11-16-dm05-competitive-optimization.zh-CN.md)；
该组合主要改善velocity和prefix，不能把组合收益归因给未经消融的单个改动。

## 8. 本文证据限制

架构／工具能力核对基于拉取后的`e98653d4`代码；Fast延迟仍绑定`227993d5`的已保存
正式样本。新增FastWAM原生ApxInf绑定独立源码快照／329256，DM0.5作者参考绑定
03f7991／329331；DM0.5原生ApxInf绑定另一独立源码快照／329543。按照用户要求，
没有新增测试、smoke、profiling或额外评审；数值结论只来自已保存正式输出的算术。
两条原生候选均完成了本轮完整模型实跑，仍不宣称通用shape／eager分支或部署验收通过。
