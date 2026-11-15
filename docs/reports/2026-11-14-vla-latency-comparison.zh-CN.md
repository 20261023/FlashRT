# 四模型延迟对比：本项目已选方法与官方 Native

**状态：四模型A800主对比已齐，2026-11-15。** 四个本项目已选配置与四个作者原实现均取得
实际延迟。附加的π0.5 Native8步点因二次编译触及进程总预算而未完成，明确保留NR；
不为此再排作业，也不把已有10步结果或失败记录改写掉。文献数字只作参考，不用于异卡比值。

**新增请求已完成：** 作业`327380`以COMPLETED 0:0结束，四个FlashRT基线列已补齐。
FastWAM保持20步／BF16／shift5／H32／八案例，接入prefill＋denoise编译后，
同卡中位数 **250.98 → 137.65 ms，降低45.16%（1.8233×）**；作者同20步compiled为
142.44 ms。这一轮略低3.36%，不外推为跨硬件／任务的普遍优势。

机器可读：[CSV](2026-11-14-vla-latency-comparison.csv)／[JSON](2026-11-14-vla-latency-comparison.json)。
共19条点记录：18条完成计时、1条未完成，保留旧Fast路径和旧Native对照的历史行；
JSON另外保留作者文献参考与限制说明，不含私有输入／路径。

## A800实测大表

全部使用 **NVIDIA A800-SXM4-80GB**。每个模型的项目／Native配对在同一张卡、同一
allocation内完成：π0为`326708`，π0.5为`326869`，Hy为`326641`，Fast新路径和强对照均为`327380`。
新增FlashRT列四行都来自`327380`；π0／π0.5／Hy这一列与各自旧主配对不是同allocation，
只同为A800型号，不据此新增跨作业加速比。Fast三条当前路径在同一张卡同一作业内配对。
每个方法8次warmup。表内均为毫秒；Native“10步”仅指默认采样步数，其他输入几何按本轮固定。

| 模型 | 本项目当前已选方法 | 本项目Median / Mean / P95 | FlashRT固定步数基线Median / Mean / P95 | 作者Native，10步Median / Mean / P95 | 作者Native，相同步数Median / Mean / P95 | 样本数（项目／FlashRT／Native） | 官方文献参考，非同卡基线 |
|---|---|---:|---:|---:|---:|---:|---|
| π0 | FlashRT FP16 Euler2，H50、3view | **26.1164 / 26.1463 / 26.4021** | FP16 Euler10：**40.3384 / 40.4244 / 40.7971** | JAX BF16：**69.3703 / 69.4761 / 70.1619** | Euler2 BF16：**48.3255 / 48.3931 / 48.9294** | 50／50／50 | 73 ms，RTX4090、Euler10 |
| π0.5 | FlashRT BF16 Euler8，H50、3view | **39.2392 / 39.3056 / 39.5815** | BF16 Euler10：**41.9108 / 41.9764 / 42.3033** | 作者PyTorch compiled：**59.4795 / 59.5625 / 59.8169** | Euler8：**NR，二次编译触及预算** | 40／40／40 | 76 ms，RTX4090、BF16、5步、GPU-only mean |
| FastWAM | 本项目Torch compiled **prefill＋denoise**，BF16 Euler20、H32、shift5、cached-T5 | **137.6503 / 137.7422 / 138.1893** | partial FlashRT Linear cached＋graph Euler20：**263.3574 / 263.4317 / 263.9883** | 作者compiled：**85.7984 / 85.9070 / 86.5387** | Euler20 compiled：**142.4363 / 142.5240 / 143.0105** | 80／80／80 | 官方compiled 110 ms／4090；论文190 ms／5090D V2 |
| Hy | FlashRT semantic-BF16 Euler10 graph **基线** | **123.9355 / 123.7039 / 129.6583** | 相同E10配置重新计时：**121.2805 / 122.1176 / 128.2098** | **804.2853 / 802.2074 / 810.7750** | 同左，双方Euler10 | 140／70／140 | 作者未报告完整推理延迟 |

新增FlashRT列指本fork当前可用的明确执行绑定，不是官方Native。FastWAM这一列仅为部分
Linear换成FlashRT GEMM，其他算子仍使用上游Torch，不能冒充完整FlashRT移植；采用已有
cached-plan＋raw graph路径，不故意用更慢的legacy规划充当对照。Hy基线与当前已选配置
本来相同，不把独立重测的自然抖动说成算法优化。

**历史对照仍保留：** 优化前`326708`中，作者compiled20步141.39 ms、旧项目Torch graph
249.83 ms，旧方法约慢1.77×，该结论不被抹掉。新`327380`是在保持20步等条件下补齐编译
后的新结果，不是把旧记录改名或用新数据回填历史。

### FastWAM新执行路径的实际收益（327380，同卡同负载）

| 路径 | NFE | Median ms | Mean ms | P95 ms |
|---|---:|---:|---:|---:|
| 旧项目Torch raw-denoise graph | 20 | 250.9826 | 251.0515 | 251.9099 |
| **新项目compiled prefill＋denoise** | **20** | **137.6503** | **137.7422** | **138.1893** |
| 作者原始compiled API | 20 | 142.4363 | 142.5240 | 143.0105 |
| partial FlashRT Linear cached＋graph | 20 | 263.3574 | 263.4317 | 263.9883 |

新路径相对本轮旧路径降低45.1554%中位延迟、延迟比1.8233×；相对作者compiled在本轮低3.3601%。
不把旧800 ms eager当主要分母，不改步数／精度／几何，不声称新写了更快的原生FlashRT kernel。
这次是将作者已有编译思想接进本项目固定noise／采样接口，并保留预计算时间表等现有编排。
作者API含按seed生成等值CPU noise，新接口接受显式noise；小幅差异属于整套接口编排，不是纯kernel收益。

由**同批已保存的正式输出**计算：相对旧graph最坏relative L2为0.0393335742、最低cosine为
0.9992527670，满足原定public7 L2≤0.05／cosine≥0.999；最大绝对差0.2265625，最坏点为
`context-minus0p0625`数值压力案例。这不是无损／闭环任务成功证明，阈值未调整。
新路径80份保存输出与本轮作者compiled20步的80份对应输出**逐元素相同**；该结论只针对这批输入。
上述计算没有新增GPU请求或额外质量测试。

启用方式：在已有FastWAM Torch配置中设`num_steps=20`、
`options={..., "use_cuda_graph": false, "compile_action_infer": true}`。
此flag仍默认关闭，避免改变现有调用／冷启动行为；新已选配置显式开启，不和partial-native Linear混用。
编译是冷成本：本轮新路径prepare阶段约24.60 s（含编译／首轮准备），不在137.65 ms稳态统计内。

### 旧3.2×为什么不代表击败官方compiled（优化前的口径说明）

旧[迭代报告§6.1](2026-11-09-vla-autotune-report.zh-CN.md#61-library-graph-保持性)的
作业`309187`同为20步：**Torch eager804.6938 → raw-head graph255.0711 ms，约3.15×**。
本次优化前约249.83 ms与旧graph点接近，不是突然退回804 ms；但作者compiled20步141.39 ms
是更强的对照。旧eager→graph的收益不能转述成“比作者compiled快3.2倍”。

代码上，我们的旧graph只捕获raw denoise，video prefill仍为eager；作者同时compile
prefill和denoise。Graph合并提交不等于编译器融合，因此已有大幅graph收益仍可能落后。
这不是对约103 ms差距的分段定量归因：本轮没有另跑profiling去测各部分占比。
新路径把这两个编译区域接入现有固定noise／Euler20接口，默认保持关闭；实测结果见上节。

π0即使都用2步，仍是作者BF16与项目FP16，不是纯后端或等质量对比；10步与2步之差也不能
全部算作后端收益。这里只并列真实延迟，不把原本没有证明的Euler2质量等价补写出来。

π0.5项目8步相对作者10步的原始中位数延迟比为1.5158×，但包含减步收益，不能据此单独
宣称后端快1.52×。作者8步尚无正式样本；10步首次冷编译调用约276.13 s，随后8步二次编译
触及420 s的作者进程**总**预算，不代表8步推理耗时420 s。为遵守“尽快、不再测试”的要求，
在主比较已取得Native10步结果后停止这个附加点，不扩大重跑。

Hy同卡中位数延迟比为 **6.4895×**（804.2853÷123.9355）。这是记录边界内的整套实现
延迟比，不是纯kernel收益，也不是机器人质量等价证明。双方均为3路相机×6帧、Euler10、
BF16主体、host原始输入到FP32 CPU raw `[40,32]`；作者运行时为Torch2.7.1+cu128／
FA2 2.7.4.post1，本项目为既有Torch2.11执行栈。双方预处理及文本缓存策略属于各自方法。

本项目reference→bookend中位数漂移分别为：π0 −0.3694%、π0.5 +1.7349%、
FastWAM −0.1459%、Hy −2.3852%；作者Hy −0.3714%。这些是固定配置的重复测量，
没有借此新增质量胜点；Hy仍只标基线。

首批源码：[c065ba04](https://github.com/20261023/FlashRT/tree/c065ba04de2e57a93c672ed54e25a019ea38e839)；
π0／Fast配对源码：[46f846da](https://github.com/20261023/FlashRT/tree/46f846da9f5a2ff23e06b691229259d224ed8354)；
π0.5配对／共享权重加载适配：[11959303](https://github.com/20261023/FlashRT/tree/11959303f359ed437e24f11e0599661909452027)；
[OpenPI作者源码dd5](https://github.com/Physical-Intelligence/openpi/tree/dd5c5ca93ce229017477a75fdda9f16654ab91ef)，
[FastWAM作者源码7faa](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4)，
[Hy作者源码af57](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/tree/af57e7507ec5964b52fdf6296741e553cbcd3288)。
新增FlashRT列／Fast编译路径源码：[227993d5](https://github.com/20261023/FlashRT/tree/227993d52b83aa769132cd5d04ade0a5b53d5a55)。

## 1. 术语与测量目标

- **官方 Native**只指作者原论文／原仓库的执行栈，**不是FlashRT**。
- 本项目FastWAM Torch graph虽使用Torch，也不是作者原始`infer_action`执行入口，必须另列。
- 主表采用A800上作者原实现与本项目方法的实际计时；作者论文／仓库数字另作参考。
- 不预设本项目更快，不把不同GPU、输入、步数或计时边界的数字直接相除成“加速比”。

## 2. 已核实选择与首批复测（326641）

| 模型 | 本轮固定方法 | 已有选择依据与限制 | 首批A800中位数 |
|---|---|---|---|
| π0 | FlashRT FP16 Euler2，3view、chunk50、Orbax/JAX权重loader | `307979`保留Euler2；loader不是Native JAX执行；未证明Euler2等价默认Euler10质量 | 25.9319 ms median |
| π0.5 | FlashRT BF16 Euler8，3view、chunk50 | `308244`四请求数值代理范围内选择，不是泛化／机器人质量保证 | 38.7992 ms median |
| FastWAM | 本项目Torch raw-field graph Euler20，shift5、H32、2cam224×224、cached-T5 | `313779`扩大8案例拒绝Euler13/14/16，保留Euler20；不能用旧四案例Euler13冒充当前最优 | 249.7167 ms median |
| Hy | FlashRT semantic-BF16 Euler10＋graph，3cam×6帧，模型chunk40、raw32/public20 | 新public20权衡目标尚无实测搜索胜点；此项明确是稳定基线，不称全局最优 | 123.9355 ms median |

## 3. 官方已发布的数字

NR表示来源未报告，不能用当前默认配置补填历史条件。

| 模型／作者版本 | 官方 Native发布延迟 | GPU | 已知条件／范围 | 相对本轮A800 |
|---|---:|---|---|---|
| π0原论文 | **73 ms** on-board；86 ms off-board | RTX4090 | 3cam、H50、Euler10；统计量／dtype NR；off-board额外13ms网络 | 条件不同，仅作背景，不算加速比 |
| π0.5，作者RTC论文vanilla行 | **76 ms** | RTX4090 | BF16、H50、5步；5warmup后10次mean；仅GPU神经计算；不是RTC97ms | 硬件／步数／边界不同，仅作背景 |
| FastWAM论文 | **190 ms** | RTX5090D V2 32GB | H32、10步、T5计入；作者澄清无compile/graph，实机384×320而非本项目LIBERO224×448 | 硬件／输入／文本／步数不同，仅作背景 |
| FastWAM后续官方仓库compiled默认 | **110 ms**（此前190ms） | RTX4090 | 包含text/VAE；具体checkpoint／步数／统计量等NR | 不倒填条件，不算加速比 |
| 同一官方仓库compiled默认 | **210 ms**（此前470ms） | H20 | 同上，不能与论文190ms合并成一条测量 | 不算A800加速比 |
| Hy官方论文／仓库／模型卡 | **NR** | NR | 未找到完整单次推理ms；任务完成秒数／servo Hz不是推理延迟 | 本轮补跑官方实现 |

来源：

1. [π0原论文Appendix A-D／Table I](https://arxiv.org/html/2410.24164v4#A1.SS4)。
2. [作者RTC论文Appendix A.3／Table 1](https://arxiv.org/html/2506.07339v2#A1.SS3)。π0.5该行是vanilla，不是RTC、BID或网络端到端数字；精确checkpoint／私有runtime未证明等于公开OpenPI。
3. [FastWAM论文v2](https://arxiv.org/html/2603.16666v2#S4.SS1)、[作者对190ms的澄清](https://github.com/yuantianyuan01/FastWAM/issues/34#issuecomment-4250694653)。
4. [FastWAM固定提交README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)。官方默认可以包含作者自己的compile优化，Native不等于一律eager。
5. [Hy论文v2](https://arxiv.org/html/2606.14409v2)、[官方源码af57](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/tree/af57e7507ec5964b52fdf6296741e553cbcd3288)。

## 4. 本轮固定计时协议

- 单A800、单allocation、串行独立进程，30分钟硬上限；每个固定配置reference＋bookend，不重新搜索。
- 本项目π0／π0.5／FastWAM／Hy分别保留1／4／8／7个既有case；每进程8次warmup，正式50／40／80／70次，均衡覆盖。
- 计时为host observation／固定noise到完整CPU动作chunk，包含`infer`触发的输入处理／上传／模型／输出；load、prepare、warmup单列。
- FastWAM明确使用预计算T5，不包含现场文本编码；与作者含T5的发布值不假称同口径。
- 共用worker若增加profile请求，其位于正式样本之外，不拿profile耗时作为正式延迟。
- 保留完整延迟样本、动作、case ID、实际NFE、source、GPU和环境身份；核验覆盖、有限性、cleanup与reference/bookend漂移。
- 质量门在本轮只检查固定配置的bookend/replay稳定性，不借reference角色宣称新的算法质量选优或官方质量等价。

## 5. Hy官方 Native自测协议

现有作者Torch2.7.1／FA2 2.7.4.post1环境及完整checkpoint保持只读。
从作者Git提交af57导出未修改的`hy_vla`源码，并核对导出字节和实际导入位置。

计时调用原作者`prepare_images`、`prepare_state`、`prepare_language`，然后调用原作者
`policy.model.sample_actions(..., noise=固定noise)`，最后返回FP32 CPU raw `[40,32]`。
它与本项目Hy的host uint8历史／state／prompt／noise到CPU raw chunk边界相对应。
这是计时与输入转换wrapper，**没有替换作者的神经／采样实现，没有构造FlashRT frontend／pipeline／扩展**。

不采用缓存action queue命中的`select_action/get_action`时间；不将仅算预处理后`sample_actions`的时间冒充完整请求。
`forward_evaluate`本身不接受固定noise，所以明确记录上述原API组合，而不谎称调用了该入口。
一个额外的未计时请求核对Euler10调用次数，观察器在warmup／正式计时前移除。
每次仍执行完整视觉／prefix／动作采样；raw32与public20区别保留，RoboTwin模型chunk40是20relative＋20absolute位置，并非40个不同未来时刻。

官方Hy先测reference，四个本项目配置随后执行，最后补官方Hy bookend。
一个模型／官方环境失败不阻断其余独立模型的测量；失败项保留NR与错误，整批不能冒称完整成功。

## 6. 本轮结果

`326641`的10份结果与全部动作／延迟数组已回传，并从原始样本复算上方统计量。
四个v2 campaign的8个固定点全部入账为validated，仅表示各自固定配置replay作用域；
不将reference角色当作独立准确性证明。`326708`的π0／FastWAM结果已补入主表；π0.5首轮
因共享embedding别名省略而加载失败，修复后`326869`取得Native10步和同卡项目8步结果。
该作业后来在附加8步编译处触及进程预算，整批状态为FAILED1:0，但已完成的两份原始结果保留。
不能把调度失败说成整批没有结果，也不能把部分完成说成所有附加点成功。

### 输入与边界明细

| 模型 | Batch／图像 | 状态／文本 | 输出／步数 |
|---|---|---|---|
| π0 | B1；3×224×224 RGB uint8，无历史 | 已准备FP32 state[8]；每次作者原tokenizer/resize/pad，本项目可缓存prompt | CPU raw[50,32]，public7；Native BF16 Euler10及2，项目FP16 Euler2 |
| π0.5 | B1；3×224×224 RGB uint8，无历史 | 已准备FP32 state[8]；H50权重的discrete state prompt，作者max token200，项目exact prompt | CPU raw[50,32]，public7；Native BF16 Euler10及8，项目BF16 Euler8 |
| FastWAM | B1；2×224×224水平拼接，FP32[1,3,224,448] | FP32 proprio[1,8]、cached-T5[1,128,4096]及bool mask[1,128]；不含现场T5编码 | CPU FP32[1,32,7]；shift5，Native compiled Euler10及20，项目Euler20 compiled prefill＋denoise |
| Hy | B1；3路×6帧原始uint8[6,480,640,3]，预处理到224×224 | FP32 normalized state[20]及任务字符串；固定noise[40,32] | CPU raw[40,32]，public20；双方Euler10；40位置为20relative＋20absolute |

所有计时均以完整CPU动作返回为结束，不用异步enqueue时间充当推理时间；不含权重加载、
冷编译、warmup或fixture磁盘IO。π0／π0.5的作者行使用原始模型及原始Policy/model transforms，
但用本轮三图槽／H50／frontend-ready state边界，不冒充作者机器人部署默认完整入口。
作者π0冷JIT首调用约31.38 s；FastWAM作者compile首调用约82.66 s，均未混入稳态毫秒统计。

运行栈：π0作者为JAX0.5.3／Flax0.10.2／Orbax0.11.13；π0.5作者为Torch2.7.1+cu126，
Transformers4.53.2加作者原始`transformers_replace`任务overlay；FastWAM作者与项目均用
既有Torch2.11.0+cu128；Hy作者为Torch2.7.1+cu128／FA2 2.7.4.post1，项目为Torch2.11。
这些是作者原代码在所记录环境的自测，不是原论文环境的逐项复现。

### 任务与作业结束状态

| 作业 | 结束状态 | 保留结果 |
|---|---|---|
| `326641` | COMPLETED 0:0 | 四个项目方法各reference/bookend、作者Hy各reference/bookend，10份完整结果 |
| `326708` | FAILED 1:0，部分完成 | 作者π0两步数、FastWAM两步数、三个项目配对点；π0.5首次严格加载失败 |
| `326869` | FAILED 1:0，部分完成 | π0.5作者10步及同卡项目8步完成；附加作者8步二次编译未完成 |
| `327380` | COMPLETED 0:0 | 四个FlashRT基线、Fast旧／新路径、作者compiled10／20，共8份点结果 |

所有本轮作业均已结束；没有仍在占用GPU的任务。用户要求停止测试后未再运行pytest、smoke测试、
额外数值探针或评审；只执行实际计时、读取产物和统计汇总。本报告不声称新的闭环任务质量验证。

### 6.1 已确认的同卡补测

保留作业326641；新增一个串行A800作业，测作者π0 JAX BF16 Euler10／2、
作者π0.5 PyTorch BF16 Euler10／8，以及作者FastWAM compiled BF16 Euler10／20。
同一allocation内各补一个本项目已选配置计时，以便同卡配对，而不是用异卡论文数字相除。
π0官方BF16与本项目FP16仍有精度差异，不能把整套方案的差异称为纯后端收益。

π0.5复用当前LeRobot格式H50权重，通过临时最外层`model.`加载命名空间及共享权重感知的
严格safetensors loader加载原作者类，不忽略缺失推理参数、不留下随机embedding；
神经／采样实现不改。使用H50、原始三图槽和该权重的discrete-state输入设置，不能混称
官方`pi05_libero`默认H10部署。作者所需`transformers_replace`放在新任务overlay，不改旧环境。
FastWAM保留作者自己的compile默认路径，显式cached-T5、shift5，与本项目相同输入几何。

用户要求加快执行并停止测试，因此此后不运行CPU测试、smoke测试、额外数值探针或新评审；
后续只执行实际延迟测量、读取原始样本并汇总，失败项如实报告。
