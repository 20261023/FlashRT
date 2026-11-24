# VLA 自主调优整合报告：五模型历史结果、迁移过程与本轮待验收项

**整理日期：2026-11-24。状态：历史证据已整合，本轮平台与新模型验收未完成。**

本文合并 11 月 9 日阶段报告、11 月 12 日研究过程、历史逐轮实验记录、
11 月 14 日延迟比较以及 11 月 23 日五份增量报告。日期是文档整理日期，
**不是所有模型在 11 月 24 日重新测量的日期**。后续结果必须另填实数、
实际源码身份和验收范围，不能用历史毫秒数代替本轮验收。

读者可以先读大表与结论，再按模型查看过程；附录保留完整历史逐轮正文。
这是一份工程实验报告，不是机器人任务成功率、通用模型排行榜或全硬件认证。

## 1. 可以支持的结论

- **π0**：保留 FP16 Euler2 执行对照。ProbeFlow2 在原测试中更慢；
  尚未证明两步策略与默认十步策略等质量。
- **π0.5**：四输入代理曾选择 BF16 Euler8；SM80 W8A8 两轮均未过质量预算，
  包括修正残差／归一化舍入后的复测，不能按最快裸延迟选 INT8。
- **FastWAM**：先消除重复规划、验证 graph 保持性，再接入 compiled
  prefill＋denoise，最后验证请求内文本 K/V 缓存。最新历史 compiled 缓存点
  为 **127.912 ms**；此前失败作业仍为 NR。它不是完整 FlashRT 原生模型。
- **Hy**：旧 EXACT 主线有真实局部发现，但完整策略修复未闭合；用户随后将目标
  改为性能—质量权衡。七输入 Euler8＋SM80 融合的历史点为 **101.83 ms**，
  不能据此改写旧官方逐元素门的失败，也不能当作本轮 schema 3 复现已完成。
- **DM0.5**：项目 compiled 方法在固定四案例完整 CPU 动作边界显著快于未改动的
  partial FlashRT 和 ApxInf 对照；最新历史双门槛实验为 **94.8282 ms**。
- **G／SmolVLA：尚无可引用的 GPU 作者基线、项目双门槛、重复计时和调优结果。**
  已声明模型、发现权重或 CPU 测试通过都不等于接入验收通过，延迟保持 NR。

## 2. 如何读这些数字

### 2.1 术语、身份与比较规则

**官方 Native 只指作者原论文／原仓库的执行栈，不是 FlashRT。**
本项目自行改造的 Torch graph／compiled 路径单列；“partial FlashRT”说明只替换了
部分算子。“ApxInf 研究移植”不代表公开上游已有该 family 或完成部署验收。

表内均为 NVIDIA A800-SXM4-80GB 的稳态毫秒数，但**同型号不等于同一 allocation**。
正式请求从相应 host 输入到完整 CPU 动作返回，包含请求内处理、传输、模型与采样。
加载、冷编译、capture、warmup、fixture 文件 IO 和独立诊断请求不计入稳态。
FastWAM 使用准备好的 T5 context，不包含现场文本编码或未来视频 rollout。
CUDA-event 区间、graph-only、prepared 输入和完整请求不能混算加速比。

数值质量只在事先固定的输入、通道、参考和阈值内成立，不等于机器人成功率。
多数性能筛选使用所有请求最坏 relative L2≤0.05、最低 cosine≥0.999；
π0 的两步执行比较和 Hy 的旧逐元素门不能事后改套这个定义。
NR 表示未测、未完成或来源未报告，**不是 0，也不允许插值补齐**。

<a id="a800-results"></a>

### 2.2 A800 总表：旧基线、最新历史选择与作者分列

以下是一张导航表，不是一次新 campaign，也不按跨作业数值挑 winner。
Median／P95 的完整旧统计保存在未修改的
[29 点 CSV](2026-11-14-vla-latency-comparison.csv) 与
[JSON](2026-11-14-vla-latency-comparison.json)。

| 模型／输入范围 | 项目／FlashRT 固定基线 median ms | 最新已记录方法 median／P95 ms | 作者官方实现 A800 median ms | 最新点的质量与处置 | actor 与证据 |
|---|---:|---:|---:|---|---|
| π0，1 输入，B1／3×224²／H50／public7 | FP16 E10 graph **40.3384**，`327380` | FP16 E2 **26.1164／26.4021**，`326708` | JAX BF16 E10 **69.3703**；E2 **48.3255**，同 `326708` | 仅保留两步执行对照；默认质量未验收 | 历史逐点 actor NR；来源 §5.1、§9 |
| π0.5，4 输入，B1／3×224²／H50／public7 | BF16 E10 graph **41.9108**，`327380` | BF16 E8 首／尾 **38.798／38.534**，P95 **38.915／38.654**，`357483` | PyTorch compiled E10 **59.4795**，`326869`；作者 E8 **NR** | 保留 BF16；两档修正后 W8A8 均 0/4 过门 | 历史逐点 actor NR；来源 §5.2 |
| FastWAM，8 输入，B1／2×224²／H32／public7 | partial FlashRT cached＋graph E20 **263.3574**；旧项目 graph **250.9826**，`327380` | 项目 compiled＋请求内文本 KV，E20 **127.912／128.259**，`357482` | compiled E20 **142.4363**；E10 **85.7984**，`327380` | 对同轮未缓存 compiled：L2 **0.003945**／cos **0.999992**，通过；不是作者同期重测 | 历史逐点 actor NR；来源 §5.3 |
| Hy，7 输入，B1／3 相机×6 帧／40 位置／public20 | semantic-BF16 E10 graph **123.9355**，`326641`；独立旧重测 **121.2805**，`327380` | BF16 E8 graph＋RoPE/QK/KV 融合 **101.83／NR**，`356766` | 原作者 E10 **804.2853**，`326641` | 项目 L2 **0.00729749**／cos **0.99997344**；离线作者 E10 **0.03619154／0.99936386**，七例通过 | 历史逐点 actor NR；来源 §5.4 |
| DM0.5 base，4 输入，B1／3×448²／H50／public14 | partial FlashRT E10 **204.9714／204.9462** 首尾，`356766` | 项目 compiled E10 **94.8282／NR**，`356766` | 原作者 default／vision SDPA／无 Liger E10 **277.6579**，`329331` | 项目 **0.011178／0.999938**；作者 **0.008185／0.999972**，双门槛通过 | 历史逐点 actor NR；来源 §5.5 |
| G／SmolVLA，新模型 | **NR** | **NR／NR** | **NR** | 作者／项目、graph、重复测量和优化资格待补 | 当前接入 actor=agent／Codex；GPU 结果待证 |

**actor 限制：**历史报告描述了 agent 驱动的实现与测量，但没有为每个历史点公开
同期 `actor` 决策记录，因此不追填为 `human`／`agent`。本次整合是
`actor=agent`、执行者 Codex；当前决策见
[append-only 决策日志](../vla_autotune_decisions.jsonl)。
历史 NR 不能由本轮代理身份替代；确定性控制器选点也不等于人批准合入。

### 2.3 可以计算的同轮相对变化

| 作业／范围 | 比较 | 中位延迟变化 | 解释边界 |
|---|---|---:|---|
| `308244`，π0.5 四输入 | E10 41.6211 → E8 38.4342 | **−7.66%** | 减步＋数值预算内选择，不是纯后端比值 |
| `327380`，FastWAM 八输入 | 旧项目 graph 250.9826 → compiled 137.6503 | **−45.1554%／1.8233×** | 均 E20；接入作者既有编译思路，不是新 native kernel 收益 |
| 同 `327380` | 作者 compiled E20 142.4363 → 项目 compiled 137.6503 | **−3.3601%** | 整套接口编排比较；noise 生成和提交方式有差异 |
| `335362`，DM0.5 四输入 | 固定 partial FlashRT 206.390868 → compiled 96.1658565 | **−53.40595%／2.14620×** | 原始对照未改，完整 public policy 同边界 |
| 同 `335362` | 更快 ApxInf bookend 150.708811 → compiled 96.1658565 | **−36.19095%／1.56718×** | ApxInf 是研究移植；不与旧作者作业拼接纯 engine 比值 |
| `356212`，Hy 七输入 | E10 120.50 → E8 116.92 | **−2.97%** | 首尾漂移 +0.60%；先过新 public20 门 |
| `356766`，Hy 七输入 | 更快未融合基线 113.16 → 融合 101.83 | **−10.01%** | 首尾漂移 +4.43%；不与上一作业连乘 |
| `356766`，DM0.5 四输入 | 更快 mixed 复测 204.9462 → compiled 94.8282 | **−53.73%** | 此模型三个包完成并入账；整项作业后来超时 |
| `357482`，FastWAM 八输入 | 更快 Torch 基线 743.469 → 缓存 667.076 | **−10.28%** | 同轮 eager 对照，输出逐元素相同 |
| 同 `357482` | 更快 compiled 基线 137.499 → 缓存 127.912 | **−6.97%** | 热编译缓存私有副本；冷缓存成本尚不能由此得出 |

没有把 π0 的 FP16 E2 与作者 BF16 E10 的全部差额叫后端收益；
没有拿 π0.5 的 E8 与作者 E10 的 **1.5158×** 原始比值冒充同工作量加速。
Hy 旧同卡 E10 的 **804.2853÷123.9355=6.4895×** 只是所记录整套实现的延迟比：
作者 Torch2.7.1／FA2 对项目 Torch2.11，预处理／文本缓存等也属于各自方法。
它既不是纯 kernel 比值，也不是新 E8 融合点对作者的同期比较。

## 3. 原始 schema、源码身份与复现资格

| 证据层 | 实际证明了什么 | 本轮不得补写的字段／资格 |
|---|---|---|
| 早期 v1 与独立 validator 结果 | 对应冻结源码和固定输入的执行、数组与延迟观察 | 不补填 campaign-instance／confirmed-job 或 source identity 冒充 v2/v3 |
| 严格 v2：π0.5 `308244`、FastWAM `309300/309386/313779` | 完整 Candidate、campaign instance、已确认作业、质量范围与覆盖绑定 | v2 不是不可变源码快照的 schema 3 证明 |
| 11 月 14 日 JSON/CSV | 29 条历史点、28 完成计时＋1 个作者 π0.5 E8 NR；有 method／job／source／计时边界 | 这是报告数据 schema，不代表每条点的 worker plan 都是 v3 |
| 11 月 23 日五份报告 | 已记载包完成、控制器入账、动作复算、首尾与质量结果 | 原文未公开完整 plan/snapshot/native manifest；schema 3 资格 **NR，待本轮复现** |
| 本轮来源候选 | 必须完整 40-hex source、内容寻址 patch、ready baseline/candidate/bookend、同 harness、plan v3 和真实导入核验 | 仅有 Git HEAD、CPU 通过、作业提交或旧报告不能替代 |

`analysis.lever_pricing`、非 config 或未知工具、不同 `source_ref`、
任一点非空 `patch_ref` 都要求快照；不删除 `tool` 或把缺失 ref 解释成继承。
新配置候选与旧 legacy 记录分开。当前框架规范见
[autotune](../vla_autotune.md)、
[接入指南](../vla_onboarding.zh-CN.md) 与
[工具／迁移流程](../vla_autotune_pipeline.zh-CN.md)。

### 3.1 历史源码索引

这些 pin 保留旧来源，不表示今天的 checkout 就是当时执行字节。
历史快照含未提交内容时，必须保留原 content manifest；后补提交不能反向替换身份。

| 范围 | 固定来源 |
|---|---|
| 本轮整理所依据的完整旧文档集合 | [199376281801427abccbb307056b27e74eab1429](https://github.com/20261023/FlashRT/tree/199376281801427abccbb307056b27e74eab1429/docs) |
| 早期阶段记录 | [636766482d77993f968f0850c9c72e6f4eec0258](https://github.com/20261023/FlashRT/tree/636766482d77993f968f0850c9c72e6f4eec0258) |
| FastWAM library graph／v2 搜索 | `round012/a5140aac`；`round014/baef55ff`，不追认为后来编译路径 |
| 旧 Hy／首批四模型配对 | [c065ba04de2e57a93c672ed54e25a019ea38e839](https://github.com/20261023/FlashRT/tree/c065ba04de2e57a93c672ed54e25a019ea38e839) |
| π0／FastWAM 配对 | [46f846da9f5a2ff23e06b691229259d224ed8354](https://github.com/20261023/FlashRT/tree/46f846da9f5a2ff23e06b691229259d224ed8354) |
| π0.5 作者加载／配对 | [11959303f359ed437e24f11e0599661909452027](https://github.com/20261023/FlashRT/tree/11959303f359ed437e24f11e0599661909452027) |
| FastWAM 编译／四个固定 FlashRT 基线 | [227993d52b83aa769132cd5d04ade0a5b53d5a55](https://github.com/20261023/FlashRT/tree/227993d52b83aa769132cd5d04ade0a5b53d5a55) |
| DM0.5 原始 partial FlashRT | [cbacd878b7d5f58231de7780c545b3e09c0542af](https://github.com/20261023/FlashRT/tree/cbacd878b7d5f58231de7780c545b3e09c0542af)，内容摘要 `e16bb373…106b` |
| DM0.5 竞争优化 | [f111d787e884b49eaf13bf54eb89e7855a2fd35d](https://github.com/20261023/FlashRT/tree/f111d787e884b49eaf13bf54eb89e7855a2fd35d)；实跑内容摘要 `eb901a1d6cb35be4bb9d2ecad49c4011fd0794052a21e1a73853d173e87946d8`，freeze HEAD `e9ede857` 本身不包含候选 |
| π0.5 INT8 舍入修正 | [7346d21e901653d3511366b5df70b611a0bd3b0c](https://github.com/20261023/FlashRT/tree/7346d21e901653d3511366b5df70b611a0bd3b0c) |
| Hy 历史融合目标／本轮要求复现的 harness | [315ce8940bfa74bba5838b7be0e4b2682a26c72f](https://github.com/20261023/FlashRT/tree/315ce8940bfa74bba5838b7be0e4b2682a26c72f)／[1876dbc3bd6dffd52c30dc38a24aee02278c076a](https://github.com/20261023/FlashRT/tree/1876dbc3bd6dffd52c30dc38a24aee02278c076a)，本轮组合接线与包装需另记，不能隐去 |
| OpenPI 作者 | [dd5c5ca93ce229017477a75fdda9f16654ab91ef](https://github.com/Physical-Intelligence/openpi/tree/dd5c5ca93ce229017477a75fdda9f16654ab91ef) |
| FastWAM 作者 | [7faa71108368fbb3b6885649f112af607427a2d4](https://github.com/yuantianyuan01/FastWAM/tree/7faa71108368fbb3b6885649f112af607427a2d4) |
| Hy 作者 | [af57e7507ec5964b52fdf6296741e553cbcd3288](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/tree/af57e7507ec5964b52fdf6296741e553cbcd3288) |
| DM0.5 作者 | [03f799120d425ead51d9482bb1da950b6dabade9](https://github.com/dexmal/opendm/tree/03f799120d425ead51d9482bb1da950b6dabade9) |

11 月 23 日报告的最后发布提交分别为：DM0.5 `3b80a453`、FastWAM `19937628`、
Hy 配置 `3c98388a`、Hy 融合 `871c44ed`、π0.5 INT8 `eda8122b`。
**它们是报告来源，不自动等于运行 source_ref。** 本文不凭报告提交补造旧 manifest。

## 4. 迁移与自主实验过程

整个过程不是“不断调参直到通过”，而是四条线逐步分开：

1. **执行接缝**：模型 backend、raw velocity、Euler／ProbeFlow、prepare／infer／close；
   observation 和动作解码遵守模型自己的契约，不猜通用机器人预处理。
2. **测量可信度**：冻结源码／输入，reference→候选→bookend，完整动作、
   实际 NFE、覆盖、有限性、清理和独立 CPU 复算；profile 不进入正式样本。
3. **性能方法**：减少重复 host planning、graph 提交、编译融合、缓存以及有限步数搜索；
   不预设 native 胜过 Torch，不把局部算子收益冒充端到端收益。
4. **数值研究**：复现未改控制，只改可区分的因素；局部 EXACT、完整策略质量、
   稳态性能和生产资格分别验收。反例会改变下一步，而不是改变旧门槛。

### 4.1 决策归属和人的干预

| 时间／事项 | 可核实归属 | 发生了什么 | 不能推断 |
|---|---|---|---|
| 历史实现、实验和 CPU 复算 | 报告记载为 agent 驱动；逐轮同期 actor NR | 提假设、写接缝、运行固定比较、拒绝失败点 | 不能事后把每项选择都记成 agent 自主决定或人的批准 |
| 2026-11-13 Hy 目标修订 | **human**，旧研究过程明确引用用户纠偏 | 停止以旧 runtime／中间量 EXACT 为性能前提，改 public20 性能—质量预算 | 不是把旧 6/7、5/7 失败重命名为通过 |
| 作者基线／项目术语 | 人的要求在历史报告中明确保留 | 官方 Native 与 FlashRT、项目 Torch、ApxInf 分列 | 不能用更弱 eager 对照冒充击败作者 compiled |
| 本轮 A—K 验收规则 | **human**，见当前决策日志首条 | public L2/cos 双门、来源快照、停止条件、分段 token 和 H4 | 不是 agent 合入授权 |
| 本次 J 汇总、引用迁移和冗余报告移除 | **agent／Codex** | 保留旧数值、旧身份、NR 与失败，整合而不重测 | 不计作新模型／新优化验收 |

**干预次数只能给下界：**历史文字可直接核实一次 Hy 目标纠偏；
完整历史人工干预总次数 NR。当前日志条目可按 actor 统计，但不与历史叙述混算成
“全程人工次数”。成本与自动化程度不能靠事后回忆精确化。

## 5. 五模型结果与反例

<a id="pi0-history"></a>

### 5.1 π0：两步接缝成立，没有观察到 ProbeFlow2 加速

`307979`，每点 10 样本：E2／PF2／E2 bookend 为
**26.1060／26.7000／26.3815 ms**，均实际 2 NFE。
PF public7 relative L2 **0.000375709**、cos **0.999999945**，通过该次执行代理却更慢。
保留 E2，不声称完成全参数搜索。`326708` 的作者 JAX BF16 与项目 FP16 配对
只证明所记录接口的实际耗时，不证明 dtype 或步数质量等价。
`framework="jax"` 的项目路径是 Orbax 权重加载接缝，不是独立 JAX 数学执行引擎。

<a id="pi05-history"></a>

### 5.2 π0.5：单输入候选被多输入推翻，INT8 修正后仍拒绝

早期 E7 单输入 **36.9639 ms／2.56% L2** 不能泛化。Post-ViT pool2 把
prefix **911→335** 行，**28.8652 ms** 却有 **33.21% L2**；它没有减少已经执行的 ViT。
四输入 v2 `308244` 的 E7 最坏 L2 **0.665398**／cos **0.767542**，两个 PF 点
也失败；E8 **38.4342 ms／0.0361504／0.999482** 才在原预算内获选。

独立诊断 prefix 约 **24 ms**、单次 Expert **1.65–1.67 ms**，不是 kernel 求和；
17 个 prefix FFN 的 **3.11794 TFLOPs** 与单 Expert 的
**31.1427 GFLOPs＋7.08526 GFLOPs attention** 是逻辑账本，不是 HBM 流量或 roofline。
动作反归一化使用 checkpoint 声明的 `ACTION=MEAN_STD`；旧 raw 数值证据不能被称作
修正后的机器人命令。

两轮 INT8 均为 A800、E8、Graph、三路 224²、H50，4 输入各 3 次正式计时，
每轮 48 份完整输出，public7 门为 0.05／0.999：

| 作业／配置 | Median ms | P95 ms | 最坏 L2 | 最低 cosine | 处置 |
|---|---:|---:|---:|---:|---|
| `357202` BF16 首测 | 38.341 | 38.721 | 0 | 1 | 基线 |
| `357202` W8A8 前缀 | 35.493 | 35.671 | 1.181721 | 0.234738 | 拒绝 |
| `357202` W8A8 前缀＋动作 Expert | 39.325 | 39.461 | 1.180410 | 0.237044 | 拒绝 |
| `357202` BF16 复测 | 38.291 | 38.429 | 0 | 1 | 通过，漂移 −0.13% |
| `357483` 修正后 BF16 首测 | 38.798 | 38.915 | 0 | 1 | 4/4 |
| `357483` 修正后 W8A8 前缀 | 36.116 | 36.374 | 0.436680 | 0.900977 | 0/4，拒绝 |
| `357483` 修正后 W8A8 前缀＋动作 Expert | 40.413 | 40.568 | 0.429632 | 0.903941 | 0/4，拒绝 |
| `357483` BF16 复测 | 38.534 | 38.654 | 0 | 1 | 4/4 |

首轮最坏输入为 `noise-seed-20261110`，修正后为 `brightness-plus8`；
实际 INT8 GEMM 覆盖分别为 **86／176** 处。修正让残差先按 BF16 舍入、
归一化结果存 BF16 再量化，但没有救回质量。两作业 BF16 首测的最坏
public7 L2 **0.036975**、cos **0.999546** 也保留，不假定跨作业控制逐位相同。
**首个较大偏差的模型层仍未由这些输出定位；本轮 F 需补真实中间层证据。**

<a id="fastwam-history"></a>

### 5.3 FastWAM：执行保持性、采样近似和编译缓存分别验证

早期 Euler2 `308343`：legacy **202.5082**→cached **154.7789 ms**，
Torch **122.5195 ms** 更快。Cached 建 **19** 个 plan，稳态零建／零查询；
workspace **64 MiB 对 256 MiB**，因此不将全部 **47.729 ms／23.57%**
归因于 host planning。`308460` 的每请求 **920 Linear／1.22055 TFLOPs**
是两步账本；约 **14.0704 GB** 逻辑权重不是实测 HBM。

公平给两引擎 graph 后，`308611` Torch **69.0292 ms**，cached **75.8655 ms**，
各自 graph/eager 精确一致。后续 library graph 验证 E10、E20、dense restart、
六类输入刷新和 owned cleanup；E20 `309187` 的 Torch
**804.6938→255.0711 ms** 是对自身 eager 的约 3.15×，不是击败作者 compiled。
305 次稳态 host enqueue 之外，20 次 graph replay 仍编码 **6140** 次逻辑 native Linear；
不能宣称 GPU 工作消失。

四输入 v2 曾选 E14，再细化选 E13 **179.7011 ms**，对 E20
**251.6074 ms** 降低 **28.58%**。扩大八个压力输入的 `313779` 却拒绝
E13／14／16：最坏 L2 **0.194629／0.117659／0.179380**，
都在 `context-minus0p0625` 失败，保留 E20。直接改 T5 向量是离分布数值压力，
不是新自然语言任务，也不能用“增加步数”预设误差单调变小。

`327380` 才将编译接到 prefill＋denoise，保持 E20／BF16／shift5／H32：
旧 graph **250.9826**、项目 compiled **137.6503**、作者 compiled **142.4363**、
partial FlashRT **263.3574 ms**。80 份项目输出与对应作者输出逐元素相同；
对旧 graph 最坏 L2 **0.0393335742**／cos **0.9992527670**。
冷 prepare 约 **24.60 s** 单列，不混入稳态。

**请求内文本 K/V：**

| 作业／路径 | 未缓存首测／复测 median ms | 缓存 median／P95 ms | 最坏 L2／最低 cos | 结果 |
|---|---:|---:|---:|---|
| `357202` compiled 基线 | NR／NR | NR／NR | NR／NR | 基线 worker `-11` SIGSEGV；候选及 bookend 未启动 |
| `357482` 常规 Torch | 751.089／743.469 | 667.076／671.025 | 0／1 | 通过，输出逐元素相同；漂移 −1.01% |
| `357482` compiled | 137.499／137.864 | 127.912／128.259 | 0.003945／0.999992 | 通过；漂移 +0.26% |

`357482` 同一 A800 自然完成，八输入、每点 24 样本，六点共 **144** 次。
每请求缓存预计算一次、30 层 K/V 投影、20 次 Expert。Compiled 使用旧编译缓存的
私有副本，首次加载／编译／warmup 在外；**冷缓存 compiled 复验仍待补**。
首轮栈中的 Inductor／SymPy 线程不能单独确定 SIGSEGV 具体原因。
缓存按完整请求条件更新，不允许只按 prompt 跨请求复用。

**本轮新排查与旧测量分开：**
[冷缓存专题记录](2026-11-24-fastwam-cold-cache.zh-CN.md)记载，同一 Python3.11.17
环境在不导入 Torch／模型／GPU API 的有限纯解释器对照中，仅添加定时栈转储
即可从返回 0 变为 `-11`。这支持观测器竞态这一机制解释，不能替代旧作业的
原生 core 或将旧失败改成通过。独立空缓存 A800 作业 `359953` 已提交；
截至该记录截点，GPU 冷编译、稳态延迟、质量与重复实验均 **NR／待证**。

原生 ApxInf 研究移植 `329256` 的完整 policy **215.3437 ms**，prepared
**212.8693 ms**、graph 提交至完成 wall **212.5183 ms** 是三个边界。
相对作者保存输出 L2 **0.0116095123**／cos **0.9999326110**；
固定 NHWC 转换在 policy 外、不同 allocation，不给纯 engine 比值。
它未胜过已选 compiled，仍不是部署通过。结构与迁移语义详见
[FastWAM 架构说明](2026-11-15-fastwam-architecture-and-apxinf.zh-CN.md)。

<a id="hy-history"></a>

### 5.4 Hy：停止旧 EXACT 主线后才做性能—质量选择

**旧目标与新目标并存为不同历史 scope。** 旧门为 padded 40×32
逐元素 `abs(delta)≤0.01+0.01*abs(reference)` 且 cos≥0.999，public20 另报。
`308140` 原门 4/7；HWD PROJ1 局部 exact 后 `309681` 变 6/7，
state 仍有 3 元素超差，作业正确保留 FAILED。后来的组合甚至只有 5/7；
这些失败没有被新预算追认为通过。

11 月 13 日人纠偏后，新性能 scope 为 normalized 40×20、全部七输入，
相对固定 semantic-BF16 E10 的 L2≤0.05／cos≥0.999，不再要求中间量 EXACT。
没有放宽一个正在失败的旧实验来救候选，而是明确建立新任务。

| `356212` 配置 | Median ms | 新项目门 |
|---|---:|---|
| E10＋Graph 首测 | 120.50 | 基线 |
| E6＋Graph | 102.45 | 拒绝 |
| E8＋Graph | 116.92 | 通过、控制器选择 |
| E9＋Graph | 118.67 | 通过但更慢 |
| PF ε=0.002＋Graph | 124.05 | 通过但更慢 |
| PF ε=0.008＋Graph | 111.99 | 拒绝 |
| E10 关闭 Graph | 287.98 | 通过但更慢 |
| E10＋Graph 复测 | 121.23 | 通过 |

每配置 7 warmup、7 输入各 5 样本，八配置共 **280** 次。
E8 项目最坏 L2 **0.032544**、cos **0.999496**；对作者 `326641` 保存
BF16 E10 离线配对 **0.035629／0.999390**，最坏均为 noise。

`356766` 保持 E8＋Graph，仅开 `fused_rope_qknorm_kvwrite`，
七输入各 3 次，每点 21 样本：

| 配置 | Median ms | 独立视觉 ms | 独立前缀 ms | 独立八次去噪合计 ms |
|---|---:|---:|---:|---:|
| 未融合首测 | 113.16 | 24.70 | 12.08 | 37.33 |
| 融合 | 101.83 | 24.69 | 9.99 | 23.92 |
| 未融合复测 | 118.17 | 24.69 | 12.09 | 37.34 |

相对更快基线降低 **10.01%**，漂移 **+4.43%**。项目 public20
**0.00729749／0.99997344**，旧作者 E10 离线配对
**0.03619154／0.99936386**，七例预算内。
阶段数来自独立诊断，不能把它们求和当完整请求或 kernel-duration sum。
这不是已完成本轮 B 指定历史字节的 plan v3、算子与作者绑定复现。

<a id="dm05-history"></a>

### 5.5 DM0.5：保留原始对照，优化项目编译路径

固定为 `Dexmal/DM05` **base**：三路 448² 当前图像、无历史、
BF16 E10、H50／内部32／public14，DOS W1 绝对 FP32 CPU `[50,14]`。
不是两图 H10／7 维 LIBERO 变体。完整边界包含原始 PIL／state／prompt
处理与输出还原，不包含 HTTP、base64 或文件解码。

| 历史作业／路径 | Median／Mean／P95 ms | 对作者最坏 L2／最低 cos |
|---|---:|---:|
| `329331` 作者 default、vision SDPA、无 Liger | 277.6579／277.5912／278.9247 | 参考 |
| `329543` ApxInf 完整 public policy | 149.3522／149.3726／150.1001 | 0.0093243955／0.9999608552 |
| `334487` 项目 Torch 全图 | 194.2449／194.3161／195.2568 | 0.0077728340／0.9999727883 |
| 同作业 partial FlashRT | 206.0562／206.2177／207.8401 | 0.0086970135／0.9999650775 |
| 同作业 Torch bookend | 194.1798／194.4545／196.4995 | 同首测，动作逐元素相同 |
| `335362` 未改动 partial FlashRT | 206.3908680／206.5768056／207.56426970 | 0.0086970135／0.9999650775 |
| 同作业未改动 ApxInf | 151.1550295／151.29895675／152.51795445 | 0.0093243955／0.9999608552 |
| 同作业项目 compiled | 96.1658565／96.386913325／97.87547085 | 0.0091867243／0.9999653347 |
| 同作业 ApxInf bookend | 150.7088110／150.79157885／151.72859245 | 同首测，动作逐元素相同 |

每个完成点四案例各 5 warmup／10 formal。项目对同卡 ApxInf 最坏
**0.0121681631／0.9999260550**；对同卡 FlashRT
**0.0098358635／0.9999557181**，原预算未变。
方法组合为 QKV／gate-up packing、直接 SDPA、action GQA 和三类 layer 的 Inductor
编译，外层全图保留；没有单项消融，不能把组合收益逐项分摊。
旧图中 **2801 cached GEMM／367 FA2／401 strict GELU** 是逻辑调用，不是 Nsight kernel 数。

`335362` 独立诊断：FlashRT／compiled 的 vision
**18.958336／19.048449 ms**、prefix **51.077118／33.533953**、
十次 velocity **123.767809／31.680512**、Euler/other
**0.122885／0.123904**、whole-model **193.926147／84.382721**。
它们不是完整 CPU 请求的 **96.17 ms**，也不是单项贡献证明。
候选 cold load **41.5804 s**、内含 preparation **16.5874 s**，
两 profile compile-plus-first-forward **7.7712／3.8029 s**、capture **1.0147 s**；
不重复累加，不声称冷启动或总显存全面更优。
详细组合方法、内容身份和边界见
[DM0.5 竞争优化](2026-11-16-dm05-competitive-optimization.zh-CN.md)。

**最新历史统一流程 `356766`：**

| 路径 | Median ms | 对作者最坏 L2／最低 cos |
|---|---:|---:|
| FlashRT mixed 首测 | 204.9714 | 0.008697／0.999965 |
| 项目 compiled | 94.8282 | 0.008185／0.999972 |
| FlashRT mixed 复测 | 204.9462 | 0.008697／0.999965 |

四输入各 3 次；候选对项目首测 **0.011178／0.999938**，作者参考绑定旧
`329331` 保存输出，项目／作者双门均通过。独立视觉诊断首／候选
**19.12／19.14 ms**、prefix **50.99／33.38**、十次 Expert
**124.22／31.67**。整项 Slurm 作业在后续模型运行时达到时限，但 DM 三包完整、
已控制器导入；不能把部分完成写成整批成功，也不能把完整 DM 结果丢掉。

<a id="research-history"></a>

## 6. Hy 研究过程的认识修正

完整逐轮数值、旧源码与反例保留在附录。以下把重复叙述合成“问题—判别—修订”：

| 阶段／作业 | 竞争解释与区分实验 | 实际发现 | 后续决定／限制 |
|---|---|---|---|
| `308690/309016`，原始边界与冻结算子 | 同值输入区分 PatchConv batching、插值 cast、PROJ1 调用形式 | Patch／位置在受测值上 exact；PROJ1 B3 不同 **314260/1204224**，contiguous **440808** | 不盲换 dtype 或 `.contiguous()`；上游 vision 输入已经不同 |
| `309182/309409`，attention backend | 固定 QKV 比较外部 FA2 与四个 Torch backend／两 runtime | Flash **559** 值不同；Efficient **853921**、cuDNN **905372**、Math **925829**；没有 exact 替代 | 实际 CUDA dispatch 与 annotation 分开，不凭 API 名称推 kernel |
| `309625/309681`，PROJ1 | B1 singleton stride 反例后，仅移除 singleton metadata 保留 HWD 视图 | B1 仍 **314261** 不同；HWD **1204224** 值 exact，实际 **3 mm＋3 add_** | 局部通过后全策略 **4/7→6/7**，state **2→3** 超差，拒绝晋升 |
| `313766/313996/314150` | 固定官方 latent/time/KV，观察实际投影／Qnorm／attention context | 仍有同值下游残差；separate/fused K/V 每例 **9** BF16 值不同；prefix 排列有局部效应 | 不把 sampler、vision 或 QKV 单独当充分原因 |
| `314305/314364/314485/314589` | 单独恢复顺序、真实 Expert0 GU、32 层 teacher/propagated、QKV 单项干预 | 顺序或拆 QKV 单独不充分；GU 在 **20** 组观测输入 exact；局部 block 与传播效应可分 | 不用 hidden L2 计算“解释动作误差百分比” |
| R25 `315024` | 在 split QKV 条件下仅恢复原 prefix 顺序 | Torch2.7 全 32 层／两例／十步 exact；2.11 仍非零 | 两个单项不充分不表示联合无效；仍非 native vision/free-run |
| R26–28 `315256/315323/315395` | 同输入 RMS 八阶段、旧 variance 切口、真实 block 守卫 | **16/16** 首差在 FP32 mean；**320** 自值控制 exact；旧 variance 在 **16/16** suffix 充分，四 block 恢复 | “首差”到“局部充分”分两步证明，保存答案不是算法 |
| R29–31 `315596/315622/316521` | 修正 CPU/GPU annotation 双计数，再观测 layout/launch，并从输入计算显式树 | 两版 kernel 符号／block相同仍 mean 不同；source-old 宽度选树 **16/16** 复现旧 mean，vector4 降序 **16/16** 复现新 mean | 未直接读出内部编译分支；多 eager kernel 原型不是性能候选 |
| R32 `316722` | LIVE input/weight RMS，固定 split QKV／原序／Gold KV／外部轨迹 | **5200** 数组审计，原型每例 **1300/1300** exact 旧 runtime，20 个传播速度 exact | 只在联合固定条件闭合；final head 参与干预，不叫不变负对照 |
| R33 `316975` | 真实 native prefix／free-running policy | 完整机械检查通过仍 **6/7**，state 超差 **3→4**，RMSE变差 | 否定“条件 Expert exact 足以修整策略”；转查上游 |
| R34–35 `317360/317555` | M1 输入来源、M2 调用形式、M3 runtime、M4 重建忠实度；同值最后 Linear 七臂 | 两来源／两形式跨 runtime 阶段 exact；首形式差在 PROJ2 **254／287** 值；biased M49/M147 分组恢复两端点 | 支持有限调用几何效应；显式 BF16 `mm→add` 不复现任一端点，不归因未测 cuBLAS epilogue |
| R36 `317665` | 仅 PROJ2 三个 camera-B1 biased 调用 | 原／候选都 **6/7**，但 state 超差 **3→0**、noise **0→6** | “6/7没变”会隐藏失败迁移；候选拒绝。一个共享 Merger context 有局部 L2 净抵消，不等于策略归因 |
| R37 `317924` | 预注册 E×P 四格，先复现三格再测 both | base/Eonly/Ponly/both **6/6/6/5**；首速度和动作交互都非零，history 新失败 | 非加性不等于有利；**执行预先停止规则，结束组合搜索** |
| R38 `318174` | 最短可锚定四块 ViT 前缀，观察真实端点 | Patch/position/norm1/QKV exact；首差 block0 attention **投影后 2139** 值，block3 **2089847** | attention context／投影／runtime 尚混杂，后层差异不是独立原因 |
| R39 `319502` | 真 context 与投影四格；O6 是否恢复是结果项而非硬设通过 | 实际 context **559** 值不同；O18/O6 都恢复旧投影端点，同值分组6/18无效，FP64交互全零 | 旧 context 值在此局部充分，仍是答案交接，不是 LIVE attention 修复 |
| R40 `319757` | 零权重、同实际 QKV 的真实 row-LSE，context exact 是前置门 | **5811/56448** FP32 行不同；152 个 context 差异行仅 **22** 与 LSE 差异重合 | LSE不是context检测器；不从此定位 FMA/exp2 或宣称库 bug |
| 2026-11-13 人纠偏 | 性能任务不再以旧 runtime EXACT 闭合为前置 | 建立新 public20 scope，保留七例 | 停止继续 LSE/FMA/SASS，回到正式性能 campaign |

R34–40 的原始“结果前预测”与“结果后追加”可以在
[固定研究过程原文](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-12-vla-autotune-research-process.zh-CN.md)
逐项核对，没有改写当时的失败预测。完整历史逐轮正文亦在本文附录原样保留。

### 6.1 失败、部分完成与 NR 仍保留

| 作业 | 实际状态／失败层 | 应保留的结果 |
|---|---|---|
| `308140/309625/309681` | 旧 Hy 完整门／冻结前置门 FAILED | 4/7、B1 拒绝、HWD 6/7 均不追认为成功 |
| `315596` | 首个旧 runtime trace parser 双计数；native 未运行 | partial trace，不是 128 数组完整结果；新 `315622` 不覆盖旧失败 |
| `326708` | FAILED，π0.5 首次严格加载失败 | 已完成 π0／FastWAM 作者与项目点仍有效 |
| `326869` | FAILED，附加作者 E8 二次编译触及进程总预算 | 作者 E10／项目 E8 完成；作者 E8 **NR**。420 s 是总预算，不是 E8 单次延迟 |
| `329232` | DM 作者依赖导入失败，未加载模型 | 无动作／延迟，NR；兼容 overlay 后 `329331` 独立完成 |
| `329215/329441` | FastWAM I8／DM I32 输入 fixture reader 失败 | 不算数值质量失败；分别由新作业完成，不改旧状态 |
| `334440` | DM warmup CPU-list 输入契约失败 | 无有效稳态；新 `334487` 完成 |
| `335329` | DM 私有诊断遗漏 inference mode | 保留 partial APX，不是成功配对；新 `335362` 完成 |
| `356766` | 整批在后续模型超时 | 已完成 Hy／DM 包按各自范围保留，不称全部模型成功 |
| `357202` FastWAM | 基线 `-11`，候选与 bookend 未启动 | 三点全部 NR；同作业已完成 π0.5 四点不抹除 |

传输／SSH 观察超时不自动等于 GPU 作业失败，不能因此重复提交；
调度 COMPLETED 也不自动等于结果质量通过。

## 7. Token、全过程成本与可解释范围

H 负责原生 usage 汇总，本报告**链接同一份生成统计，不复制一个会漂移的总量**：

- [Token 方法与当前截点](2026-11-24-vla-token-usage.zh-CN.md)、
  [JSON](2026-11-24-vla-token-usage.json)；
- [模型全过程](2026-11-24-vla-token-usage-model-lifecycle.csv)：
  接入／迁移／优化／排查／测量／框架六阶段；
- [模型×游](2026-11-24-vla-token-usage-model-stage.csv)；
- [方法×模型迁移成本](2026-11-24-vla-token-usage-method-model.csv)；
- [接入成本趋势](2026-11-24-vla-token-usage-onboarding-trend.csv)。

五项原生计数（input、output、reasoning、cache-read、cache-write）与非缓存合计
以该 JSON／CSV 的定义为准。总处理包含反复缓存读取，不等于新增 token、费用、
VLA 推理 token 或每次模型推理成本。缺失 usage、unknown 阶段和 shared 共有工作
都单列；历史模型归属与空档归属为估算，不伪造成原生计数。

历史截至 2026-11-16 21:02:44（北京时间）的累计快照仍独立保留：
π0 **18.304／0.433 M**、π0.5 **42.478／0.169 M**、
FastWAM **164.187／3.286 M**、Hy **773.134／14.489 M**、
DM0.5 **116.318／4.618 M**，共有 **627.277／28.419 M**（总处理／非缓存）。
这些是旧截点，不与更早 16:48:53、17:48:57 的累计快照相加；
不能用旧累计额与新增窗口重复计费。底层
[旧 JSON](2026-11-16-vla-development-token-usage-dm05-optimized.json)、
[CSV](2026-11-16-vla-development-token-usage-dm05-optimized.csv) 继续保留。

**待验收：**H 收口截点、未归属占比与 missing-usage 审计、方法／游对应的实际
节省毫秒、新模型真实接入起止时间。目前没有证据的单元保持 unknown／NR。
不能从历史单模型累计量推出“新模型接入更便宜”，也不能将全部排查投入除以一个
事后选择的跨作业加速比。完整快照、原生计数源和物理实验结果在 H 合并与复现确认前不删除。

<a id="new-qualification"></a>

## 8. 本轮新验收：留待实测，不用历史替代

本节是填写位置，不是执行结果。最终证据必须是公开可引用的源码／manifest 摘要、
机器可读聚合与实际验证命令，不公开个人文件系统、原始会话日志、权重、动作张量或凭据。

| 项／模型 | 本轮需补的实数与身份 | 当前结论 |
|---|---|---|
| B／Hy 历史来源复现 | 指定旧产品字节、显式接线差异、共同 harness、双侧 ready snapshot、算子 gate、plan v3、reference/candidate/bookend、作者绑定 | **待证**；历史 101.83 ms 不回填 |
| C/F／五模型 | 新冻结 harness 的五模型基线／作者双门／重复测量、真实阶段覆盖率；各 point source identity | **待证** |
| F／π0.5 | INT8 首个较大偏离层、独立中间量对照与质量结论 | **待证**；旧最终动作反例不是首层定位 |
| F／FastWAM | 空编译缓存下冷编译、首调用／capture／warmup、稳态、失败终态和重复 campaign；`359953` 已提交 | **待证**；纯解释器观测器反例和热副本 `357482` 均不能替代 GPU 完成证据 |
| E／新 attention／compiler 方法 | 真实算子输入、适用范围、算子／capture gate、实现与模型级 v3 campaign；失败也保留 | **待证**；工具登记或局部运行不是模型 winner |
| G／SmolVLA | 干净作者 source pin、权重／输入身份、原作者完整边界；项目 eager/graph、变化输入传播、模型动作头契约、重复双门槛与优化 | **NR／待证，尚无 G GPU 结果** |
| H／成本 | 本轮完成截点、六阶段与四表、missing/unknown/shared、未归属＜5% 的实际检查 | **待收口**；生成中的数字不是最终结账 |
| I/J/K | 技能流程与许可、报告来源／链接、精确 staged tree 的 CPU 集、当前 best/失败清单、人工 H4 | **待主代理最终验证／人批准** |

当前进度以[平台验收清单](../vla_autotune_progress.zh-CN.md)为准。
工作分支的 keep 不是合入或默认配置批准。本次文档整合没有启动新 GPU、
提交／推送或为旧测量补造强 schema。

## 9. 公开证据、文献与历史迁移

### 9.1 机器真值与固定原文

- [11 月 14 日原始 JSON](2026-11-14-vla-latency-comparison.json)／
  [CSV](2026-11-14-vla-latency-comparison.csv)：29 条历史点，保留旧值、NR、
  method、来源与边界，不用新数据覆盖。
- [固定阶段报告原文](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-09-vla-autotune-report.zh-CN.md)、
  [固定研究过程原文](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-12-vla-autotune-research-process.zh-CN.md)、
  [固定延迟报告原文](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-14-vla-latency-comparison.zh-CN.md)：
  可核对原有预测、范围、测试统计及未逐项重述的历史细节；不再维护多份当前结论。
- 五份 11 月 23 日原报告在同一固定提交下：
  [DM0.5](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-23-dm05-author-loop.zh-CN.md)、
  [FastWAM](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-23-fastwam-text-kv-campaign.zh-CN.md)、
  [Hy 配置](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-23-hy-config-loop.zh-CN.md)、
  [Hy 融合](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-23-hy-sm80-fusion.zh-CN.md)、
  [π0.5 INT8](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-23-pi05-sm80-int8.zh-CN.md)。
- [清理映射](../vla_autotune_cleanup.zh-CN.md)解释撤下哪些入口、为何撤下、如何取回；
  token JSON/CSV、作者计时入口、工具来源、许可证、snapshot 机制与物理快照不在本次删除范围。

### 9.2 官方发布值只作背景

| 作者来源 | 已发布值 | 条件／不可比点 |
|---|---:|---|
| [π0 论文 Appendix A-D／Table I](https://arxiv.org/html/2410.24164v4#A1.SS4) | 73 ms on-board／86 ms off-board，RTX4090 | 三相机／H50／E10；dtype／统计量 NR，off-board 多 13 ms 网络 |
| [π0.5 作者 RTC 论文 vanilla 行](https://arxiv.org/html/2506.07339v2#A1.SS3) | 76 ms，RTX4090 | BF16／H50／5步、5 warmup 后10次 mean、GPU-only；不是 RTC 97 ms，不等同本轮公开 OpenPI 环境 |
| [FastWAM 论文 v2](https://arxiv.org/html/2603.16666v2#S4.SS1)与[作者澄清](https://github.com/yuantianyuan01/FastWAM/issues/34#issuecomment-4250694653) | 190 ms，RTX5090D V2 | H32／10步、含T5、不用compile/graph、384×320；不是本项目224×448 |
| [FastWAM 固定 README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md) | compiled 110 ms／RTX4090；210 ms／H20，旧值190／470 ms | 含 text/VAE，checkpoint／步数／统计量等 NR，不倒填默认 |
| [Hy 论文 v2](https://arxiv.org/html/2606.14409v2)／固定作者仓库 | 完整推理延迟 NR | 任务完成秒数或 servo Hz 不当作单次推理延迟 |

不把这些异硬件、异几何、异边界的文献值除进 A800 加速比。
作者自测也不是论文环境逐项复现：π0 JAX0.5.3／Flax0.10.2／Orbax0.11.13；
π0.5 Torch2.7.1+cu126／Transformers4.53.2＋作者 overlay；
Hy Torch2.7.1+cu128／FA2 2.7.4.post1；FastWAM 作者与项目 Torch2.11.0+cu128。

<a id="historical-record"></a>

## 附录：完整历史逐轮记录

下文保留旧实验正文与数字，**不是当前状态覆盖层**。其中“下一步”“尚未有胜点”
是写作当时的判断；当前结论以本文上半部及新验收证据为准。旧 v1/v2、旧 EXACT
门和未发布的私有张量不因迁移获得新身份。历史原文仅调整相对链接以适应新目录。

<details>
<summary>展开历史逐轮记录：π0／π0.5、FastWAM graph 与采样、Hy R1—R40</summary>

### 历史记录原入口说明

**Current Hy objective, changed by the user on2026-11-13:** performance/quality
tradeoff like the other models, not old-runtime/intermediate EXACT closure.
New campaigns use normalizedpublic20 relativeL2<=.05/cosine>=.999 against a
fixed native semantic-BF16 Euler10 reference; all seven cases remain required.
See the [explicit contract](../../examples/vla_autotune/hyvla_tradeoff_quality.json)
and [current objective](../vla_autotune.md#hy-performancequality-objective-2026-11-13-revision).
The stricter padded/official gates, failures and R1–R40 precision findings below
are historical evidence, not a prerequisite for future speed search and not
retroactively requalified. LSE/FMA/SASS research is no longer the active task.

For the Chinese research narrative—observations, competing hypotheses,
discriminating experiments, negative findings and subsequent revisions—see
[the historical research-process record](https://github.com/20261023/FlashRT/blob/199376281801427abccbb307056b27e74eab1429/docs/reports/2026-11-12-vla-autotune-research-process.zh-CN.md).

These are engineering observations from a bounded A800 iteration, **not** a
blanket hardware qualification, a model leaderboard, or robot task-success
results. Current validation and profiling scripts are under
[`examples/vla_autotune`](../../examples/vla_autotune/README.md). Retired Hy diagnostics
and the old graph pilot remain available at their recorded Git revisions; the
[cleanup inventory](../vla_autotune_cleanup.zh-CN.md) maps their provenance.
Weights, private
environments, raw tensors, traces and generated result directories are not
distributed with these notes.

## Experimental boundary

- One NVIDIA A800-SXM4-80GB allocation; serial fresh processes for compared points.
- Existing native execution runtime: Python3.11, Torch2.11+cu128. Hy recorded
  official references use their separate Torch2.7.1 runtime.
- Formal latency includes frontend validation/transfers/model/completed CPU
  output; load, preparation/capture, warmup and post-return artifact checks are
  separate. Independent instrumented traces are never formal latency samples.
- Action arrays and actual NFE are retained. Fixed-input numerical similarity
  is a scoped proxy, not robot performance. No tolerance was relaxed to rescue
  a fast failing point.
- Earlier runs used version1 result records. Their historical observations were
  not backfilled into version2 identities. The strict v2 campaign below was
  actually run with instance/full-candidate/confirmed-job binding.
- Runs used immutable work-in-progress source snapshots. Later Git checkpoints
  package the source; they do not retroactively replace the original provenance.

## Pi0: real FP16/JAX seam, not an observed acceleration

Job307979 compared Euler2, ProbeFlow2 and Euler2 bookend on the resident Pi0
checkpoint. Ten samples per point: medians26.1060/26.7000/26.3815ms, both2NFE.
ProbeFlow public7 relativeL2=.000375709 and cosine=.999999945 passed that run's
explicit numerical proxy, but was slower. Euler2 was retained as the control.
The slight FP16 beta-zero head/external-integration rounding difference was
measured, not asserted to be bitwise equality.

This was a two-step execution comparison, not a qualification of two-step
sampling against the policy's default quality or a complete Pi0 search.

## Pi0.5: multi-request gating reverses a single-input choice

The first campaign (job307967,20samples/point) measured:

| Point | Actual NFE | Median ms | Public7 relativeL2 | Proxy |
|---|---:|---:|---:|---|
| Euler10 control | 10 | 41.9109 | reference | control |
| Euler2 | 2 | 28.5946 | .224762 | fail |
| Euler5 | 5 | 33.5839 | .0607605 | fail |
| ProbeFlow eps=.002,max20 | 21 | 60.7568 | .0342706 | pass, slower |
| ProbeFlow eps=.008,max20 | 13 | 46.7985 | .0162128 | pass, slower |

The fixed synthetic fixture had911prefix rows:768vision and143prompt/state
rows. Diagnostic prefix intervals were about24ms; each full Action Expert
evaluation about1.65–1.67ms. These intervals include submission/dependency gaps
and are not kernel-duration sums or proof of compute/memory boundedness.

The source/runtime cost ledger gives3.11794TFLOPs for the17prefix FFNs alone
at M911,K2048,N16384, versus31.1427GFLOPs projections/FFNs plus7.08526GFLOPs
attention products per Action Expert evaluation at M50. Logical linear-weight
bytes are not measured HBM traffic.

Round2 (job308132) found Euler7 at36.9639ms and2.56%relativeL2 for one input.
Post-ViT pool2 really changed911->335prefix rows and recorded28.8652ms, but its
33.21%relativeL2 failed. It reduces prefix GEMM M, not the ViT already performed.
The fastest raw point was therefore rejected rather than promoted.

The strict v2 round3 (job308244) broadened to original input, two new fixed
noises and RGBbrightness+8, keeping prompt/state/tensor geometry unchanged.
Each point had8warmups and40total samples,10per request. Gate was explicitly
**normalized public7 relativeL2<=.05 and cosine>=.999** across all four requests.
This is an engineering sensitivity budget, not a universal VLA tolerance.

| Point | Actual NFE | Median ms | Worst relativeL2 | Min cosine | Disposition |
|---|---|---:|---:|---:|---|
| Euler10 reference | 10 | 41.6211 | reference | reference | control |
| Euler7 | 7 | 36.8776 | .665398 | .767542 | reject |
| **Euler8** | **8** | **38.4342** | **.0361504** | **.999482** | **select in this scope** |
| PF eps=.012,max10 | 7/9 | 38.8245 | .129385 | .992178 | reject |
| PF eps=.016,max10 | 5/7 | 37.0234 | .129385 | .992178 | reject |
| Euler10 bookend | 10 | 41.6857 | .0 | ~1 | pass |

Bookend drift was+.155%. The new noise20261111 invalidated Euler7 and both PF
points. Euler8 retained about7.7%lower median latency only within this four-case
proxy. The controller's full-workload coverage and worst-case gate mattered;
the cause of every coarse-trajectory error was not claimed to be isolated.

The real LeRobot release declares ACTION=MEAN_STD despite storing quantiles.
The unified backend now decodes from raw actions using that declaration. Earlier
normalized raw arrays remain numerical evidence, but old legacy-decoded outputs
must not be described as corrected robot commands.

## FastWAM: remove repeated planning, then test graph execution fairly

This adapter is **partial native Linear execution** on the pinned upstream
first-frame action graph, not a full native port. Real T5/VAE/MoT/proprio assets
were used with a deterministic synthetic two-camera input, normalized proprio
and supplied CPU-generated noise. No future-video rollout or task success was
evaluated. All numbers here use Euler2/shift5, not default-policy quality.

The legacy BF16 NT+bias call rebuilt descriptors/layouts/preferences and queried
eight heuristics per enqueue. Job308343 performed20samples/point after3warmups:

| Point | Median ms | P95 ms |
|---|---:|---:|
| Partial-native legacy | 202.5082 | 204.6320 |
| Partial-native cached | 154.7789 | 157.3116 |
| Upstream Torch | 122.5195 | 125.1433 |
| Legacy bookend | 201.9628 | 203.4542 |

All2NFE/oneprefill. Cached and legacy actions were bitwise equal, as were their
bookend outputs. Cached firstcall built19plans; all20steady requests recorded
0builds/queries and919native Linear enqueues across612modules. The47.729ms
median reduction is23.57%; upstream was still faster. Workspace limits differ
(cached64MiB versus legacy256MiB), so the entire difference cannot be assigned
numerically to host planning without an independent algorithm/workspace ablation.

Diagnostic job308460 then found identical920Linear-call/1.22055TFLOP ledgers
(cached919native+oneTorch proprio, upstream920Torch). Logical weights14.0704GB
are not HBM bytes. Both instrumented requests had about51.79ms summed CUDA
kernel duration but long different gaps/spans. cProfile and Kineto overhead
prevent using those walltimes as formal latency or a roofline conclusion.

The private raw-velocity graph pilot was therefore applied **equally** to both
engines in job308611,20samples each, one allocation:

| Point | Median ms | P95 ms |
|---|---:|---:|
| Cached eager | 147.8591 | 149.3722 |
| Cached rawhead graph | 75.8655 | 76.4702 |
| Upstream eager | 118.8005 | 120.3240 |
| **Upstream rawhead graph** | **69.0292** | **69.3196** |
| Cached eager bookend | 148.0505 | 149.6756 |

Graph/eager outputs were bitwise equal within each engine, including six
same-weight original/noise/proprio/image/context/mask controls. Every changed
control affected the output, ruling out a vacuous unchanged-input check. Each
steady request refreshed conditions once and replayed the velocity graph twice;
sampler/integration, VAE and per-request video prefill stayed outside capture.
Owned graph cleanup passed. Coldwarmup/encoding work was retained separately.

Cached graph reduced its eager median48.69%; upstream graph41.89%. The fastest
tested path was **upstream+graph**, not a preset native winner. Cached/upstream
retain their pre-existing1.45%relativeL2 difference; this experiment only proves
graph preservation of each engine, not cross-engine official parity.

The graph pilot remains fixed-geometry/Euler2 experimental code. It is not a
registered production option or new default; adaptive PF, ordinary step counts,
other geometries/hardware and broader quality need validation before integration.

### Opt-in library-owned raw-field graph

Job309097 later exercised the **new library integration**, frozen at `a5140aac`,
not the pilot's upstream method override. Default eager behavior was unchanged.
One A800 allocation ran fresh serial processes, three additional whole-request
warmups after the separately measured first call, and ten formal samples/point:

| Point | Actual NFE | Median ms | P95 ms |
|---|---:|---:|---:|
| Independent direct-upstream Euler10 | 10 | 411.4261 | 412.9479 |
| Common Torch pipeline Euler10 eager | 10 | 389.6625 | 392.2531 |
| **Common Torch pipeline Euler10 graph** | **10** | **146.6673** | **147.4356** |
| Partial-native cached Euler10 eager | 10 | 515.0250 | 515.6989 |
| Partial-native cached Euler10 graph | 10 | 158.2449 | 159.1946 |
| Torch PF max10,eps=.008 eager | 2 | 114.5456 | 115.4969 |
| Torch PF max10,eps=.008 graph | 2 | 66.0323 | 66.3118 |
| Cached PF max10,eps=.008 eager | 2 | 147.6405 | 148.5308 |
| Cached PF max10,eps=.008 graph | 2 | 77.0178 | 79.1293 |
| Torch Euler10 eager bookend | 10 | 388.8984 | 390.8952 |

Independent CPU rereads verified actual input arrays, checkpoint path/stat and
resolved model configuration identity across all ten points. All four same-engine
graph/eager formal action arrays and sampler metadata were exact. Each graph
also passed six same-weight graph/eager original/noise/proprio/image/context/mask
pairs; every changed control affected the output. Each request had one video
prefill. Owned cleanup succeeded in every library point.

Torch Euler10 graph was also bitwise equal to the independent direct-upstream
Euler10 original-input output. Cached Euler10 eager/graph retained the same
cross-engine difference, relativeL2=.00594835 and maxabs=.0078125; that is not
graph error or an official-parity guarantee. Torch eager bookend drift was
-.1961%. Graph reduced its own Torch Euler10 median62.36% and cached Euler10
median69.27%; Torch+graph remained faster than partial-native+graph.

Cached Euler10 steady metadata recorded305CPU native enqueues (prefill), but
ten raw graph replays each encoded307native Linear calls:3070logical native
Linear calls in the action field, **not zero GPU work or a kernel-count claim**.
Steady plan builds/heuristic queries were zero. The first capture's three eager
head warmups and one encoded body were recorded separately from sampler NFE.

Two additional **real-CUDA** regressions passed in this allocation: raw capture/
refresh/cleanup without device/host global cache flushing, and eager standalone
velocity completion before close released internal timestep storage. This
followed RED->GREEN fixes for both findings from independent static review.

The PF configuration above was adaptive, but **all six tested requests routed
to2NFE**. It does not validate dense restart. Its original-input output differed
from same-engine Euler10 by26.36%relativeL2 (Torch) or25.93%(cached). No robot
quality or FastWAM sampler-approximation acceptance gate was defined, so the
66ms point must not be selected as a quality-qualified policy merely by speed.
Dense PF, default20steps, other geometries/hardware and robot success remain
unqualified by this run. The option is still disabled by default.

Two subsequent parameter-only jobs reused the **unchanged** `round012/a5140aac`
source; neither edited or retrospectively relabeled job309097:

| Job / point | Scheduled steps | Actual NFE | Eager median ms | Graph median ms |
|---|---:|---:|---:|---:|
| 309138 Torch PF max10,eps=.0001 | 8 | 9 | 390.1538 | 140.6916 |
| 309138 cached PF max10,eps=.0001 | 8 | 9 | 489.7465 | 151.5507 |
| 309187 Torch Euler20 | 20 | 20 | 804.6938 | 255.0711 |
| 309187 cached Euler20 | 20 | 20 | 971.2947 | 262.6851 |

Job309138 really exercised discarded-probe/dense restart: both engines selected
8steps/9NFE on the original input; changed inputs selected8/10steps with9/11NFE.
Torch proprio selected10/11 while cached proprio selected8/9, illustrating that
small engine numerical differences can change an adaptive routing decision.
Same-engine graph/eager arrays, schedules and all twelve refresh pairs were
exact; all ten mutants affected output and cleanup passed. On identity-checked
inputs, numerical-only comparison to saved same-engine Euler10 refresh outputs
gave worst relativeL2=.0408773 (Torch) and.0436711 (cached) across six cases.
That comparison spans jobs and is **not** a cross-job latency ranking or a robot
quality gate.

Job309187 covered the library's default **20-step budget** with independent
direct-upstream Euler20 (median797.6525ms), Torch and cached eager/graph. Actual
NFE20, one prefill/request, complete raw sample equality, both six-case refresh
sets and owned cleanup all passed. Torch original-input output was bitwise equal
to the independent upstream control; cached retained relativeL2=.00471522 versus
that control, identically in eager/graph. Own-engine median reductions were
68.30% and72.96%; cached graph still was not faster than Torch graph.

For cached Euler20,305steady host enqueues do not replace the20graph replays
containing6140logical native Linear calls. Cold3head warmups/oneencoded body remain
separate. Euler10 original-input output differed from same-engine Euler20 by
6.55%relativeL2 (Torch) or6.51%(cached), so even an exactly preserved faster
10-step graph is not automatically qualified as a20-step policy replacement.
The measurements establish execution preservation, not task success or a
universal sampler budget. Other geometry/hardware remain unqualified.

### Bound v2 FastWAM graph/sampler search

Job309300 exercised the actual controller/worker path after fixing graph replay
counters being mistaken for immutable backend identity. Its immutable source
was `round014/baef55ff`; full Candidate, campaign-instance and confirmed job
binding were enforced, not backfilled into older standalone validation records.

Four requests held geometry/context/proprio fixed: original prepared input/noise,
new CPU-Torch noises20261110/20261111, and imagebrightness+.1 clipped to[-1,1].
This is synthetic sensitivity, not representative task evaluation. Before the
run, the explicit engineering gate was **all7normalized channels, worst relative
L2<=.05 and minimum cosine>=.999 versus Torch graph Euler20 across all four**.
It is not a universal VLA tolerance; no threshold was loosened after rejection.
Hy's original oracle gate and all library defaults remained unchanged.

Each point used8excluded warmups and40total measured requests,10per case, in
fresh serial processes within one A800 allocation:

| Graph point | Actual NFE | Median ms | Worst relative L2 | Min cosine | Disposition |
|---|---|---:|---:|---:|---|
| Torch Euler20 reference | 20 | 252.0941 | 0 | ~1 | control |
| Torch Euler10 | 10 | 150.3535 | .107391 | .994934 | reject |
| **Torch Euler14** | **14** | **191.7414** | **.0372622** | **.999441** | **select in this scope** |
| Torch Euler16 | 16 | 211.8832 | .0220205 | .999802 | pass, slower |
| Torch Euler18 | 18 | 232.7558 | .0119571 | .999930 | pass, slower |
| Torch PF eps=.00005,max20 | 15/17/21 | 221.6153 | .0303350 | .999632 | pass, slower |
| Torch PF eps=.0001,max20 | 9/21 | 139.4244 | .104311 | .995595 | reject |
| Cached native Euler16 | 16 | 223.8490 | .0203627 | .999831 | pass, slower |
| Cached native Euler18 | 18 | 243.2042 | .0110135 | .999942 | pass, slower |
| Cached native Euler20 | 20 | 263.3780 | .00555848 | .999985 | pass, slower |
| Torch Euler20 bookend | 20 | 251.8910 | 0 | ~1 | pass |

All11complete raw bundles were fetched. Independent CPU checks covered every
saved action, request coverage, deterministic repeats, numerical metrics, full
candidate/instance/job identity, actual NFE/replay equality, one prefill/request,
and steady zero captures/builds/queries before strict ingestion. The ledger
recorded9validated/2rejected; bookend drift was-.08054%. The selected median was
23.94%lower than the graph Euler20 reference **within this four-case proxy**.
The139ms PF point was rejected rather than called the fastest acceptable policy.

The separate profile retained raw-velocity CUDA-event intervals and explicitly
unknown bound classification; missing prefix-stage evidence was not zeroed or
inferred from a residual walltime. No robot success, global parameter optimum,
cross-job performance ranking, or automatic default promotion is claimed.

The next bound refinement (job309386) reused the **same immutable round014
source, workload and exact gate**, with a new campaign instance and confirmed
job. It tested the observed10-fail/14-pass region rather than loosening a budget:

| Graph point | Actual NFE | Median ms | P95 ms | Worst relative L2 | Min cosine | Disposition |
|---|---|---:|---:|---:|---:|---|
| Torch Euler20 reference | 20 | 251.6074 | 252.3159 | 0 | ~1 | control |
| Torch Euler11 | 11 | 159.8610 | 160.8566 | .0784776 | .997424 | reject |
| Torch Euler12 | 12 | 170.1411 | 171.7190 | .0560137 | .998713 | reject |
| **Torch Euler13** | **13** | **179.7011** | **180.0663** | **.0453660** | **.999179** | **select in this scope** |
| Torch Euler14 | 14 | 190.6152 | 190.9841 | .0372622 | .999441 | pass, slower |
| Torch PF eps=.00006,max20 | 13/15/21 | 200.5839 | 262.2253 | .0461508 | .999132 | pass, slower |
| Torch PF eps=.00007,max20 | 11/13/21 | 180.5588 | 262.7861 | .0702523 | .998064 | reject |
| Cached native Euler11 | 11 | 170.5311 | 171.0508 | .0744344 | .997701 | reject |
| Cached native Euler12 | 12 | 180.7251 | 181.2629 | .0544527 | .998793 | reject |
| Cached native Euler13 | 13 | 190.8152 | 191.5830 | .0428495 | .999268 | pass, slower |
| Cached native Euler14 | 14 | 201.5966 | 202.1167 | .0340776 | .999540 | pass, slower |
| Torch Euler20 bookend | 20 | 252.6042 | 252.9031 | 0 | ~1 | pass |

Every40sample bundle was independently checked as in round1 before strict
ingestion:7validated/5rejected, bookend drift+.39614%. Euler13 reduced the
same-allocation reference median28.58%. Its per-case relativeL2 values were
.0328582/.0453660/.0382539/.0346602 for original/noise20261110/noise20261111/image.
Paired native points sometimes had smaller error, but did not beat the selected
Torch point's latency; engine choice was measured, not presumed.

The PF points' P95 reflects their actual21NFE requests, not the requested max
misreported as a fixed work budget. No new tail-latency constraint was invented
afterward; raw samples/NFE and the existing median objective stay explicit.
This closes a finite controlled-proxy refinement, **not** a global optimum,
held-out dataset result, robot-success claim or automatic policy-default change.

Job313779 subsequently audited the prior choice with eight additional synthetic
numeric requests, on the unchanged `round014/baef55ff` implementation. These were
four new fixed noises, negative image brightness, normalized proprio, T5-context
and mask perturbations, not representative robot or natural-language tasks. The
three candidates (Euler13/14/16) were predeclared; no parameter refinement was
performed after observing the results. Each point had8warmups/80total requests,
ten per case, and the same L2<=.05/cos>=.999 limits in a new explicit scope:

| Torch graph point | Median ms | P95 ms | Worst relativeL2 | Min cosine | New-scope result |
|---|---:|---:|---:|---:|---|
| Euler20 reference | 249.4247 | 250.8880 | 0 | ~1 | retain control |
| Euler13 | 178.7498 | 180.1187 | .194629 | .980923 | reject |
| Euler14 | 189.9475 | 191.0666 | .117659 | .993055 | reject |
| Euler16 | 209.1321 | 209.9623 | .179380 | .983885 | reject |
| Euler20 bookend | 249.9272 | 250.8169 | 0 | ~1 | pass |

All three failures occurred on `context-minus0p0625`; the other seven cases
passed. Directly perturbing prepared T5 vectors, including previously zero
padding, is an off-distribution numerical stress test, not a new prompt sampled
from a task dataset. It nevertheless falsifies a universal numerical-budget
claim for Euler13 and shows that error need not decrease monotonically with
step count. It does not establish failure on representative natural-language
tasks. The earlier four-case result remains true only in its historical scope.

Independent raw-array/candidate-instance-job/coverage/NFE/graph checks preceded
strict ingestion:2validated/3rejected, reference bookend drift+.20146%. No
accelerated candidate passed this larger scope; the controller retained Euler20.
No tolerance change, automatic default promotion or further tuning on these
eight cases followed. The graph-preservation qualification is separate and is
not contradicted by a sampler-approximation quality failure.

## Hy: keep mechanical checks and official acceptance separate

Semantic BF16 execution, fixed-noise replay and Euler2/PF2/graph-eager mechanics
passed their measured checks. It was not assumed that BF16 labels imply parity.

Job308140 tested the optional official-local cast profile under the original
elementwise `.01+.01*abs(reference)` and cosine>=.999 gate. Baseline,
episode_start,current,prompt passed; history/state/noise failed with4/2/6public
elements outside tolerance. Job status correctly remained FAILED/1:0. The gate
was not loosened and the option was not made default or called official-exact.

The corrected frozen-boundary diagnostic (job308690) reproduced three old
Torch2.7.1 oracles exactly and native eager/prior graph exactly, matched input
identity and preserved prefix invariance before frozen evaluation. Prepared
images/state/tokens/masks matched. Sorting-before prefix embeddings already
differed; replacing native KV with official KV reduced but did not eliminate
raw-velocity differences. This is localization evidence, not a seven-case fix.

CPU inspection of saved tensors found all190464text/special-prefix elements
exact, while266483/301056vision-prefix elements differed. Both implementations'
vision-prefix slices exactly equaled their own merger outputs. Merger inputs
were already different (per-camera relativeL2 about.0057–.0075), so a transformer
sorting/projection hypothesis cannot explain the entire earlier visual mismatch.
Live merger layouts are non-contiguous in both implementations; canonical CPU
artifact strides must never be substituted for them in a layout experiment.

The next **frozen single-operator** diagnostic (job309016, source checkpoint
`d91ef23b`) ran the recorded Torch2.7.1 and native Torch2.11 runtimes serially in
one A800 allocation. It constructed no full policy and changed no action gate.
All seven saved operator arms were independently reread on CPU and were bitwise
identical **across these two runtimes**. Relative to the accepted official trace:

| Frozen control | Unequal elements | Relative L2 | Maxabs |
|---|---:|---:|---:|
| PatchConv, per-camera batch6 | 0 | 0 | 0 |
| PatchConv, combined batch18 | 0 | 0 | 0 |
| Position interpolation, direct BF16 | 0 | 0 | 0 |
| Position interpolation, FP32 then BF16 | 0 | 0 | 0 |
| proj1, official per-camera live views | 0 | 0 | 0 |
| proj1, native combined non-contiguous layout | 314260/1204224 | .002465559 | 1 |
| proj1, contiguous combined batch control | 440808/1204224 | .003142936 | 1 |

Thus patch batching and the interpolation cast hypothesis do **not** explain the
observed difference for these exact frozen inputs, weights and targets. No
speculative interpolation/profile change was made. This is not a universal
statement about other input geometry or hardware.

For proj1, CPU profiler lowering actually recorded three `mm` plus three `add_`
calls for the official views, versus one `bmm` plus one `add_` for the native
batched layout. The contiguous control used one `addmm`. These are operation
counts, not GPU timings or a measured reduction-kernel/algorithm attribution.
They prove an independent batching/layout contribution with values held fixed;
blindly adding `.contiguous()` does not restore official equality here.

This does **not** repair the earlier mismatch already present at the ViT exit,
or qualify the seven original action fixtures. Attention/block arithmetic after
the exact patch/position inputs still needs localization before a product fix.

Job309182 (`round013/ddbb5faa`) then reconstructed block0 input from those accepted
patch+position values and loaded only norm1/QKV weight and bias. LayerNorm
camera6/batch18 and QKV Linear camera6/batch18 on identical normalized values
were exact; the reconstructed inputs, norms, QKV and auto-SDPA outputs were also
bitwise equal across Torch2.7.1+cu128 and2.11.0+cu128, independently reread from
raw tensors. FA2 2.7.4.post1 per-camera/combined attention was exact too.

Holding those official QKV values fixed, both Torch runtimes' auto-SDPA differed
from the external FA2 arm at559/4064256elements, affecting148/3528tokens:
maxabs=.00390625, relativeL2=2.94742066e-5. SDPA camera6/batch18 were exact to each
other. Thus this frozen point demonstrates an attention-API implementation
contribution, not a measured runtime-version or batch contribution. It does not
identify the actual auto-selected CUDA kernel merely by the SDPA API name.

No recorded block0 output oracle was present in the earlier traces, so this
reconstruction/operator comparison must not be labeled reproduced block0 or full
policy parity. Nor does its small local residual quantify amplification through
projection/residual/MLP/later blocks or explain all the separate proj1 difference.
No product attention backend, numerical profile, default or original gate was
changed from this observation. The optional frozen-backend probe is a separate
diagnostic experiment, not a pre-qualified product implementation.

The explicit backend matrix subsequently ran as job309409 on
`round015/d78ecfb0`. All four one-hot backends executed at grouping6/18 in both
runtimes with unchanged frozen official QKV, layout, dtype, scale and masking:

| Torch SDPA backend | Unequal elements versus external FA2 | Maxabs | Relative L2 | Exact versus AUTO |
|---|---:|---:|---:|---|
| FLASH_ATTENTION | 559 | .00390625 | 2.94742066e-5 | yes |
| EFFICIENT_ATTENTION | 853921 | .015625 | .000971597 | no |
| CUDNN_ATTENTION | 905372 | .015625 | .001023550 | no |
| MATH | 925829 | .015625 | .001049684 | no |

Independent raw-tensor checks confirmed all15common output arms exact across
Torch2.7.1/cuDNN90701 and Torch2.11.0/cuDNN91900. Grouping6/18 did not change
these operator outputs. **None** of these forced Torch backends reproduced the
external FA2 arm exactly; therefore no exact-candidate confirmation branch was
eligible. Selecting Math or another backend merely because it sounds more
precise would move this frozen point farther from that reference.

The separate AUTO trace observed three
`aten::_scaled_dot_product_flash_attention` calls and three actual CUDA
`pytorch_flash::flash_fwd_kernel` launches, plus the concatenation kernel. That
is direct dispatch evidence for this region, not inference from the API name
or a performance measurement. Its original exported name map also contained a
mirrored GPU `record_function` annotation. That annotation is **not a kernel**:
it is excluded in independent analysis, and the summary-classification source
was corrected with RED->GREEN tests to separate user annotations and keep
annotation-only traces unverified. Original job artifacts were not rewritten;
the numerical backend comparisons above are unaffected.

AUTO/Flash is the closest tested Torch arm here, still not FA2-exact. Product
attention, numerical profiles, original action tolerances and the failed
seven-fixture qualification remain unchanged. The independent merger PROJ1
batch/layout contribution remains a separate possible ablation, not a repaired
ViT or policy inferred from this operator matrix.

Job309625 (`round016/aa3da4b3`) tested actual native per-camera B1 slices as an
EXACT frozen precondition **before** any full-policy construction. It failed
that gate and exited1:0:314261/1204224unequal elements, maxabs1, relativeL2
.00246554148, versus314260/.00246555918 for native B3. The two native arms differed
at160elements (maxabs.125). No baseline/candidate seven-fixture model runs took
place; neither the old failed job nor original action gate was reclassified.
The corrected profiler filter did separately observe actual CUDA kernels while
excluding the mirrored annotation in this job.

Torch2.11's `should_fold` checks leading stride relations even for singleton
dimensions. Thus official B1 leading stride196 satisfies14*14 while native
225792 does not. The next example-only `--fold-spatial` candidate removes that
singleton axis before Linear, preserving the non-contiguous HWD view/storage/
offset and restoring B1 afterwards. It retains separate bias addition rather
than copying into a contiguous or 2D+bias-fused path. This source-based hypothesis
must pass a **new** exact GPU precondition, with the rejected B1/B3 controls kept,
before any full-policy ablation. No production/profile/attention/default or
quality threshold is changed, and no GPU success is inferred from CPU tests.

That follow-up actually ran as job309681 on `round017/97218269`. The HWD frozen
candidate was **bitwise equal** to the saved official projection at all1204224
elements, while the retained native B1/B3 controls still differed314261/314260.
Observed lowering was3`mm`+3`add_` for HWD, versus3`bmm`+3`add_` for B1 and
1`bmm`+1`add_` for B3. Actual HWD views retained strides[14,1,196] and offsets
0/225792/451584. Source folding reasoning was therefore tested, not assumed.

All14fresh full-policy processes then completed their mechanical checks and
owned cleanup. Every baseline reproduced old308140 inputs, oracles, configuration,
runtime, checkpoint and saved actions exactly; snapshot source differences were
explicitly reported. Candidate fixed-noise replay and graph/eager outputs were
exact in every case, with actual Euler10NFE. Independent raw-array recomputation
confirmed the original padded40x32 gate and separate public40x20 dispositions:

| Fixture | Baseline elements outside original gate | HWD candidate elements outside gate | Baseline maxabs | HWD maxabs | Candidate disposition |
|---|---:|---:|---:|---:|---|
| baseline | 0 | 0 | .00634766 | .00854492 | pass |
| episode_start | 0 | 0 | .01171875 | .01171875 | pass |
| current | 0 | 0 | .00830078 | .01025391 | pass |
| history | 4 | 0 | .01196289 | .00993347 | pass |
| prompt | 0 | 0 | .00683594 | .01171875 | pass |
| state | 2 | **3** | .01123047 | **.01663208** | **reject** |
| noise | 6 | 0 | .02349854 | .01318359 | pass |

All outliers above are in public20 as well; no padded-only failure was hidden
by public truncation. The gate is elementwise`.01+.01*abs(reference)` plus
cosine>=.999, not a uniform maxabs threshold: a maxabs>.01 can still pass.
State candidate failures are at zero-based(step,channel)(14,9),(15,9),(35,9),
with absdeltas .01342773/.01663208/.01406860 versus allowed
.01331171/.01191901/.01136251. Its cosine .99998731 and lower RMSE than baseline
do not override these violations. RMSE also increased in several already-passing
fixtures, so neither a monotonic accuracy improvement nor full official equality
follows from fixing this one frozen operator.

The resulting4/7->6/7 is a **partial causal ablation**, not a complete gate fix:
the job correctly remains FAILED1:0 with aggregate exit1. No production option,
default, tolerance, source snapshot or old failed qualification was changed.
The prototype is not promoted. Merger-before-input ViT differences remain a
separate known source; the next useful work is boundary localization of the state
residual, not another blind cast/backend/tolerance change. This is still recorded
numerical testing, not representative robot task-success qualification.

Job313766 (`round018/2959bd87`) continued localization with an explicitly eager
folded candidate, not a new policy gate. A fresh exact projection precondition
passed. Baseline/state actions exactly reproduced the corresponding309681
candidate arrays; each ten-step native raw trajectory replayed exactly, and
recorded increments/latents also matched the characterized BF16-scalar/BF16-
velocity then FP32-add recipe. Actual time embeddings matched the embeddings
recomputed at the precise historical official schedule. Inputs, masks and the
state embedding were exact; prefix remained invariant between these two cases.

Teacher-forced comparisons held each official latent/time fixed and changed only
native versus sorted/expanded official prefix KV, with independent scratch and
runtime prefix/read-only guards. Independent raw checks found:

| Case | Native-KV raw-velocity relativeL2 range | Official-KV relativeL2 range |
|---|---:|---:|
| baseline | .00422149–.00696753 | .00329106–.00537082 |
| state | .00399431–.00657527 | .00313123–.00510265 |

Every official-KV velocity arm was also bitwise equal to its historical308690
native injection counterpart. Thus fixing this projector did not eliminate the
downstream residual at identical external KV. Current prefix embedding/K/V
relativeL2 values were .01257736/.00915026/.01402888; the state embedding was
exact. This is evidence against attributing all remaining error to vision/KV or
the initial state projection, not proof of a particular Expert-layer cause.
No free-running official-KV policy rollout or new seven-fixture acceptance was
performed. The prior6/7 qualification failure remains unchanged. Final raw files
preceded an atomic completion marker after cleanup; both cases were independently
audited from saved tensors rather than accepted from scheduler exit alone.

The step0-only frozen suffix/Expert0 probe then completed as313996 on
`round019/c7e21b1b`, reading13selected tensors and constructing no policy.
Baseline/state time/RoPE, the recorded suffix entry, and flat/batched input
RMS admission were exact in both2.7.1and2.11. State/action/time-MLP embeddings,
normalized suffix and both raw projection arms were exact across runtimes.

Separate versus concatenated QKV differed at9BF16values in each case: Q0,
K5(maxabs.015625),V4(maxabs.00390625). These are the same9action-token locations
in baseline/state; the state-token projection did not produce these differences.
Within each runtime, both projection arms yielded identical normalized queries.
Torch2.7 reproduced the actual recorded query-norm boundary exactly, while2.11
had15900/15910FP32values different(maxabs1.43051147e-6,relativeL2about4.6e-8).
This tiny post-projection/runtime difference was not blamed on QKV fusion, nor
assumed to explain the full policy residual.

Thus step0 suffix generation and the Q side of the projection-fusion hypothesis
were narrowed by real recorded-boundary evidence. K/V projection differences,
attention layout/reduction order, later layers and other steps remain separate
questions. No new precision patch, attention backend or full-policy gate was
promoted from this local observation. The next diagnostic records actual
original eager attention inputs/output/probabilities before controlled replay,
rather than substituting another unverified reconstruction for its oracle.

That actual capture/replay completed as314150 on `round020/78208988`. The new
old-runtime producer exactly reproduced historical inputs/actions, all ten
latent/time/velocity records, query-norm boundaries, prefix KV and masks. Its
32calls/phase and10suffix phases were checked. Both original API output and
probabilities replayed exactly; each case's51consumer tensors were bitwise equal
across2.7and2.11 in independent raw checks.

Holding observed official Q/K/V fixed, both the native matching-layout arm and
contiguous-KV **original-order** arm reproduced the actual recorded context
exactly at all ten steps in both cases. Only the prefix-order intervention
(matching K/V/mask permutation, unchanged suffix and Q) introduced differences:

| Case | BF16 output differences per step after sorting | Maxabs | RelativeL2 range |
|---|---:|---:|---:|
| baseline | 2–20 | .000244140625 | 3.04072e-8–6.82992e-6 |
| state | 3–71 | .0009765625 | 5.68060e-8–2.67927e-5 |

Thus this recorded layer0 matrix isolates a prefix-order contribution, not a
contiguous-cache or runtime-version contribution under its fixed inputs. It
does **not** show that arbitrary layouts are harmless, eliminate native Q/K/V
projection/Qnorm differences, or quantify how much final action error order
explains across32layers. Those local context norms cannot be divided into the
later raw-velocity norm and called an explained-error fraction.

The next minimal diagnostic will reverse only the prefix key/value/mask order
inside explicitly eager Expert evaluations, on the SAME official latent/time
trajectory and with both native and official KV conditions. A fresh zero-weight
adapter precondition must reproduce all recorded contexts before constructing a
policy. Native own-trajectory replay remains outside the intervention. This is
conditional localization, not a new free-running policy or seven-fixture pass;
the prior6/7 failure and all production defaults/tolerances remain unchanged.

The full-Expert order diagnostic subsequently completed as314305 on
`round021/50b86d7f`. Both fresh exact proofs passed, including all20actual
original-order attention contexts and their historical sorted controls. Both
cases reproduced the complete native trace and all50sorted frozen tensors from
313766 exactly. Native raw replay remained outside the intervention; original
mask eligibility, independent scratch, read-only conditions and cleanup passed.
Each intervened case executed640eager attention calls (20raw evaluations x32
layers), not640sampler NFE or a measured GPU-kernel count.

Independent raw audit found the following raw-velocity relativeL2 ranges against
the SAME official latent/time trajectories:

| Case / prefix KV | Sorted attention order | Restored original attention order |
|---|---:|---:|
| baseline / native | .00422149–.00696753 | .00406991–.00700935 |
| baseline / official | .00329106–.00537082 | .00340313–.00548461 |
| state / native | .00399431–.00657527 | .00406906–.00669422 |
| state / official | .00313123–.00510265 | .00325715–.00515597 |

No condition became exact, and restoring order did not consistently improve
the residual. This is a negative result for order restoration as a sufficient
repair, despite the real local order effect in314150. The adapter itself does
not alter Q, but earlier layer changes may propagate into later hidden states
and queries; this is the conditional total effect through the Expert, not a
fixed-Q32layer experiment. No new free-running actions, seven-fixture gate or
robot success were tested. The previous6/7 failure remains unchanged; next
localization needs another recorded operator boundary rather than a production
order/precision patch.

The next recorded Expert0 MLP diagnostic completed as314364 on
`round022/179218ec`. The actual original producer retained exact historical
inputs/actions/trajectories/query boundaries and captured ten ordered
input/gate/up/activation/output events per case. Old API replay and every
separate intermediate matched those real BF16 records exactly. Independently
reading all outputs confirmed each case's90consumer tensors were bitwise equal
across2.7and2.11. At every recorded step, separate versus fused GU gate, up,
activation and down output were also EXACT (zero unequal values/maxabs/L2).

This is a negative result for GU fusion as the discrepancy at **Expert0 on
these20observed normalized inputs**. It does not prove all32layers, arbitrary
hidden inputs or full-policy fusion equivalence. No production splitting,
precision change or new seven-fixture pass follows. With the local MLP boundary
now measured rather than guessed, further localization should observe layer
inputs/residuals/outputs across the Expert and distinguish same-input local
operator error from propagated hidden-state differences. The historical6/7
policy failure remains unchanged.

Job314485 (`round023/fe66081a`) broadened the actual observation to all32Expert
layers, ten suffix steps and both cases. Every970-event sequence and320actual
residual-to-next-input chains per case passed; step0 entry was exact. All20native
H0 reconstructions passed runtime exact admission. More importantly, independent
raw-array checks confirmed all20native2.11 propagated velocities were bitwise
equal to313766's official-KV frozen outputs. Thus the selected-body replay is
bound to that measured native path, not merely a new plausible reconstruction.

With official prefix KV held fixed, same-input local residual/output differences
already appeared at layer0. Teacher-forced hidden inputs isolate local block
error; layer-propagated mode additionally includes earlier hidden differences.
For2.11, final-layer hidden-output relativeL2 ranges were:

| Case | Teacher-forced layer31 | Layer-propagated layer31 |
|---|---:|---:|
| baseline | .0000517155–.0004999602 | .0023364851–.0035240699 |
| state | 0–.0003188937 | .0024410601–.0035831674 |

No layer was output-exact at all ten steps in either case. These are hidden
boundary norms, not action tolerance or additive explained-error percentages.
The Gold-H32-only output-head control was exact20/20 in2.7 and19/20 in2.11:
state step9 differed at two BF16 velocity values (maxabs.001220703125,
relativeL2.0000340360). This small local head runtime effect is retained, not
silently declared exact or conflated with free-running action outliers.

Across runtimes only523/1300baseline and498/1300state replay tensors were exact;
the old consumer is the same native body on2.7, **not** an asserted historical
old-native replay. Its propagated velocities were not exact to2.11's baseline.
Both native local errors and propagated changes therefore remain measured
questions. Next intervention can vary Expert QKV projection geometry on the
same admitted boundaries, retaining the unchanged1300tensor replay as a hard
control before interpreting it. No new full-policy/robot gate was performed;
historical6/7 remainsFAILED, with all default and tolerance constraints intact.

The QKV-only intervention completed as314589 (`round024/eddf597d`). All18raw
files were retrieved without rerunning the model; a first SCP tool timeout was
resolved by bounded compressed transport, not by dropping large tensors.
Independent raw audit confirmed all1300fused tensors/case/runtime EXACT314485,
all20native-fused velocities EXACT313766, and all40fixed-Gold head negative
controls unchanged. Both runtimes completed the split arm only after the fused
controls. Every original Gold comparison and split-vs-fused effect was recomputed.

With only Expert QKV GEMM geometry changed, propagated raw-velocity relativeL2
against Gold remained:

| Runtime / case | Fused QKV | Separate Q/K/V |
|---|---:|---:|
| 2.7 / baseline | .00333524–.00546292 | .00299707–.00529732 |
| 2.7 / state | .00318833–.00562335 | .00297147–.00517791 |
| 2.11 / baseline | .00329106–.00537082 | .00327744–.00533305 |
| 2.11 / state | .00313123–.00510265 | .00315046–.00503325 |

No split step became velocity-exact and no local layer output was exact at all
ten steps. Separate projection improved L2 at8/10and9/10old-runtime steps, but
only6/10and7/10native steps; it also worsened some steps. It is therefore not a
sufficient repair or a basis for policy/default promotion. The small change in
worst residual must not be called a new seven-fixture pass. Split cross-runtime
exact tensor counts were471/1300baseline and512/1300state, not universal runtime
equivalence.

The remaining interaction can be tested by restoring prefix order **within the
already split projection condition**, first reproducing its complete sorted
baseline. That would isolate a new variable relative to this arm; the fact that
either change alone was insufficient does not establish that their combined
effect is zero. No combined-order result, sampler free-run or robot quality is
claimed here; current6/7FAILED and original thresholds remain unchanged.

Job315024 (`round025/5248e14b`) then changed only attention key order while QKV
was fixed separate. Source staging used the immutable round024 base plus the
five exact Git files, without editing that base; a persistent submission receipt
prevented duplicate submission after transport timeouts. After the user-requested
pause/resume, exact accounting confirmedCOMPLETED0:0 and all27raw files were
retrieved. Independent audit verified all1300sorted-control tensors/case/runtime
EXACT314589, all80zero-weight preflight contexts EXACTGold, unchanged40Gold-head
negative controls, code/input records and every reported contrast.

The interaction was decisive in2.7 but did not transfer to2.11:

| Runtime / case | Separate + sorted raw-velocity relativeL2 | Separate + original |
|---|---:|---:|
| 2.7 / baseline | .00299707–.00529732 | **0 (10/10EXACT)** |
| 2.7 / state | .00297147–.00517791 | **0 (10/10EXACT)** |
| 2.11 / baseline | .00327744–.00533305 | .00307602–.00494808 |
| 2.11 / state | .00315046–.00503325 | .00316721–.00527045 |

In2.7, every recorded residual AND output of all32layers, both input modes,
both cases and ten steps was bitwise equal to Gold; all20propagated raw velocities
were exact too. This is conditional Expert equivalence under Gold KV/external
trajectories, not a native-vision/full-policy/robot qualification. It also shows
why two individually insufficient interventions cannot be dismissed as having
zero joint effect.

Actual native query layouts were observed as FP32[1,16,41,128] with strides
[2048,128,2048,1] in all arms/runtimes. Equal layouts do not by themselves explain
the runtime difference, nor prove arbitrary layout changes harmless. The2.11
remaining error is not universally present at layer0: step0baseline first differs
at local layer1(residual1BF16value,maxabs.00048828125); step0state first residual
differs at layer3 and first output at layer4. Some other steps have exact residual
but differing FFN output earlier, so normalization stages need direct observation
rather than blame from the final error alone.

The next smallest diagnostic fixes captured normalization inputs and weights and
records square/variance/epsilon/rsqrt/scaling/cast/output stages, with unchanged
block replay as a precondition. PyTorch's version-matched numerical-accuracy
notes ([2.7](https://docs.pytorch.org/docs/2.7/notes/numerical_accuracy.html),
[2.11](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html)) explicitly
do not guarantee bitwise results across releases; this is a
compatibility/localization finding, not automatically a library bug. No dtype,
primitive, runtime-default or tolerance change was promoted, and historical6/7
policyFAILED remains unchanged.

Job315256 (`round026/b522e8fd`) measured four predeclared points: baseline/state
at(step0,layer1) and(step2,layer0). The selected actual block residual/output
oracles exactly reproduced315024in both runtimes. An unchanged observer captured
four real RMS calls per block; standalone eight-stage reconstruction reproduced
every same-runtime actual norm endpoint. The current old reference's complete
184role set, dtype/shape/finite/live layouts and16observed endpoints were checked
before native owner creation, not accepted from tensor count or header flags.

Independent audit read184old/328native tensors, all8block executions,32own norm
endpoints,16fixed-old native endpoints and128cross-runtime stage contrasts.
For **all16identical old inputs**, FP32 conversion and squared values were EXACT;
the first differing stage was `variance = square.mean(-1, keepdim=True)`.
Variance differences propagated into epsilon/rsqrt/scaling; this observation
does not establish whether later primitives would also differ with identical
intermediate inputs, or what fraction of the complete policy error it explains.

FP32 Q/K norm outputs differed in all8fixed-input calls. Seven of the8BF16norm
endpoints still rounded to the same values, but baseline(step2,layer0) postnorm
did not: input/residual was EXACT,17variance entries differed(maxabs1.5258789e-5),
five cast-back values andfour weighted BF16output values differed(maxabs.0078125).
This directly anchors a BF16 normalization change upstream of the measured FFN
output difference, rather than blaming GU fusion on a changed input.

One own-input postnorm already differed due to its upstream residual; the fixed
old-input arm removed that confound. All128contrast metrics were recomputed;
82stages differed and46were exact. The collector's outcome was calculated, not
hardcoded. No new RMS implementation, dtype, reduction flag, runtime default or
tolerance was installed. The next causal check can hold the old variance (then
other intermediates separately) fixed and test current downstream arithmetic;
that is still standalone localization, not a production primitive repair or a
new seven-fixture qualification. Historical6/7FAILED remains unchanged.

Job315323 (`round027/af8e74df`) tested that next causal cut, consuming the retained
315256old184/native328 arrays with no checkpoint tensor read, block/model run or
new old-runtime process. It first reproduced all144native fixed-old arrays
EXACT. On the same16norm inputs, each of five independent old-value cuts used
the CURRENT native intermediate layout; old intermediate live strides were not
recorded and were not invented. All320same-native-value suffix controls EXACT.

| Injected old intermediate | Weighted endpoints EXACT old | Remaining differing suffix stages |
|---|---:|---:|
| variance | 16/16 | 0 |
| variance+epsilon | 16/16 | 0 |
| inverse rsqrt | 16/16 | 0 |
| scaled | 16/16 | 0 |
| cast-back | 16/16 | 0 |

Accounting confirmedCOMPLETED0:0/errors0. Full784raw tensors were independently
audited:144historical baselines,320native controls and320old-cut contrasts,
injected values/layout descriptors, every reported metric and outcome count.
Thus old variance was sufficient to recover the ENTIRE measured downstream RMS
suffix in all16fixed-input cases, not merely round its final BF16 endpoint to
the same value. This excludes a residual same-intermediate downstream arithmetic
gap within these probes/layouts; it does not establish arbitrary layout safety.

No live block used that intervention yet, so neither complete Expert nor
full-policy/native-vision/free-running/robot equivalence follows. The next
smallest check is actual selected-block replay with old-variance handoff only
when each norm's actual input and weight match the captured old input/weight;
an input mismatch must be reported, never silently handed an unrelated variance.
Production math/dtype/runtime/reduction flags/defaults/tolerances stay unchanged,
and historical6/7FAILED remains. Review found noCritical/Important; one deferred
Minor test-fixture descriptor alias weakens isolation of a negative layout test,
not the production pairing check or the actual raw baseline/control gates.

Job315395 (`round028/0aeea139`) then composed the variance handoff through the
same four actual native blocks. Before owner creation/checkpoint tensor reads,
the historical315024input reports were bound to315256's retained report stats,
and mandatory manifest/Gold-trace/native-trace records bound to the exact paths
opened. A mismatch plus unexpected synchronization failure now aborts at the
probe boundary; dtype-only negative outcomes remain distinguishable in audit.
These review fixes had observed failing counterexamples before their green run.

Independent audit read all144raw tensors:56native baseline roles EXACT315256,
all16intervention norm inputs/weights/live layouts EXACTold, all16unmodified
native observations EXACTthe prior fixed-old native endpoints, all16injected
variances andreturned norms EXACTold. No input-mismatch or cleanup failure.

| Probe | Baseline residual unequal | Baseline output unequal | Handoff residual / output |
|---|---:|---:|---|
| baseline.s0.l1 | 1 | 2 | **EXACT / EXACT** |
| baseline.s2.l0 | 0 | 102 | **EXACT / EXACT** |
| state.s0.l1 | 0 | 0 | EXACT / EXACT (unchanged control) |
| state.s2.l0 | 0 | 0 | EXACT / EXACT (unchanged control) |

COMPLETED0:0/errors0. Thus the guarded variance-only intervention was sufficient
to recover these actual block residuals/outputs, not just standalone norms;
two previously different baseline probes recovered and two already-exact state
controls stayed exact. It did not paste Gold block outputs, replace GU/attention,
or apply an unrelated variance to changed inputs. Only22unique BF16weight keys
were read(44events), no full model/extension built or production norm changed.

This is still four teacher-input/GoldKV block probes, not propagated32layers,
all steps, native vision KV, free-running full policy or robot success. Do not
generalize four local causal recoveries into a whole-policy error fraction.
The next investigation checks the actual runtime versions and installed/source
reduction implementations before choosing any compatible mean prototype; no
precision/tolerance/runtime/default or historical6/7FAILED change follows here.

Version-matched source inspection after315395 used the recorded Torch Git IDs:
old `e2d141dbde55c2a4370fac5165b0561b6af4798b`, native
`70d99e998b4955e0049d13a98d77ae1b14db1f45`. The actual installed `Reduce.cuh`
headers also exhibit the same relevant changes; this is source evidence, not a
new GPU result or proof of a particular compiled branch:

- CUDA warp shuffle reduction changes ascending offsets1,2,4,8,16 to descending
  16,8,4,2,1 (ROCM/FBCODE exceptions remain). See
  [old block_x_reduce](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/cuda/Reduce.cuh#L630-L661)
  and [new block_x_reduce](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Reduce.cuh#L634-L671).
- Input vectorization changes from `dim0>128` to `dim0>=128`; at FP32 default
  `vt0==input_vec_size==4`, the boundary change matters for128-wide rows. See
  [old gate](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/cuda/Reduce.cuh#L1059-L1072)
  and [new gate](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/Reduce.cuh#L1097-L1108).
- Both versions' CUDA FP32 `mean_kernel_impl<float>` use FP32 accumulation and
  `MeanOps` addition followed by one final `a*factor`. The new8-element input
  vector specialization is for16-bit scalar inputs, NOT these FP32 squared
  inputs. There is no observed switch to BF16 accumulation or per-element mean
  scaling: [old dispatch](https://github.com/pytorch/pytorch/blob/e2d141dbde55c2a4370fac5165b0561b6af4798b/aten/src/ATen/native/cuda/ReduceMomentKernel.cu#L39-L66),
  [new dispatch](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/cuda/ReduceMomentKernel.cu#L39-L71),
  [new MeanOps](https://github.com/pytorch/pytorch/blob/70d99e998b4955e0049d13a98d77ae1b14db1f45/aten/src/ATen/native/SharedReduceOps.h#L164-L191).

These changes supply concrete FP32 addition-tree mechanisms consistent with the
local findings, not measured per-change ULP attribution. The x/y block reduction
order also changed, but source-based shape predictions do not enable y reduction
for these probes; do not attribute the measured error to that branch without
dispatch evidence. Mean directly consumes the allocated square tensor, so the
original Q/K stride cannot substitute for the square tensor's actual layout.
The next bounded observation records that layout/alignment and isolated mean's
full kernel symbols/grid/block in both runtimes. Equal launch names/dimensions
would still not prove equal runtime `ReduceConfig` vectorization or shuffle paths.
No compatibility mean, library-bug assertion or policy/default change is made.

Job315596 (`round029/a9e3acfa`) FAILED1:0/errors0 at the FIRST old-runtime trace
parser, not in mean arithmetic or CUDA launch. Its retained partial Chrome trace
contains one CPU `user_annotation` and one same-named `gpu_user_annotation`
projection. The parser incorrectly counted both as two labels; it therefore
rejected an otherwise present CPUmean/CUDAkernel observation. No final JSON/raw
array file was published and the native process did not run. This is NOT a
32trace/128array qualified result, and the failed job will not be resubmitted.

The actual partial trace was replayed on CPU: counting the unique CPU category
instead of combining CPU/GPU ranges yields one CPUmean and one full mean kernel,
grid[3,1,1]/block[32,16,1]. A reduced actual-schema regression and the independent
auditor's CPU-range predicate were observed RED then GREEN; two real CPU ranges
remain an error and GPU-only annotation cannot satisfy CPU-label admission.
The correction changes trace interpretation only, not a reduction, tolerance,
kernel-name filter or qualification threshold. New frozen source/new round is
required for a complete bounded observation; the partial trace alone supplies no
independent tensor equality proof or internalReduceConfig branch conclusion.

Job315622 (`round030/071ea364`) completed0:0/errors0 after the CPU-category-only
schema correction. All37files/12615540bytes were retrieved; an independent audit
checked all128raw arrays and32full Chrome windows. Every old/native profiled and
unprofiled mean remained EXACTits own315256variance; FP32input andsquare remained
EXACTold. CPUlabels, GPUprojections andALLkernel/launch records were retained.

Both runtimes actually launched the same full FP32 mean symbol, abbreviated
`reduce_kernel<512,1,ReduceOp<float,MeanOps<float,float,float,float>,...,4,4>>`,
with block[32,16,1]. Direct live square metadata and launches were:

| Role / direct square shape | Live stride | Grid | Windows per runtime |
|---|---|---|---:|
| input/postnorm [1,41,1024] | [41984,1024,1] | [3,1,1] | 8 |
| Q [1,16,41,128] | [2048,128,2048,1] | [41,1,1] | 4 |
| K [1,4,41,128] | [512,128,512,1] | [11,1,1] | 4 |

All32square inputs were FP32, storage_offset0, pointermod16=0; the layouts and
kernel symbols/launch dimensions matched across runtimes. This confirms the
observed direct-input/layout/launch parts of the conditional source prediction,
not equality of internal runtime vectorization or shuffle branches. It does not
undo the different mean arrays already measured; same symbols/grid/block can
contain different source-compiled instruction trees and runtimeconfig values.
No durations are used as candidate-latency evidence, and no raw trace was made
public. Historical315596remains FAILED with its originalsource/partialtrace.

The next bounded candidate may reproduce the source's explicit old FP32 addition
tree from actual inputs, then test all16means before considering any block or
policy use. It must not read stored old variance as its answer, assume generic
shapes/layouts safe, or promote default precision/runtime/tolerance changes.
Historical6/7FAILED and full-policy/native-vision/free-running/robot limits stay.

Job316521 (`round031/19e87bcc`) tested input-derived FP32 source-tree models.
Four predeclared scalar/vector4 grouping × ascending/descending shuffle arms
executed on the same16direct squares, with no stored variance passed into the
tree function and no native `sum`/`mean` fallback. Unchanged square/native mean
baselines first reproduced315622EXACT. All96raw arrays were independently
audited; all64GPU tree outputs also EXACTan independently source-indexed NumPy
FP32 implementation with per-lane/per-component accumulation and final factor.
All128old/native contrast metrics were recomputed, retaining the negative arms:

| Explicit FP32 tree | Mean arrays EXACT old | Mean arrays EXACT native2.11 |
|---|---:|---:|
| scalar grouping + ascending | 8/16 | 0/16 |
| scalar grouping + descending | 0/16 | 0/16 |
| vector4 grouping + ascending | 8/16 | 0/16 |
| vector4 grouping + descending | 0/16 | **16/16** |

The source-old width-specific choice (scalar-ascending128, vector-ascending1024)
reproduced **16/16old means** from actual input values, while vector-descending
reproduced16/16native means. This advances beyond supplying a saved old variance:
the specified arithmetic models now reproduce both observed endpoints locally.
It is still not a direct observation of internal compiled runtimeconfig, a proof
for arbitrary shapes/data/layouts, a production norm or a performance candidate.
The prototype uses multiple eager FP32 additions, no FP64 alternative/JIT/build.

COMPLETED0:0/errors0. Fullthree-file transfer and64independent CPU/GPU tree
comparisons passed. Review's deferredMinor is weak later-group/nonuniform-row
unit-fixture coverage; real nonuniform probe outputs were checked against the
independent source-indexed auditor, not accepted from producer summary flags.
Next use, if any, must compute RMS variance from its LIVEinput rather than read
Gold variance, reproduce the unchanged full native baseline first and retain
all-layer positive/negative outcomes. No full-policy/native-vision/free-running/
robot qualification, default/tolerance/runtime change or6/7FAILED rewrite.

Job316722 (`round032/d8cdfec9`) tested the functional source-old RMS across the
FULL32-layer Expert and final heads, with separateQKV/originalprefixorder held
fixed. The prototype derives variance from each LIVEinput/weight only; it does
not supply captured norm inputs, variances or outputs. The same shared replay
core runs teacher and propagated hidden modes, baseline/state, ten external
official trajectory steps, with Gold KV. Full old/native input/header/path/stat
admission occurs before ASTowner creation/checkpoint tensor reads.

Accounting confirmedCOMPLETED0:0/errors0. Fullcompressedtransfer (337071861bytes,
432558080byte tar) checked paths/file set/byte sizes/source marker and a separately
retrieved header, without a new hash. Independent audit checked all5200BF16raw
arrays and recomputed every old-oracle metric, not just worker success flags:

| Case | Unchanged control EXACT prior native | Control EXACT old2.7 | LIVE-tree EXACT old2.7 |
|---|---:|---:|---:|
| baseline | 1300/1300 | 509/1300 | **1300/1300** |
| state | 1300/1300 | 510/1300 | **1300/1300** |

| Case | Control propagated velocity relativeL2 | LIVE-tree velocity relativeL2 |
|---|---:|---:|
| baseline | .00307602–.00494808 | **0 (10/10EXACT)** |
| state | .00316721–.00527045 | **0 (10/10EXACT)** |

For EACH prototype case, all320teacher residuals,320teacher outputs,
320propagated residuals and320propagated outputs were EXACTold; all20propagated
velocities and20fixed-Gold head velocities across both cases were EXACTtoo.
The head participates in the RMS intervention, so it is not an unchanged
negative control: the control state head was9/10EXACT, the prototype10/10.
Each arm observed5160norm calls (2560FP32/128 and2600BF16/1024, including40heads),
with binding restoration/H0/order/attention cleanup verified. Only363unique
checkpoint keys/arm were loaded,726read events; no full model or new kernel build.

The login-only frozen input-loading preflight had an SSH180s timeout with no
recoverable completion and was NOT counted as passed. A read-only recovery found
no lingering process/no submission receipt; the identical full input checks
remained inside the allocation before owner/weights, and the completed control
and independent raw audit now establish that path. No repeated oldjob or gate
relaxation was used to bypass the uncertainty.

This closes the measured cross-runtime Expert gap under the JOINT fixed
projection/order/GoldKV/externaltrajectory conditions, using an input-derived
arithmetic model rather than saved Gold norm answers. It is not a fast candidate,
direct compiledbranch proof, native vision KV or free-running full policy result.
The original seven-fixture6/7FAILED remains untouched. Next investigate a strictly
Expert-phase-only native diagnostic hook and its actual prefix permutation before
testing native KV/free-running fixtures; do not change VLM/prefill, integrator,
precision, runtime, production defaults or original quality thresholds here.
An additional independent BF16-as-int16 pattern comparison, including signed
zero, found zero unequal bit patterns in all2600control/prior-native pairs and
all2600prototype/old2.7pairs. This is an array comparison, not a content hash.

Job316975 (`round033/54a661d5`) tested the triple Expert intervention inside
actual native-prefix/free-running policy requests. A single eager HWD backend
ran seven controls twice BEFORE seven candidates twice (28requests/280NFE).
Vision/prefill and native KV were not modified; separateQKV, prefix-only inverse
columns and LIVEsource-old RMS existed only during `evaluate_velocity`, including
the final norm/head. No Gold KV/latent/norm/velocity was injected and no timing
or graph qualification was performed.

The observer first exhausted SSH retries after lastPENDING(Priority); a pure
no-filesystem/no-Slurm SSH probe also timed out. This was NOT treated as a job
failure. Backoff recovery later confirmed the SAME316975COMPLETED0:0, rechecked
its persistent receipt/source and fetched all60files/739398628bytes. Existing
zstd transfer was477389305compressed/739471360tarbytes, with full file set/sizes/
source/separately-fetched-header checks and no new hash or model rerun.

Independent audit verified all1680request tensors/280NFE, actual permutation and
mask reconstruction, read-only native-prefix/conditions, exact double replays,
129RMS/32block/32attention calls perNFE, unchanged BF16-increment Euler chain,
seven historical HWD action controls EXACT and both313766legacy step controls
(47fields/case) EXACT. The other five old NPZs contain NO step trajectories;
their new control captures were labelled current-job replay evidence, not forged
historical velocity checks. Candidate native-prefix/conditions/time embeddings
were EXACTthe same-case control. Model close and all request cleanups passed.

The candidate is **REJECTED as a seven-fixture-qualified policy**: both padded
40x32 AND public40x20 remain6/7, with state failing. Original budgets stay
`abs(delta)<=.01+.01*abs(ref)` and cosine>=.999; no threshold rescues the result.

| State / padded40x32 | HWD control | Expert triple candidate |
|---|---:|---:|
| Outside tolerance | 3 | **4** |
| Maxabs | .016632080078125 | .015106201171875 |
| RMSE | .0016343457521882086 | **.0017574000621361032** |
| Cosine | .9999873074562833 | .9999853985877847 |

Allsixalready-passing cases remained passing, but RMSE improved only for
episode_start/history/noise and worsened forbaseline/current/prompt/state. The
smaller state maxabs is not a universal accuracy improvement or a policy pass.
R32conditionalExpertEXACT remains valid; native-prefix/free-run coupling does
not make component-error corrections monotonic. No production/default/runtime/
precision/sampler/gate change, speed claim or robot qualification follows.

Additional CPU comparison of the retained baseline/state native prefixes against
accepted314485Gold shows divergence BEFOREVLMprefill: all266738unequal embedding
values lie in image tokens(relativeL2.01257768617,maxabs.75), while non-image
tokens, pad and modality masks are EXACT. Native K/V already differ at layer0
and remain different at layer31. This does not prove a causal error fraction or
that VLM RMS alone is responsible: the VLM receives differing vision embeddings.
Next localize the vision/merger boundary with frozen identical inputs, rather
than retune the failed state case or broaden RMS widths without evidence.

Job317360 (`round034/c990f271`) separated frozen input origin, complete Merger
call form and runtime. No model was constructed; each process read only eight
BF16Merger weights. Official AST Tensor-forward/pooler ran per camera, while
the unchanged native HWD control kept post-PROJ1 operations batched. Two input
origins (accepted308690Gold and historical313766native) were each reconstructed
in both recorded input-layout forms, with bounded zero storage offsets explicitly
an assumption rather than an observed historical offset.

COMPLETED0:0/errors0. Allsixfiles/227763005bytes were retrieved using existing
zstd(137207959bytes;227778560byte tar), checking file set/sizes/source marker
and both separately fetched headers, with no new hash/model rerun. Independent
audit checked all120BF16arrays, all8own-API endpoint equalities, old-runtime
officialGold andnew-runtime nativehistorical anchors, R33output bridge, fixed
origin inputs/grouping/concat roles andevery reported within-runtime contrast.
It separately recomputed all56cross-runtime stage contrasts. Each corresponding
stage andactual endpoint was EXACTacross2.7/2.11 for allfourorigin/form pairs;
an additional60paired BF16-as-int16 comparison, including signedzero, foundzero
unequal bit patterns. This is not content hashing or arbitraryinput equivalence.

Within EACH runtime, form differences on identical values first appeared at
PROJ2. Input, PROJ1, grouping/mean, predictor/GELU, softmax/products/sum andthe
pre-PROJ2 activation were EXACTbetween the official/native-HWD forms:

| Fixed input origin / form contrast | First difference | Unequal merged values | Maxabs | RelativeL2 |
|---|---|---:|---:|---:|
| Gold / nativeHWD vs official | PROJ2 | 254 | .015625 | .00004932582823326906 |
| historical native / nativeHWD vs official | PROJ2 | 287 | .015625 | .00007520447640456106 |

Thus the previous PROJ1-only exact precondition was not a complete Merger
equivalence proof. The observed final Linear inputs were contiguous BF16 with
shape[1,7,7,2048] per official camera versus[3,7,7,2048] native batch; their
reported strides were[100352,14336,2048,1],offset0. Matched Torch source predicts
flattened biased addmm with M49 versusM147, but no particular cuBLAS algorithm,
epilogue or accumulation tree was observed here. PROJ2 call geometry is the next
specific hypothesis to isolate, not a proven compiled-kernel explanation.

Changing input origin while fixing form still produced input-stage differences:
476775input values,maxabs16,relativeL2.00657350586. The resulting merged origin
contrasts were266736unequal/relativeL2.01257794642 forofficial form and266740/
.01257768738 fornativeHWD,bothmaxabs.75. These FIXED-form contrasts are different
comparisons from R33's mixed native-output/Gold-output266738unequal boundary;
the counts must not be substituted or divided into an explained-policy fraction.

Research hypotheses update: upstreaminput differences remain; an additional
complete-Merger form effect is supported atPROJ2; no cross-runtime Merger gap was
observed on these two frozen origins/forms. Eight own-API anchors also support
the stage reconstruction on these calls. This does not establish otherinputs,
the cause of the upstream ViT mismatch, a per-cameraPROJ2 policy repair or a new
seven-fixture pass. Next isolate the final Linear from the admitted activation,
retaining both historical form endpoints before any policy integration. Original
6/7FAILED and productionprecision/runtime/defaults/quality thresholds stay.

Job317555 (`round035/32ea141e`) isolated PROJ2 from two admitted R34activations,
using only its two BF16checkpoint tensors. All17files/18425066bytes were fetched;
file set/sizes/source and the separately retrieved header matched. Independent
audit checked all30raw arrays,14CPUChrome traces,14profiled/unprofiled exact
replays,4historical form anchors and28reference contrasts. The seven arms were
fixed before the run; no parameter refinement or model/policy rerun followed.

Both origins gave the same grouping of observed endpoints:

| PROJ2 arm | EXACT R34official endpoint | EXACT R34nativeHWD endpoint | Actual CPU-side GEMM lowering |
|---|---:|---:|---|
| batched_bias | 0/2 | 2/2 | 1addmm, M147 |
| per_camera_b1_bias | 2/2 | 0/2 | 3addmm, M49each |
| per_camera_hwd_bias | 2/2 | 0/2 | 3addmm, M49each |
| per_camera_flat_bias | 2/2 | 0/2 | 3addmm, M49each |
| flat147_bias | 0/2 | 2/2 | 1addmm, M147 |
| batched_mm_add | 0/2 | 0/2 | 1mm, M147, followed by BF16add |
| per_camera_mm_add | 0/2 | 0/2 | 3mm, M49each, followed by BF16adds |

The computation ran on A800; only CPUaten activities/input shapes were profiled,
not CUDA kernels or durations. Parent `aten::matmul` and child `aten::mm` records
are not counted as two separate GEMMs. Thus the source-based M49/M147 prediction
is now an actual aten-lowering observation on these calls, while cuBLAS algorithm,
epilogue implementation and accumulation tree remain unobserved.

Within the biased-Linear path, row grouping reproduced the two historical
endpoints independently of B1/HWD/flat exterior rank. The explicit BF16mm-then-add
expressions did NOT reproduce either endpoint: againstofficialGold-origin output,
76773values differed(relativeL2.002756387646,maxabs.5); fornative-origin,
77608values differed(.002678609636,maxabs.5). Splitting bias is therefore a
negative arithmetic control, not an exact fix. This does not assign all those
differences exclusively to a specific fused GPU epilogue.

An additional CPU comparison of the already-saved predeclared outputs found
batched_mm_add EXACTper_camera_mm_add on both origins, including BF16bit patterns.
No new GPU trial or adaptive arm was introduced. This counterexample rules out
"M49 versus M147 must always differ regardless of expression" on this matrix;
the measured row-geometry effect is conditional on the biased execution path.
Separately,18paired BF16-as-int16 checks (14replays+4historicalanchors), including
signedzero, foundzero unequal patterns; ordinary numerical EXACTwas not assumed
to imply bit identity without this extra check. No content hash was computed.

COMPLETED0:0/errors0. The next step can restore per-camera biased PROJ2 in an
examples-only complete HWDMerger, first require actual frozen endpoints and
then test a singly changed native-policy arm. That would be a NEWqualification,
not a policy conclusion from these two saved activations. Upstreamvisioninput
differences and historical6/7FAILED remain; no runtime/precision/default/gate,
speed, robot or generalization promotion follows.

Job317665 (`round036/a5078137`) changed ONLY complete-Merger PROJ2 from one
batched biased Linear to three camera-B1 biased calls. HWDPROJ1, native Expert
QKV/order/RMS, vision/prefill, supplied noise and Euler10 stayed unchanged.
The fresh zero-model precondition checked20BF16arrays: unobserved/observed HWD
controls recovered both R34native endpoints; candidates recovered R34official
endpoints on both input origins; a separately named old-kind PROJ1child matched
captured Gold projection values. This was not a relabelled whole-policy proof.

COMPLETED0:0/errors0. Full60files/835570416bytes were retrieved with existing
zstd(546242654bytes,835645440byte tar), exact inventory/sizes/source and both
separately fetched headers checked, no new hash/model rerun. Independent audit
verified20precondition arrays+1792request arrays,28requests/280NFE,64roles each,
all replays/actual mask+permutation+visual bridges/Euler chains/callback counts,
both arms'129ORIGINALRMS/32block/32attention perNFE and close/cleanup. Actual
Merger inputs/activations/layouts and non-image/fixed conditions were EXACTacross
arms; intended image embeddings and all prefix KV were allowed to change.

All7control actions were EXACT309681, baseline/state47legacy fields each EXACT
313766. Additionally every control's60old fields were EXACTR33/316975control
(420comparisons). These are explicitly R33trajectories, not invented velocities
in the older309681NPZs. Four newly captured actual Vision fields were not passed
off as historical R33vision-input evidence.

The candidate is **REJECTED as a unified seven-fixture qualification**: padded
ANDpublic stay6/7, but the failing fixture CHANGES. State now passes while a new
noise failure appears. Equal pass counts do not mean equal failure patterns:

| Case / padded40x32 | Control outside | Candidate outside | Control RMSE | Candidate RMSE |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | .001600906810 | .001715194554 |
| episode_start | 0 | 0 | .002484001517 | .001481730841 |
| current | 0 | 0 | .001986950988 | .001720648773 |
| history | 0 | 0 | .001957458629 | .001740908140 |
| prompt | 0 | 0 | .001997666450 | .001289391827 |
| state | 3 | **0** | .001634345752 | .001717230982 |
| noise | 0 | **6** | .001724358996 | .002977203459 |

State maxabs fell .016632080078125→.0078125 even though RMSE increased; noise
maxabs rose .01318359375→.039031982421875. Candidate noise cosine remained
.9999559557532167, but could not rescue six elementwise failures. RMSE improved
for4cases and worsened for3. No input-dependent routing, tolerance/default,
precision/runtime/sampler change, speed or robot claim was introduced.

Supplementary saved-array CPU analysis confirmed that baseline/state/noise have
EXACTthe same actual Merger inputs/activations/outputs and prefix embeddings/KV
within each arm. State differs in suffix state, noise in initial latent/noise;
the same prefix perturbation therefore has context-dependent action effects.
This is not a new GPU trial or a measured policy-error fraction.

For baseline/state's ONEshared vision context, current actual Merger input was
EXACThistorical313766input, still476775values/L2.00657350586 from Gold input.
Current candidate output was EXACTR34official form on that native input. Against
Gold-origin merged output, its L2 slightly increased .01257768617059049→
.01257794642374306 although unequal values decreased266738→266736.
Let U=official(native_input)-official(Gold_input),
B=native(native_input)-official(native_input), E_control=U+B. Saved vectors
verified that identity and the normalized squared-error relation:
`(||E_control||²-||U||²)/||Gold||² = 2<U,B>/||Gold||²+||B||²/||Gold||²`.
The two terms were -1.2201970422635243e-8 and5.655137735145709e-9; net
-6.5468326874961355e-9. Thus a NETL2compensation is measured at THISMerger
boundary, not merely guessed; it is not proof of the cause/fraction of final
policy error or of R33's Expert failure. R33vision input remains uncaptured.

The next proposed experiment is one predeclared E×P interaction matrix (Expert
bundle and PROJ2 form), not case-specific repair: requalify allthreeknown cells
before the joint cell, measure actual first-velocity/action finite differences,
and do not infer joint behavior by adding old errors. A failed joint cell would
end this combination search and redirect localization upstream; even7/7would
only qualify these existing proxies, not production/generalization/robot quality.

Job317924 (`round037/e97f22e7`) executed the preregistered four-cell Expert×PROJ2
matrix, uniformly across all seven fixtures: base(E0P0), Eonly(E1P0), Ponly(E0P1),
both(E1P1). E is the existing three-change Expert bundle, not three separately
identified factors. This was post-selected mechanistic interaction validation,
not a held-out test or case-dependent routing.

COMPLETED0:0/errors0, no resubmit/cancel. All117files/1654149558bytes were fetched
(1083298926zstd/1654302720tar bytes), with exact inventory/sizes/source marker
and two separately fetched headers. Independent source-free CPU audit checked
all20fresh precondition arrays,3584request arrays and14FP64interaction arrays,
56requests/560NFE, every typed role/replay/actual mask/visual bridge/Euler chain,
factor binding, Expert/P2 counts, within-request read-only guards and cleanup.
Full three-cell historical reproduction happened BEFORE any both request:
base896raw-field comparisons to R36control, Eonly840 to R33candidate, Ponly896
to R36candidate, including both replays. Eonly's four new Vision fields were
EXACTcurrent base, explicitly NOT historical R33vision-input observations.

Padded ANDpublic gate counts were base6/Eonly6/Ponly6/**both5**. Both is REJECTED:

| Fixture / padded outside count | base | Eonly | Ponly | both |
|---|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 0 |
| episode_start | 0 | 0 | 0 | 0 |
| current | 0 | 0 | 0 | 0 |
| history | 0 | 0 | 0 | **1** |
| prompt | 0 | 0 | 0 | 0 |
| state | 3 | 4 | 0 | 0 |
| noise | 0 | 0 | 6 | **6** |

Both-history maxabs=.0107421875, RMSE=.0017224562468873739 and cosine=
.9999835400631613; both-noise maxabs=.0421142578125, RMSE=.0031483730505876707,
cosine=.9999509332927161. Both's history RMSE improved versus base/Eonly/Ponly,
yet a new elementwise failure appeared. State remained passed, but did not rescue
history/noise. No quality threshold or fixture was dropped to preserve success.

The predeclared interaction `I=(both-Ponly)-(Eonly-base)` was recomputed from
actual arrays independently in NumPy FP64, not inferred from gate counts/RMSE.
All seven first-velocity interactions were nonzero:747–797/1280elements,
maxabs=.03125, L2=.17198954905–.19135488884. All seven final-action interactions
were also nonzero:1096–1119/1280elements, maxabs=.00537109375–.0069580078125,
L2=.04744580724–.05132326981. The saved fourteen complete arrays matched the
independent formulas EXACT numerically; no unperformed signed-zero bit proof is
claimed. GPU dtypes were unchanged; FP64 was CPU analysis only.

Fixed-P E toggles preserved every Vision/prefix/KV field; all four cells shared
initial latent/time/state and fixed input/activation/non-image/masks/segments/
conditions. Thus nonadditivity is already measured in first velocity's bundle
response to different prefix/KV, rather than only inferred from free-running
feedback. Final actions include feedback as well. This identifies neither each
E subcomponent, a particular cuBLAS algorithm nor a policy-error percentage.

The hypothesis "these two uniform interventions jointly qualify all seven"
is falsified. The distinct hypothesis "their measured effects are additive"
is also falsified, but nonzero interaction did not imply quality improvement.
**The preregistered stop rule is now executed: end this combination search and
move to upstream frozen ViT localization.** Do not add case-specific routing,
new combinations or looser gates. R33/R36 failures remain in history; production
defaults/runtime/precision/sampler and original numerical budgets are unchanged.

Job318174 (`round038/311c66d0`) followed the R37stop rule with actual ViT
blocks0–3, not another policy combination. One shared baseline visual context
used three cameras/six frames and51selectedBF16weights(161436544bytes) per
process; no full VLA/backend/VLM/Expert/Merger orblock4 was constructed/executed.
The official process constructed only the original four-block vision module;
native retained the original `vit_forward` loop and stopped AFTER actualblock3.

COMPLETED0:0/errors0/no resubmit/cancel. Allsixfiles/770439476bytes were fetched
(598739766zstd/770447360tar), exact inventory/sizes/source marker and two
separately retrieved headers checked. Independent source-free CPU audit verified
allofficial124/native42arrays, three Goldblock3 FULLsix-frame anchors, Patch/
position anchors, native stop-only/observedblock3 parity, typed complete roles,
chain/residual/timePE/current-frame bridges, recorded layouts/calls and cleanup.
All41comparison boundaries and per-camera/per-frame metrics were independently
recomputed. Ordinary numerical EXACT is not an unperformed signed-zero bit test.

The earliest observed mismatch is **block0.attention AFTER output projection**.
Prepared images, Patch, position, block0input/norm1input/norm1/QKV all EXACT:

| Boundary | Unequal values | Maxabs | RMSE | RelativeL2 (native denominator) |
|---|---:|---:|---:|---:|
| block0 projected attention | 2139 | .03125 | .000043909956944 | .000023615980828 |
| block0 residual / actualnorm2input | 642 | .03125 | .000053090967356 | .000025562209115 |
| block0 norm2 | 1406 | .03125 | .000046897356430 | .000088078770820 |
| block0 MLP output | 21532 | .0625 | .000157515718338 | .000139708879685 |
| block0 output | 9889 | .0625 | .000211912479530 | .000116026314386 |
| block1 output | 438831 | .125 | .001227866431582 | .000973342106066 |
| block2 output | 1432143 | .5 | .002584514271401 | .002063878126848 |
| block3 output / allsixframes | 2089847 | .5 | .003857332748271 | .003056802672383 |

The first boundary's camera counts were483/675/981. Subsequent blocks already
receive different values; their output gaps are NOT independent primitive causes
or a per-layer policy-error attribution. The post-projection boundary still
combines attention context with output Linear. Runtime AND call form/backend
(oldper-cameraFA2 vsnewcombinedSDPA) are confounded, so2139values cannot be called
"the SDPA error" or "the Linear grouping error" from this experiment alone.
All relativeL2 comparisons here use the native endpoint norm, per the producer's
old-versus-native argument order; equality/maxabs/RMSE are symmetric. These are
local comparison metrics, not a new quality gate.

One additional fixed endpoint was observed at block3timePE: oldper-camera values
were EXACTeachother, but each differed from the nativePE at ONEentry(frame4,
feature571), -.0830078125 versus-.08251953125, maxabs.00048828125. The canonical
three-camera comparison counts3because the SAMEfixedPE is reused—not three
independent contexts. Unlike propagated hidden values, this PE is built only
from fixed frame/width/base/dtype inputs. Its precise arithmetic cause and effect
on final policy remain untested; it does not displace the earlier block0boundary.

Supplementary CPU comparison of saved arrays bridged actual current block0input,
norm1 andQKV to the corresponding retained309182 AND309409 official/native
reconstructed values: all12paired arrays EXACTdtype/shape/value. Thus the older
same-QKV frozen attention controls (including the measured559-value TorchFlash/
externalFA2 context difference) now have an actual-input VALUEbridge for this
context. This does not retrospectively observe old internal layouts or make
their saved attention contexts actual R38pre-projection captures; R38did not save
that boundary. No new GPU trial was run for this supplemental analysis.

The next proposed minimal question is to split ACTUALpre-projection attention
context from output projection on this admitted block0input, requiring own actual
API endpoints before any same-input context/call-form cuts. It must not re-run
the entire SDPAbackend search, import oldFA2 into newruntime, broaden RMS widths,
extend blocks4–26 or revive failed policy combinations. No defaults/runtime/
precision/gates, robot/performance/generalization qualification changed.

Job319502 (`round039/e058161a`) split that actual block0boundary using only
four BF16QKV/output-projection tensors(10626048bytes/process). Original old
`_ViTAttention.forward` ran three camera6 groups; original native QKV/spatial
methods ran combined18. Actual projection PREhooks/functional observers captured
NEWcontexts, rather than renaming R38 or older frozen-loop outputs.

COMPLETED0:0/errors0/no resubmit/cancel. Allsevenfiles/260161742bytes were fetched
(178880783zstd/260177920tar), source marker/inventory/sizes/two separately
fetched headers checked. Source-free independent CPU audit passed all38BF16raw
and oneFP64interaction array:old21/native17, actual R38norm1/QKV/post/API/
unobserved-control/self-projection anchors, recorded live layouts/call counts,
same-buffer context substitutions/B6views/input+weight read-only and exactN
restoration/post-restore projection. No full prefix/policy or new backend ran.

The ACTUALcontext gap is559BF16values,maxabs.00390625,RMSE5.784658939572094e-6,
relativeL2.000029474206625764763(reference=old actualcontext). Both actual old
contexts and native contexts EXACTtheir corresponding retained309182/309409
context values. Thus the earlier frozen-operator gap now has an actual API
context bridge, not only an equal-QKV bridge; it still does not create a
retroactive R38pre-projection capture or historical layout observation.

| Native-runtime projection cell | Compared with current native post | Compared with old actual post |
|---|---|---|
| N18: nativecontext,combined18 | EXACT | 2139unequal,maxabs.03125 |
| N6: nativecontext,threecamera6 | EXACT | 2139unequal,maxabs.03125 |
| O18: oldactualcontext,combined18 | 2139unequal,maxabs.03125 | **EXACT** |
| O6: oldactualcontext,threecamera6 | 2139unequal,maxabs.03125 | **EXACT** |

N18self-control andrestoration were hard gates. O6recovery was a preregistered
OUTCOME, not forced by admission. Crucially O18already recoveredoldpost: replacing
only the actual context values at the CURRENTnative projection layout/call
locally sufficed to recover this one measured endpoint. Grouping6vs18 hadzero
numerical effect on BOTHcontexts; this is not a universalGEMM/row-geometry law.
CPUFP64 `I=(O6-O18)-(N6-N18)` matched the complete saved array, all4064256elements
zero,maxabs/L2=0. This projection-level finite difference is not the different
Expert×PROJ2 policy interaction measured in R37 and does not contradict it.

This is an explicit intermediate-answer handoff for causal diagnosis. Native
attention still does not compute oldcontext fromLIVEQKV; no deployable fix,
whole-ViT equivalence or seven-fixture policy qualification was established.
559context versus2139post differences are not an explained-error percentage
or an eightfold/element-count amplification estimate.

Additional saved-array CPU analysis (NOnewGPUtrial) compared16retained context
outputs on the SAMEadmitted actualQKV:oldTorch2.7AUTO/forcedFlashSDPA group6/18
and newTorch2.11AUTO/forcedFlashSDPA allEXACTcurrent actualnativecontext; retained
externalFA2 group6/18 allEXACTcurrent actualoldcontext. Thus a TorchVERSIONchange
is NOTNECESSARYto produce this gap on this input: it already exists between
externalFA2andTorchSDPA within2.7. Neither does this prove universal runtime/
backend equivalence or a specific compiled-kernel arithmetic cause. The external
package was actually flash_attn2.7.4.post1; older309409GPU names showed Torch's
Flash_fwd_kernel_traits<96,128,64,4,...> on bothTorchversions, not every internal
branch or the standaloneFA2 kernel configuration.

The remaining question is inside context computation, not projection grouping.
Before any new execution, inspect exact-version existing APIs/source for a
weightless same-QKV row-LSE observation with its own R39context endpoint hard
gates; no backend matrix/kernel build/oldFA2import into newruntime/policysearch
is authorized by this result. No runtime/precision/sampler/gate/default, robot,
speed or generalization promotion follows.

Job319757 (`round040/cf4d9741`) observed existing row-LSE on the actual R39QKV,
with ZERO learned checkpoint AND trace-weight reads. Metadata-only original
R39/R38 namespaces and205role-allowlisted nonweight support arrays were admitted;
no old full tensor loader, prefix/policy/PE, kernel build or backend sweep ran.

COMPLETED0:0/errors0/no resubmit/cancel. Allsevenfiles/164137692bytes were fetched
(124389566zstd/164147200tar); inventory/sizes/source marker and two independently
retrieved headers matched. Independent source-free audit passed all16actual+
1CPUFP64reference arrays, all plain/capture contexts EXACT R39 at both groupings,
QKV before/after, actual axis/stride/offset/APIreturn metadata, zero-weight role
boundary, finite FP32 LSE, numeric AND int32bit comparisons, camera/frame/head
metrics, row alignment and reference formula. CUDA's dense sequence placeholders
were indeed recorded as None in positions2/3; context/LSE/debug remained strict.

Both engines' group6vs18 LSE were **bit-EXACT** across56448FP32rows each; actual
int32patterns were compared, not inferred from numeric equality or hashes.
Old-vs-native LSE differed in5811/56448rows BOTH numerically and by bit pattern,
maxabs3.814697265625e-6, RMSE2.6749114708546303e-7, L2=6.35526541478637e-5.
The per-camera unequal counts were1986/1870/1955; every head had some LSE
differences. Thus a stored-FP32-normalizer gap is observable, but not an isolated
QK/softmax/scale/exp/reduction/log/FMA cause or a library bug.

Context-any-channel differences were mapped explicitly toB/H/N rows. The R39
559BF16value differences occupy152rows; group6and18 gave the SAME cross-tab:

| LSE / context row comparison | Rows |
|---|---:|
| both different | 22 |
| LSE different, context equal | 5789 |
| LSE equal, context different | 130 |
| both equal | 50507 |
| total | 56448 |

Consequently stored LSE equality is not a detector of context equality: many
changed LSE rows keep the same BF16context, and130changed context rows have equal
stored FP32LSE. Equal summaries do not imply equal probabilities/internal row_sum,
nor do the22overlapping rows establish a causal contribution or policy-error
percentage. No scalar-ratio "repair" is justified by this table.

CPUFP64 mathematical-reference distances were identical between group6and18:

| Actual LSE vs mathematical reference | Maxabs | RMSE | L2 |
|---|---:|---:|---:|
| old externalFA2 | 8.087605710471735e-6 | 8.035376395867188e-7 | .00019091080306717477 |
| native ATenFlash | 8.087605710471735e-6 | 7.905559534719443e-7 | .00018782651180408936 |

The independent NumPy reference recomputation differed from the saved Torch
CPUFP64reference by at most3.552713678800501e-15, within the preregistered FP64
machine-order audit budget. This budget applies ONLY to the mathematical
reference, never the actual context EXACT gates. The slightly different reference
RMSEs are not a kernel correctness ranking, policy-quality proxy or new gate.

The LSE observation is complete; the primitive rounding mechanism and LIVE
native-policy remedy are not. FMA/exp2 source branches remain hypotheses, not
installed-binary proof. Next work must first assess whether existing nonintrusive
interfaces expose another genuinely discriminating boundary; do not simply
repeat the backend matrix, change dropout to obtain an attention matrix, infer
probabilities from LSE, build a kernel or revive policy combination search.

## Remaining acceptance and testing limits

- These input scopes are synthetic numerical/implementation probes. No robot
  task-success or representative evaluation suite was run in this iteration.
- The graph pilot and causal diagnostics are examples, not product capability
  declarations. Failed jobs and partial source coverage remain explicit.
- Selected CPU suites and staged-only clean exports passed; native import skips
  are not GPU proofs. Real A800 artifacts establish only the stated per-run scope.
- The full-repository CPU pytest attempt had optional/native dependency
  collection failures; it was **not** reported as all-green. Dependency-specific
  environment changes can change that count without qualifying GPU paths.
- Raw tensor/output domains and decoded controls are different. Use the exact
  recorded numerical quality scope, never an inferred task-quality guarantee.

</details>
