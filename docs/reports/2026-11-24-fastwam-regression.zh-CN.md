# FastWAM 八输入步数与原作者双门槛回归

执行者：Codex，所有本轮决策使用 CLI 记录为 `actor=agent`。
范围仅 F/C FastWAM 测量；没有修改主代码、默认配置、提交或推送。

## 实际提交与当前证据

本轮唯一作业为 **360054**，campaign 为 `f-fastwam-regression`，
instance 为 `64872805-fa37-448d-8692-087765a13bf8`。
提交前已确认 prepare 完整完成、四条 ledger 均为 `proposed` 且 job ID 为空，
队列中没有本轮作业，再由 `controller submit` 提交一次。
本报告当前等待该作业自然终态、完整取回和 ingest；新延迟、质量判定及 winner
均为 **NR**，不得用旧作业结果补齐。

提交使用仓库 renderer 生成的单个 sbatch：
`--partition=gpu_a800 --gpus=1`，脚本直接运行 Python；
不指定 CPU 数、绑核、walltime、模型子进程 timeout，不发送定时栈转储。
已运行 Bash 语法检查。后台观察器只读取同一 job 的 scheduler 状态，
没有等待截止、取消或自动重提逻辑。

冷缓存诊断作业 **359953** 是独立实验；本轮结果不替代它，也不将冷编译
崩溃解释为时间预算问题。Graph/eager parity 不属于本轮验收范围。

## 冻结来源、配置与覆盖

- FlashRT 所有测量代码从冻结提交
  `03d24c2d0899a10bbb30432656f1d2084135d0df` 的 `git archive` 导出。
  远端提交前核验 2,924 个文件 SHA-256，分配内 preflight 另设相同检查，
  以该独立目录执行；共享主树后续变化不进入本轮。
- 所有点同源、无 patch，保留真实 `Candidate.tool`。
  `cache.fastwam_text_crossattn_kv` 和 `sampler.euler_steps` 均为注册 config
  lane，因此使用普通 plan schema 2；没有通过删写 provenance 回避快照规则。
- 基线和 bookend 为 BF16、Euler20、`backend=torch`、
  `compile_action_infer=true`、`cache_text_crossattn_kv=true`、
  `use_cuda_graph=false`。两个候选仅改既有步数为 E16、E18。
  串行顺序为原作者 E20 → 项目 E20 reference → E16 → E18 → E20 bookend。
- H32、7 个完整动作通道、两路 224×224 图像横向拼接、proprio 8 维、
  已准备 T5 context。保留原有八请求 manifest 与 NPZ 字节。
  输入是 post-selection synthetic numerical sensitivity fixtures，
  不是代表性机器人 held-out 数据；没有闭环任务成功率证据。
- 每点 warmup 8、正式样本 24，覆盖八请求各三次。
  fixture 文件 IO/宿主端复制不计入正式延迟；
  prepared host observation/noise → completed CPU actions 为记录的请求边界。
  加载、prepare、编译及诊断 profile 与稳态 latency 分列。
- 普通稳态测量使用旧成功作业 357482 编译缓存的独立私有副本：
  6,718 个文件、625,494,784 字节；不改旧缓存，不宣称冷启动。
  新增编译或 prepare 成本以本轮实际日志和字段为准。

冻结源码的仓库 CPU 套件实际为 **739 passed、8 skipped**。
本机可选 Torch 测试跳过不等于 GPU 推理通过。

## 独立原作者 E20 与实际随机数证据

统一 adapter 调用原作者 `FastWAM.infer_action`，显式 E20，不安装项目 sampler
或模型实现。源码固定为
`7faa71108368fbb3b6885649f112af607427a2d4`；
原作者源码包 SHA-256 为
`43102a7c1d207f0a4a691991b6562d8b2dcfbc09b6b22b5a52ecc7c6b6f81520`。
提交前与远端作者源码逐文件核对了包内全部 `src/fastwam` 与 configs，共 69
个文件；分配内还要求作者 Python 文件集合完全一致。
加载实际模型类来源、完整 checkpoint、VAE 与 resolved config 身份由 producer
及 worker 绑定，完成后才报告具体 digest。

| 请求 | 原作者 seed |
|---|---:|
| noise-20261112 | 20261112 |
| noise-20261113 | 20261113 |
| noise-20261114 | 20261114 |
| noise-20261115 | 20261115 |
| image-minus0p1 | 20261109 |
| proprio-plus0p1 | 20261109 |
| context-minus0p0625 | 20261109 |
| mask-last-valid-off | 20261109 |

两层检查均在正式计时外：preflight 用真实 Torch CPU FP32 generator 重放每个
seed 并与 fixture noise 逐元素精确比较；统一 producer 随后观察作者
`infer_action` 内部真正的初始 `torch.randn` 输出。每个请求必须观察到且仅
观察到一次对应形状的 draw，输出未被替换，完整类型化值必须精确相等。
这八条实际 producer 证据尚待作业返回，当前不能仅根据种子映射宣布通过。

## 固定双门槛与结果

质量 scope 保持 `fastwam-loop-closeout-same-sampler`，
全部 7 个通道及 H32：**max relative L2 ≤ 0.05 且 min cosine ≥ 0.999**。
不放宽门槛、不剔除失败请求，不用平均误差掩盖最坏请求。
`quality` 对项目 E20 reference，`author_quality` 对同分配独立生成的作者 E20。
项目 reference 的内部自比较不能替代作者门槛。

| 方法 | 正式样本 | P50 / P95 ms | 内部门槛 | 作者门槛 | 实际 NFE |
|---|---:|---|---|---|---|
| 官方 `infer_action` E20 | NR | NR | 不适用 | 独立参考，待产出 | NR |
| 项目 compiled+text-KV E20 reference | NR | NR | NR | NR | NR |
| 项目 compiled+text-KV E16 | NR | NR | NR | NR | NR |
| 项目 compiled+text-KV E18 | NR | NR | NR | NR | NR |
| 项目 compiled+text-KV E20 bookend | NR | NR | NR | NR | NR |

尚未 ingest；首尾漂移、逐请求重算指标、profile 覆盖、执行覆盖和 winner 均为 NR。
本轮 author 使用 seeded RNG，项目使用 provided noise；CPU 返回字段合同也需核对。
即便质量通过，同卡同 E20 本身不足以声称严格作者加速比，应保留 controller
返回的不可比原因。

## 终态、停止与回归条件

等待同一 360054 自然终态后，取回 control portable manifest、作者完整动作、
四个 worker JSON/NPZ、日志和 scheduler 证据，再由固定 harness ingest。
ingest 校验 campaign/instance/job/GPU/内容 digest，并从动作重算双门槛；
另用私有 CPU 分析脚本独立重算逐请求 relL2、cosine、max-abs、RMSE，
核对所有 24 个完整动作与实际 NFE。

若作者生成失败或缺失，普通 batch 必须拒绝内部测量，延迟/质量继续标 NR；
若 reference/bookend 任一门槛失败、结果不齐或首尾漂移超过 10%，winner 为空。
E16/E18 任一门槛失败即 reject；通过但收益未明显超过首尾漂移只记 keep。
不因失败扩大网格、不自动重提、不接受或修改默认步数。

本轮已有证据仅支持上述固定输入、权重、源码、精度、环境及计时边界。
改变这些条件、GPU 架构或 mask/state/context 分布应重新回归。
即使八输入数值门槛通过，也不推出闭环任务质量或 Graph/eager parity 通过。

原始命令、plan/ledger、提交回执、源码 manifest、私有日志和分析脚本保存在
`devlocal/f-fast-measure/`，不发布输入、权重或完整动作。

## 2026-11-25 作业 360054 实测与数组复算

八输入、24 个样本、每点七个公共通道。回收的控制器计划、portable manifest、作者及各 worker 的动作数组通过内容摘要、作业、设备、请求与候选身份复核。下表列出由数组复算并与 worker 原始记录核对的最坏相对 L2 和最低余弦。作者沿用发布的噪声种子，项目显式输入噪声。作者 E20 中位数为 141.9 ms。控制器对两者计时边界中包含的噪声生成和 CPU 返回字段保留可比性限制。

| 项目配置 | 中位数 ms | 项目 L2 / 余弦 | 作者 L2 / 余弦 | 计时区间覆盖 | 决定 |
|---|---:|---|---|---:|---|
| compiled + KV E20 首测 | 128.1 | 0 / 1 | 0.003945 / 0.999992 | 99% | 保留，待人工 H4 |
| 同路径 E16 | 107.6 | 0.190669 / 0.981708 | 0.188056 / 0.982208 | 99% | NR，双门失败 |
| 同路径 E18 | 117.7 | 0.063478 / 0.997984 | 0.060801 / 0.998151 | 99% | NR，双门失败 |
| E20 复测 | 127.9 | 0 / 1 | 0.003945 / 0.999992 | 99% | 保留 |

四次独立诊断的实际 NFE 依次为 20、16、18、20。E20 首测诊断 wall 为 128.4 ms，其中 20 次动作去噪共约 101.2 ms（79%），已归属区间外的残差为 0.7 ms。此前作业 313779 的八输入 E13、E14、E16 和本次 E16、E18 都未获得合格降步候选。两轮降步搜索都保留各自的 E20，合格提速均为 0%。两轮间项目引擎发生过改变，原跨轮次停止条件仍需结合同一 compiled 基线下的第二轮证据。冷编译另由 364133 实测。
