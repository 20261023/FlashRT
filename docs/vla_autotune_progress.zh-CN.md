# 通用 VLA 平台验收进度

开发分支 `autotune/complete` 的起点为 `199376281801427abccbb307056b27e74eab1429`。本地 `origin` 指向 `flashrt-project/FlashRT`，该远端没有 `dev/vla-autotune-flashrt` 引用。包含指定基线的 fork 为 `20261023/FlashRT`。已 fetch 两个远端，并核实指定提交在 fork 开发分支历史中。

## 验收清单

| 项 | 状态 | 完成所需证据 |
|---|---|---|
| A | 已清理脚本，收口中 | 31 个旧脚本退出，历史报告正文已整合；token 快照合并及最终链接复核进行中 |
| B | CPU 已验，GPU 实验排队 | `4ec71d02` 强制快照；`c8d868cd` 修复仓库合法内部 skill 链接导致快照拒绝的问题；历史对照见下表 |
| C | 接口已提交，GPU 回归中 | 五份模型声明、特征工具、原作者适配器和 profile 已接通；五模型仍需各自实测并复算 |
| D | CLI 和两类头的 CPU 验收已通过 | `9888bf51` 包含两类主要头、三种骨架、校验和指南；生产接入证据跟踪 G |
| E | 环境和算子实验进行中 | ForgeKernel 五处 SM80 补丁和 KernelAgent venv 已落地；注意力实验待实测 |
| F | 实验进行中 | 七输入 π0 作者／项目双门槛、FastWAM 空缓存、π0.5 中间量采集已提交 |
| G | 首次 GPU campaign 已提交 | SmolVLA 通过 `onboard check`；真实 LIBERO 图像/state/task，固定作者提交、权重和噪声。验收和后续调优未完成 |
| H | 进行中 | 六阶段账本、真实 usage、四张表、历史增量和低于 5% 的未归属占比 |
| I | 待验收 | 流程写回 skill，校验许可和 vendor 副本 |
| J | 待验收 | 模型结果、全过程成本、决策归属、干预次数与新模型验收报告 |
| K | 待验收 | 每个提交的 CPU 检查、收集问题清单、最终复核与 H4 交接 |

## 工作顺序

先核对 A 的引用与历史证据，同期准备 B 的 GPU 作业和 H 的逐段计数。C 提供模型声明接口后，D、E 和 G 共用同一接口。新 harness 冻结后，F 和 G 重新测量对应基线。J 合并历史报告，A 再删除归档，最后核对 K。

## 删除、作业与 BLOCKED

逐文件删除及保留原因见[清理清单](vla_autotune_cleanup.zh-CN.md)。旧的三个作者计时脚本统一到 `examples/vla_autotune/author_baseline.py`，DM 旧产物按原脚本 SHA 校验。当前首次 GPU 验收冻结在 `03d24c2d0899a10bbb30432656f1d2084135d0df`，harness tree `0f103bd8db67997d14d81bef882851acab176e63`。

| 作业 | 目的 | 是否计入优化结果 |
|---|---|---|
| 359805 | Hy 历史融合，schema 3，双快照原生构建及算子验收 | 待终态。原始 `315ce89` 缺 registry 接线，旧父提交的快照检查也拒绝它自己的技能链接。此作业的共同包装基底和最小接线已单列，不把它冒充两个原始提交的精确直接对照 |
| 359809、359829 | DM0.5 Sage 隔离环境构建、视觉注意力验收 | 未产生模型延迟 |
| 359812、359816 | Hy Flex、Turbo 注意力热点与算子观测 | 诊断作业，未产生模型 winner |
| 359953、359958 | FastWAM 空编译缓存与 π0.5 首个偏离层 | 诊断作业，等待结果 |
| 359987、359991 | π0 同卡原作者 E10／项目 E10/E2，SmolVLA 新接入验收 | 已提交，待 ingest 和动作复算 |
| 360006、360007、360012 | DM0.5、Hy、π0.5 当前配置回归 | 三份控制器提交回执均为 `acknowledged`，调度器在 04:37 UTC 显示三项均为 `PENDING (Priority)` |

**BLOCKED 项：**当前 Git 身份没有 `origin`（上游仓库）写权限：需要仓库维护者授予 `autotune/complete` 写入权限。KernelAgent／ForgeKernel 缺可用于生成的已授权 LLM 凭据；需要提供工具运行环境可使用的 provider。其他本地开发和 GPU 实验继续。A800 上 FP8/NVFP4 不可用；`dev/vla-autotune-flashrt` 的 H4 合入与默认配置变更由人决定。

## 轮次

| UTC | 工作 | 验证 | Token |
|---|---|---|---|
| 2026-11-24 开工 | 核对基线与用户规则，建立主会话及子会话工作段，读取五模型历史结果 | 基线 CPU 套件 606 passed，5 skipped。消息 API 核实 `tokens.input/output/reasoning/cache.read/cache.write` | 第一次主会话截点 16 条已完成记录，总处理 9,255,748，非缓存 1,365,828。后续窗口由 H 重算 |
| 2026-11-24 03:57 UTC 成本截点 | A 清理、B 准入、C 模型声明、D 接口及 E/G 准备 | 分段 CLI 和证据测试通过，未归属占已知处理量 0.74%；23 条 usage 缺失单列 | 本轮累计总处理 190,213,718；历史冻结总处理 1,741,698,368；中间空档按任务估算归属 154,124,813。不是三份可以重复累计的快照 |
| 2026-11-24 冻结轮 | 独立暂存树逐提交运行 CPU 套件；优先切出可实测的模型接入与双门槛版本 | `17d6bf6d` 647 passed；`4ec71d02` 670 passed；`9888bf51` 786 passed；`03d24c2d` 800 passed、6 skipped；外部安装资产 `d16ff819` 752 passed、6 skipped。vendor 核实 91 文件／5 上游 | 本轮增量待下次原生 usage 截点结账 |
| 2026-11-24 04:37 UTC 恢复核查 | 已将阶段代码推送至 `20261023/FlashRT` 的 `autotune/complete` 分支，`git ls-remote` 回读为 `d16ff819df200b3220cd68c883478a7eff10e39d` | 12 个关联作业均待分配，其中 359829 等待前置构建。历史整合逐项复查通过，29 个历史延迟点含 28 个实测和 1 个 NR | 延用 03:57 UTC 已结账数字，本段计入 K |

**K 当前采集检查：**在只有 NumPy、pytest 的 CPU 环境执行全仓 `pytest --collect-only -q tests` 得到 3,361 个已收集测试、10 个收集错误。π0.5 图模式和 ProbeFlow frontend 缺 `ml_dtypes`；π0.5 FP4 decoder/fusion 缺 `flash_rt.flash_rt_fp4`；π0 ProbeFlow 缺 `libcudart.so`。其他目录还缺 Pillow、pandas 或原生 CUDA 扩展。最终收口将重新执行。

工作段以用户任务消息的原生 UTC 时间 `2026-11-24T02:21:18.867Z` 为起点。历史 token 窗口保留其原有来源和计数口径。
