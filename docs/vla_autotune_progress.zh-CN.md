# 通用 VLA 平台验收进度

开发分支 `autotune/complete` 的起点为 `199376281801427abccbb307056b27e74eab1429`。本地 `origin` 指向 `flashrt-project/FlashRT`，该远端没有 `dev/vla-autotune-flashrt` 引用。包含指定基线的 fork 为 `20261023/FlashRT`。已 fetch 两个远端，并核实指定提交在 fork 开发分支历史中。

## 验收清单

| 项 | 状态 | 完成所需证据 |
|---|---|---|
| A | 已清理脚本，收口中 | 31 个旧脚本退出，历史报告正文已整合；token 快照合并及最终链接复核进行中 |
| B | CPU 已验，GPU 实验排队 | `4ec71d02` 强制快照；`c8d868cd` 修复仓库合法内部 skill 链接导致快照拒绝的问题；历史对照见下表 |
| C | 接口已提交，GPU 回归中 | 五份模型声明、特征工具、原作者适配器和 profile 已接通；五模型仍需各自实测并复算 |
| D | CLI 和两类头的 CPU 验收已通过 | `9888bf51` 包含两类主要头、三种骨架、校验和指南；生产接入证据跟踪 G |
| E | 工具路由和离线环境已验，等待算子实测 | ForgeKernel 五处 SM80 补丁和 KernelAgent venv 已落地。`dd6cad89` 按调用点路由三种 attention。AITune 的 CUDA 12.8 环境通过 `pip check` 和 CPU 导入 |
| F | 实验进行中 | 七输入 π0 作者／项目双门槛、FastWAM 空缓存、π0.5 中间量采集已提交 |
| G | 首次 GPU campaign 已提交 | 真实 LIBERO 图像、state、任务和固定噪声。首轮配置预检曾失败，`aad3312a` 已修复作者和项目前端的配置加载，真实配置 CPU 测试均通过 |
| H | 本轮已更新，结项待刷新 | 05:17 UTC 截点的已知用量未归属占 1.50%。历史窗口和旧累计基线归入数据包，保存原始摘要 |
| I | 流程代码与 CPU 验证已提交 | `4224f1c6` 固化按游选优、快照、预算和停止规则；新工具适用条件已写进路由文档 |
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
| 360054 | FastWAM 八输入 E20 → E16 → E18 → E20，加原作者 E20 | 控制器已提交，`sacct` 回读 `PENDING`。该作业使用稳态缓存；空缓存成本由 359953 测量 |
| SmolVLA 第二轮 | 使用 `aad3312a` 的 ready snapshot `f98c0ba181a2a96063ff`，schema 3，参考 → CUDA Graph → 复测 | 已通过来源与计划预检。首轮 359991 自然结束后提交，届时取得新作业号 |

**BLOCKED 项：**当前 Git 身份没有 `origin`（上游仓库）写权限：需要仓库维护者授予 `autotune/complete` 写入权限。KernelAgent／ForgeKernel 缺可用于生成的已授权 LLM 凭据；需要提供工具运行环境可使用的 provider。其他本地开发和 GPU 实验继续。A800 上 FP8/NVFP4 不可用；`dev/vla-autotune-flashrt` 的 H4 合入与默认配置变更由人决定。

## 轮次

| UTC | 工作 | 验证 | Token |
|---|---|---|---|
| 2026-11-24 开工 | 核对基线与用户规则，建立主会话及子会话工作段，读取五模型历史结果 | 基线 CPU 套件 606 passed，5 skipped。消息 API 核实 `tokens.input/output/reasoning/cache.read/cache.write` | 第一次主会话截点 16 条已完成记录，总处理 9,255,748，非缓存 1,365,828。后续窗口由 H 重算 |
| 2026-11-24 03:57 UTC 成本截点 | A 清理、B 准入、C 模型声明、D 接口及 E/G 准备 | 分段 CLI 和证据测试通过，未归属占已知处理量 0.74%；23 条 usage 缺失单列 | 本轮累计总处理 190,213,718；历史冻结总处理 1,741,698,368；中间空档按任务估算归属 154,124,813。不是三份可以重复累计的快照 |
| 2026-11-24 冻结轮 | 独立暂存树逐提交运行 CPU 套件；优先切出可实测的模型接入与双门槛版本 | `17d6bf6d` 647 passed；`4ec71d02` 670 passed；`9888bf51` 786 passed；`03d24c2d` 800 passed、6 skipped；外部安装资产 `d16ff819` 752 passed、6 skipped。vendor 核实 91 文件／5 上游 | 本轮增量待下次原生 usage 截点结账 |
| 2026-11-24 04:37 UTC 恢复核查 | 已将阶段代码推送至 `20261023/FlashRT` 的 `autotune/complete` 分支，`git ls-remote` 回读为 `d16ff819df200b3220cd68c883478a7eff10e39d` | 12 个关联作业均待分配，其中 359829 等待前置构建。历史整合逐项复查通过，29 个历史延迟点含 28 个实测和 1 个 NR | 延用 03:57 UTC 已结账数字，本段计入 K |
| 2026-11-24 05:17 UTC 进度与用量 | SmolVLA 配置分派修正，补齐 FastWAM 提交，历史归档，阶段选择器和 attention 路由 | `aad3312a` 802 passed、8 skipped，另有 2 条真实配置 CPU 测试通过。`1b1395ef` 752 passed，`4224f1c6` 785 passed，`dd6cad89` 889 passed。13 项关联 Slurm 作业等待，12 项优先级、1 项依赖 | 本轮 380,783,504 总处理，22,672,528 非缓存。未归属 5,710,916（1.50%），57 条缺失。另计旧累计 1,741,698,368 和估算归属空档 154,124,813 |
| 2026-11-24 15:07 UTC 工作树恢复 | 从已推送的 `f38e2232` 恢复 `autotune/complete`。重启作业观察，恢复阶段选择器的算子证据入口 | 恢复后的 CPU 套件 786 passed、6 skipped。调度器回读 13 个作业仍在排队。SmolVLA 修正版计划和提交脚本均保存在集群 | 本段已记账，待下次用量截点汇总 |

调度器在本轮预计 11 月 25 日 15:43 开始分配 A800，按调度器所在时区记录。收到自然终态后取回每项原始动作和诊断输出，逐一复算并更新 [整合报告](reports/2026-11-24-vla-autotune-final.zh-CN.md)。

**K 全仓收集检查**在同一 CPU 环境（PyTorch `2.11.0+cpu`、NumPy `2.5.4`、pytest `9.2.0`）分别对精确提交执行 `python -m pytest --collect-only -q tests --tb=short`。指定起点 `1993762` 收集 3,995 项，`00685ce6` 收集 3,362 项。两次报告相同的 10 组文件与异常。阶段 CPU 套件的通过数另见轮次表。

| 涉及的测试文件（`tests/` 下） | 当前环境缺少 |
|---|---|
| `test_pi05_cuda_graph_modes.py`、`test_pi05_probeflow_frontend.py` | `ml_dtypes` |
| `test_pi05_decoder_fp4_kernels.py`、`test_pi05_fp4_fusion_kernels.py`、共享算子 `test_fp4_integration_meta.py` | 编译扩展 `flash_rt.flash_rt_fp4` |
| `test_pi0_probeflow.py` | CUDA 运行库 `libcudart.so` |
| `test_recap_async_camera_map.py`（π0.5 trainer 的相机映射） | `pandas` |
| `test_chameleon_contracts.py`、`test_chameleon_thor_vqgan_backend.py` | Pillow |
| `test_chameleon_thor_fused_kernels.py` | 编译扩展 `flash_rt.flash_rt_kernels` |

工作段以用户任务消息的原生 UTC 时间 `2026-11-24T02:21:18.867Z` 为起点。历史 token 窗口保留其原有来源和计数口径。
