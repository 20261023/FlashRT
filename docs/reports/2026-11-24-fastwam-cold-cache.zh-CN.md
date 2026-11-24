# FastWAM 冷编译缓存与栈转储排查

执行者：Codex，`actor=agent`。范围仅 FastWAM；没有修改模型源码、默认配置，
没有主分支提交或推送。

## 当前结论与证据等级

1. 原作业 **357202** 的 `-11` 仍保留为失败，计时和质量均为 **NR**。
   它的日志在启动后 120 秒开始全线程定时栈转储，并在 `  File ` 处截断。
   末尾出现 SymPy/Inductor 不等于已经定位到编译器的致错指令。
2. 同一 Python **3.11.17** 环境的纯解释器对照只改变了定时栈转储设置。
   关闭时进程正常退出，开启时进程返回 `−11`。两组都未加载模型或 GPU 运行时。
   因此定时转储是作业 `357202` 故障的一个可能原因。原作业缺少可核实该原因的原生 core。
3. 唯一独立 A800 冷缓存作业 **359953** 已提交，等待自然终态。
   **此刻尚无 GPU 冷启动通过结论；本轮 GPU 延迟、质量与冷编译耗时均为 NR。**
   不重复提交，不用历史热缓存结果填补。

旧作业 357482 的 compiled 请求内文本 K/V 缓存结果是旧缓存副本下的稳态证据，
不是干净缓存启动证据，详见
[整合报告中的原缓存比较](2026-11-24-vla-autotune-final.zh-CN.md#fastwam-history)。

## 独立冷缓存协议

- 保持 BF16、Euler20、H32、两路 224×224 图像拼接、预计算 T5，以及
  compiled video prefill / raw denoise。缓存选项只使用既有
  `cache.fastwam_text_crossattn_kv`，没有新产品实现。
- 完整复用原八输入 manifest 和 NPZ 字节，每点 warmup 8、正式样本 24。
  四个固定噪声和图像、state、context、mask 控制均保留。原 manifest 将其
  定义为 **post-selection synthetic numerical sensitivity**，不是代表性
  机器人 held-out 数据；本次不把它改称真实任务质量验证。
- 单个 `sbatch` A800 分配中直接执行 Python；不指定 CPU 数、绑核、walltime，
  不包 `srun`，不向模型子进程传 timeout，不发送栈转储信号。
- 每个 worker 都新建自己的 Inductor、Triton、CUDA、extensions、XDG、临时
  缓存目录，启动前逐项证明为空，已有目录则拒绝重用。禁用远程编译缓存。
  允许正常写回新生成缓存；不复制任何旧缓存，也不把“永远关闭所有缓存”
  混同于正常冷启动后可复用缓存的行为。
- 保留 `-X faulthandler` 的致命信号处理；删除的是定时 watchdog，不是吞掉
  原生故障。编译线程仍为此前的 4，计时/质量 worker 和模型实现不变。
  用普通文件保存编译日志，不创建读取模型线程栈的诊断线程。
- 先运行未缓存 reference／缓存点／未缓存 bookend，再用另外两个独立空缓存
  worker 验证当前 best 缓存路径。后者不是旧进程重放。
- 每方法预算 6 GPUh，仅在启动下一 attempt 前检查预算；不擅自截断在跑的
  编译。当前只有 359953 一个 GPU 作业。

## 源码身份与 CPU 验证

产品基线 `4ec71d02fa638ba0448bf3555501a2fd1b979ff9`。
严格 snapshot admission 拒绝 tracked Claude skill symlink；私有打包提交
`ac5112916a0a1c33cce0ad239f5945afc5125b1d` 只省略 21 个这类元数据链接。
整个 `flash_rt`、`csrc`、`cmake`、CMake 文件和 GEMM 构建脚本的 Git 对象
均与产品基线相同，没有弱化 admission 或修改 harness。

全部配置绑定同一 ready snapshot **`cd9864c6674e25f20f27`**、plan schema 3。
该 Torch 路径不加载 FlashRT native extension，manifest 原生状态明确为
`none`，并非伪造一个已构建 `.so`。FastWAM frontend 与全部七个模型模块的
SHA-256 均与 357482 使用的源码一致。

| 检查 | 实际结果 |
|---|---|
| 固定产品基线的 AGENTS CPU 套件 | 668 passed、7 skipped，12.47 秒 |
| snapshot/batch 与 FastWAM fixture/profile/adapter 相关子集 | 232 passed、38 skipped、10 subtests passed |
| 私有冷缓存执行协议 CPU 行为检查 | 4 passed；真实子进程验证空目录、不可重跑、退出码保留和预算准入 |
| sbatch Bash 语法 | 通过 |

跳过项涉及当前本机未安装 Torch 的可选测试，不能解释为 GPU 功能测试通过。
共享主工作树同时有其他负责人正在演进 ModelSpec 等接口；上表明确针对固定
基线，而不是宣称未冻结的共享主树整体通过。

## 回归与步数重评条件

冷协议只有在真实作业自然结束、每个 worker 真实产生编译缓存、schema 3
source identity 与 ledger 一致、八输入动作完整且有限、真实 NFE=20、
相对未缓存基线满足 relL2≤0.05 / cosine≥0.999 后，才可报告“本轮通过”。
已有失败永不改写成通过。

以下变化必须重新回归：Python/Torch/Triton/CUDA/驱动、GPU 架构、源码或
checkpoint、图像/文本形状和精度、mask/state 输入分布、编译线程/worker
策略、计时边界，或重新引入异步栈转储。无定时转储仍出现 `-11` 时，要重新
收集原生崩溃证据，不能继续沿用“只是报告器”的解释。

**本次不减少 Euler20。** 仅在冷启动可靠、当前 best 全输入质量回归通过，
且新 profile 证明 denoise 在新基线仍占主导后，才重评步数的 Amdahl 收益。
步数是近似方法，必须单独保留 E20 reference/bookend、固定质量预算，并扩大
到未参与既往选择的噪声、图像和 state；代表性真实机器人输入与独立作者
质量门槛未验证时，不得把这八个控制案例的通过上升为部署接受。

原始脚本、私有 manifest/路径、动作、日志和解释器崩溃证据只保存在
`devlocal/f-fast-cold/` 及对应私有实验目录，不随本报告发布。

参考：[CPython #116008](https://github.com/python/cpython/issues/116008)、
[#140815](https://github.com/python/cpython/issues/140815)；两者支持观察器
竞态机制，不代替本次数据。
