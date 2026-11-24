# 外部工具独立环境：E 段

本段仅做共享工具安装、CPU 检查和权限核对；**没有启动 kernel 生成、
提交 Slurm 作业、修改个人配置、提交或推送代码**。现有作业没有重提。
资产全部位于 [`examples/vla_autotune/external/`](../examples/vla_autotune/external/)；
无顶层 `tools/external/`。

## 固定来源和已验证范围

| 工具 ID | 公开上游固定 SHA | 当前证据 |
|---|---|---|
| `kernel.forgekernel` | OpenBMB/ForgeKernel `b2c8ae8510e02c28576808910a6b8827721ab7dc` | 独立 editable 环境；五项 SM80 补丁、CLI help 和补丁 CPU 行为检查通过 |
| `kernel.kernelagent` | meta-pytorch/KernelAgent `e0647170da36ef9b059ac0bd3d60103aa4ed378b` | 独立 editable venv；生成/优化 API 真实 CPU 导入通过；A100 SXM4 标签存在 |
| `compile.aitune` | ai-dynamo/aitune `1d3a4236ad8f458669d5633715b6956125f2289c` | CUDA 12.8 隔离环境安装完成。`pip check` 通过，CPU 导入版本为 `0.4.1.dev194+g1d3a4236a` |

远端运行时版本为 Python **3.11.17**、Torch **2.11.0+cu128**、
Triton **3.6.0**、NumPy **1.26.4**。ForgeKernel 和 KernelAgent 只读继承
FlashRT 基础运行时。AITune 续段用独立 wheelhouse 安装完整依赖，
`include-system-site-packages=false`。三者的源码、缓存和安装记录各自保存。
精确私有共享盘路径交接于本地 `devlocal/platform/E-interface.md`，不进入公开报告。

## 可复现操作

安装命令和离线 wheelhouse 流程见
[`external/README.md`](../examples/vla_autotune/external/README.md)。
安装器使用新目录。AITune 安装上游的 `torch211` 和
`onnxruntime-gpu-cuda12` extras，并从哈希锁文件安装完整依赖闭包。
wheelhouse 包含 **86** 个 wheel，共 **8,726,799,408 B**。

| 回执 | SHA-256 |
|---|---|
| `runtime-lock.txt` | `aa820c088f3788fcb3cfdd40ee950ef05f0f58bcb64e20258c04500a2ec39c37` |
| `wheelhouse-manifest.json` | `92d23570f2ee93498f608abac4108cb546a69532db3cbd607fe47605c39a847f` |
| AITune Git bundle（固定 commit 和 `v0.4.0` tag） | `f4435eab891018a90e268b8e3e4eb0f251d1585b18b2596a93657cf237616b81` |

ForgeKernel 五项补丁：

1. CUDA/CuTe DSL SM80 知识库空目录，明确尚无认证 recipe；
2. `CUTE_ARCH["sm80"]="sm_80"`；
3. CUDA 默认 `compute_80/sm_80`；
4. SASS 接受 HMMA/LDGSTS/LDSM，同时保留指令数、主导 kernel 和 profiler 错误检查；
5. PM 默认、显式覆盖和重试均不低于 **20000 cycles**。

补丁包 manifest SHA-256：
`ff6264b1ba75faa7da97a3d95c5460e3c3be9cebac846a01014c763f3f6bcf93`。
它只适用于专门的 SM80 安装，不修改其他硬件默认值。安装器校验固定 commit、
补丁 hash 和逐文件前后像，拒绝源码漂移、混合状态及符号链接越界。

KernelAgent 的优化 worker 参数由 `kernelagent-sm80.json` 显式提供：
`gpu_name="NVIDIA A100 SXM4 80GB"`。便利环境变量不是上游自动读取接口。
实际硬件证据仍必须写真实 A800/SM80，不能把别名当成设备测量。

## 权限与环境修复

- 未继承 `ANTHROPIC_API_KEY`、`CLAUDE_CODE_OAUTH_TOKEN`、`OPENAI_API_KEY`；
  不读取、复制或映射其他会话的私有认证。不打印密钥，不进行付费模型请求。
- ForgeKernel 的远端 `claude` CLI 和 `bwrap` 不在 PATH，且无授权凭据；
  因而**生成 BLOCKED**，不是可运行/认证成功。
- KernelAgent 的安装和 CPU 导入可用，但授权 provider 未就绪；
  因而**生成 BLOCKED**，first-valid 不代表最快。
- AITune 的 `nvtx` 和远端索引问题已通过离线安装解决。86 个输入文件在传输后核对
  SHA-256。隔离环境实际导入 Torch `2.11.0+cu128` 和 AITune，CUDA 未初始化。
  ModelOpt 选官方 CUDA12 依赖的 0.43.0；通用 TensorRT 元包的 CUDA13 默认依赖
  用明确本地版本的 CU12 元包适配，不篡改或冒称官方二进制 wheel。
- CUDA 12.8 的 `nvcc` 和 Nsight Compute 2025.1 文件存在；本段未使用它们编译、
  申请计数器权限或执行 GPU。compute node 出网、counter 权限和 sandbox 均未验证。

真正缺少外部 LLM key 的阻塞不能通过关闭 sandbox、借用凭据、修改默认模型
或放宽数值门槛绕过；公共包安装和 DNS 问题不属于该类阻塞。

## 验证与下一段

相关 CPU 命令：

```bash
uv run --no-project --python 3.12 --with numpy --with pytest python -m pytest -q \
  tests/test_external_tool_setup.py tests/test_vla_autotune_external.py
```

续段的命令另含 `tests/test_aitune_offline_setup.py`，实测
**59 passed，1 skipped**。远端 `external/cpu_smoke.py` 验证了 ForgeKernel
五项补丁的 CPU 行为及 KernelAgent 的实际 API 导入。
完整共享 CPU 套件最新结果为 **798 passed，8 skipped**（24.90 秒）。
中间一次全量运行的两个失败和最终日志保存在
`devlocal/external-tools/aitune-offline/`。

下一步由主会话另开**模型专属工作段**，明确模型、stage、算子、授权工具和预算：
真实调用点 capture → 独立生成 → 人工审查/harvest → 独立 oracle 的 operator gate
→ source snapshot patch → reference/candidate/bookend campaign。
基础设施就绪、外部工具生成、算子数值通过和端到端加速是四个不同状态。

## 许可

三个固定上游均为 Apache-2.0；原始许可、ForgeKernel 的传递 NOTICE、
来源 commit 和复制 hash 完整保留在
[`external/source-copies.json`](../examples/vla_autotune/external/source-copies.json)
及 `external/licenses/`。只公开适配资产/补丁上下文，不公开现有私有环境、
认证、权重或其他会话材料。新生成 kernel 需独立记录来源与许可。
