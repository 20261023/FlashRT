# FlashRT VLA 后端源码导读

这份文档从一次 `infer` 调用出发，带你看懂 FlashRT 统一 VLA 后端的源码：数据怎么流动，动作怎么一步步生成，时间花在哪里，以及 autotune 怎么在这之上挑配置。覆盖四个模型：π0、π0.5、Hy-Embodied-0.5-VLA（下文简称 Hy）和 FastWAM。

**先记住一句话：**一次推理分两部分。先把不随去噪变化的条件算一次，存成 KV cache（π0、π0.5、Hy 存的是图像加文本，FastWAM 只存首帧图像，文本留到每一步通过 cross-attention 读入）；然后动作从噪声开始，循环调用模型 N 次，每次算出一个"速度"，再由采样器把速度累加成动作。后端把"模型算速度"和"采样器怎么走步"分开写，autotune 在最外面比较不同配置的速度和误差。

**建议阅读顺序：**先读第一到第四节，再跑第十三节的 CPU 练习。想深入 Hy，接着读 6.3 和第十二节路线 B。第八节 C++/CUDA 放到最后看。

## 目录

1. [整体结构](#overview)
2. [配置、描述和加载](#configuration)
3. [跟踪一次完整推理](#request)
4. [输入和输出](#contracts)
5. [去噪采样：Euler 和 ProbeFlow](#sampling)
6. [四个模型](#models)
7. [三种缓存：KV cache、CUDA Graph、GEMM 执行计划](#graphs)
8. [从一个 Linear 追到 CUDA](#native)
9. [性能怎么看](#performance)
10. [质量怎么判](#quality)
11. [autotune 怎么跑一轮](#autotune)
12. [沿源码走一遍](#walkthrough)
13. [CPU 练习](#exercises)
14. [常见误解、排查和自测](#faq)

<a id="overview"></a>
## 一、整体结构

### 1.1 后端在哪一层

这里的"后端"指 `flash_rt.vla`：在同一个 Python 进程里加载模型、执行推理的接口。它不管 HTTP 服务、机器人通信和训练。

```text
调用方（机器人 / 数据集 / 测试脚本）
    │  observation（图像、state）、prompt、初始噪声
    ▼
统一接口：BackendConfig + VLABackend          flash_rt/vla/
    │  选模型和配置，统一调用方式和返回格式
    ▼
模型 frontend                                flash_rt/frontends/torch/
    │  读 checkpoint，预处理图像、state、文本
    ▼
模型 pipeline + 采样器                        flash_rt/models/<模型>/、flash_rt/sampling/
    │  算一次 KV cache，循环 N 次算速度，累加成动作
    ▼
算子                                          Torch、FlashRT 扩展、cuBLASLt
    ▼
GPU 计算 → 返回 CPU 上的动作数组和采样信息
```

autotune（`flash_rt/vla/autotune/`）在最外层：生成候选配置，逐个实测速度和误差，挑出最快的合格配置。

### 1.2 目录地图

| 层 | 目录 | 负责什么 |
|---|---|---|
| 实验层 | `flash_rt/vla/autotune/` | 候选配置、计时、误差比较、结果记录 |
| 统一接口 | `flash_rt/vla/` | 配置、模型注册、调用流程、动作还原 |
| 模型适配 | `flash_rt/frontends/torch/`、`frontends/jax/` | 读 checkpoint，预处理输入 |
| 计算组织 | `flash_rt/models/<模型>/`、`flash_rt/sampling/` | prefix、速度计算、采样、CUDA Graph、显存 buffer |
| 算子 | `flash_rt/core/`、`csrc/` | CUDA Graph 封装、GEMM、attention 等 |

第一次读先看下面几个小文件，每个都在 210 行以内：

| 文件 | 看什么 |
|---|---|
| [`vla/config.py`](../flash_rt/vla/config.py) | 配置有哪些字段，哪些组合直接报错 |
| [`vla/registry.py`](../flash_rt/vla/registry.py) | 模型名怎么对应到 frontend，加载前检查什么 |
| [`vla/backend.py`](../flash_rt/vla/backend.py) | `prepare → infer → close` 的流程 |
| [`vla/processing.py`](../flash_rt/vla/processing.py) | 模型输出的归一化动作怎么还原 |
| [`sampling/probeflow.py`](../flash_rt/sampling/probeflow.py) | ProbeFlow 选步数的逻辑，纯 NumPy，CPU 能跑 |
| [`sampling/torch_probeflow.py`](../flash_rt/sampling/torch_probeflow.py) | 采样器怎么调用模型、怎么累加 |

frontend 文件通常很大，混着权重转换、量化、CFG、批处理等多种模式。第一次读只跟着当前配置实际走的那条分支看。

### 1.3 常用名词

| 名词 | 含义 |
|---|---|
| observation | 一次观测，包括图像和机器人 state。四个模型的字段不完全一样 |
| frontend | 模型适配器，负责读权重和预处理输入，不是网页前端 |
| pipeline | 把模型的算子和显存 buffer 组织成完整的推理过程 |
| prefix | π0、π0.5、Hy 里图像 token 加文本 token 组成的输入前缀，每次推理只算一遍 |
| KV cache | 条件部分每层 self-attention 的 K 和 V，去噪的每一步都会读。FastWAM 里只有首帧图像 |
| cross-attention | 一组 token 去读另一组 token 的 attention。FastWAM 的文本和 state 就是这样进入模型的 |
| 动作变量 x | 从噪声开始、被采样器逐步更新的动作。注意和机器人 state 区分 |
| 速度 v | 模型在当前 x 和时间 t 下的输出，表示 x 该往哪个方向走 |
| 采样器 | 决定在哪些时间点调用模型、怎么把速度累加到 x 上。这里有 Euler 和 ProbeFlow 两种 |
| 模型调用次数 | 一次推理里调用模型算速度的次数，代码里的字段叫 `nfe` |
| CUDA Graph | 把一段 GPU 计算录制一次，之后整段重放，省掉 CPU 逐个下发 kernel 的开销 |

<a id="configuration"></a>
## 二、配置、描述和加载

这是三个阶段：创建配置只检查参数组合；描述只返回信息；加载才真正读权重、上 GPU。

### 2.1 配置字段

[`BackendConfig`](../flash_rt/vla/config.py) 的字段：

| 字段 | 含义 | 取值 |
|---|---|---|
| `model` | 模型 | `pi0`、`pi05`（也可以写 `pi0.5`）、`hyvla`、`fastwam` |
| `checkpoint` | 本地权重目录 | 路径 |
| `backend` | 执行实现 | 默认 `flashrt`；只有 FastWAM 还能选 `torch`（全部用上游原版算子） |
| `framework` | frontend 用哪套权重加载方式 | `torch`；π0、π0.5 还可以选 `jax` |
| `hardware` | 目标 GPU | `auto` 或 `cuda_sm80` 等；加载时会读真实 GPU 核对 |
| `precision` | 模型精度 | 一般写 `auto`：π0 是 FP16，其余三个是 BF16 |
| `sampler` | 采样器 | `euler` 或 `probeflow` |
| `num_steps` | Euler 步数 | 正整数；ProbeFlow 不用这个字段 |
| `probeflow` | ProbeFlow 参数 | `ProbeFlowConfig`，步数上限写在 `max_steps` 里 |
| `options` | 模型专有选项 | 每个模型允许哪些选项，在 `registry.py` 里列出，多写一个就报错 |

创建配置时会直接拒绝这几种组合：Euler 带 ProbeFlow 参数；ProbeFlow 用 `num_steps` 设步数；`precision` 不是 `auto`/`fp16`/`bf16`。checkpoint 在不在、GPU 对不对，要到加载时才检查。

`framework` 和 `backend` 这两个字段最容易混，下一节单独讲。

### 2.2 frontend、pipeline、framework、backend 各管什么

先说一个容易混的地方："backend"这个词在仓库里有三个意思：

1. **统一后端**：`flash_rt.vla` 和 `VLABackend`，也就是最外层的调用接口。
2. **配置字段 `backend`**：选模型计算用谁的实现。本节讲的是这个。
3. **SDPA 后端**：Flash、Efficient、Math 这些 attention kernel，见 6.3。

**一次推理里谁做什么**（以 Hy 为例）：

| 步骤 | 谁做 | Hy 的代码 |
|---|---|---|
| 统一调用方式，管理 prompt，传 noise，整理返回格式 | `VLABackend` | `vla/backend.py` |
| 读 `config.json` 和权重文件 | frontend | `hyvla_thor.py` 的 `_load_weights` |
| 图像 resize、pad、归一化；分词；state 补零；拼 prefix；算 mask 和 RoPE | frontend | `_preprocess_images`、`_assemble_prefix`、`_make_att_2d` |
| 录 CUDA Graph，把输入拷进 buffer，调用采样器 | frontend | `hyvla_probeflow.py` 的 `_graph_forward`、`_sample_latent` |
| ViT、merger、VLM、action expert 的实际计算 | pipeline | `pipeline_thor.py` |
| 最底层的矩阵乘和 attention | 算子 | `F.linear`、SDPA |

frontend 负责"把输入整理成模型能用的样子，并组织整次推理"，pipeline 负责"模型本身的计算"。这个仓库里 frontend 管得比名字暗示的多，录 graph、调用采样器都在它里面。

**`framework`：用哪套代码读权重。**π0 的 `torch` 版读 PyTorch safetensors；`jax` 版读 openpi 原版的 Orbax checkpoint，先转成一样的 FP16 Torch tensor，之后直接继承 Torch 版的全部代码，用同一个 `Pi0Pipeline`、同一套 CUDA kernel。所以 `framework="jax"` 只是换了读权重的方式，并不会用 JAX 推理。π0.5 同理。

**`backend`：模型计算用谁的实现。**

| 模型 | `framework` 可选 | `backend` 可选 | `backend="flashrt"` 实际是什么 |
|---|---|---|---|
| π0 | torch、jax | flashrt | FlashRT 原生 CUDA kernel，FP16 |
| π0.5 | torch、jax | flashrt | FlashRT 原生 CUDA kernel，BF16 |
| Hy | torch | flashrt | FlashRT 照着官方模型重写的 `pipeline_thor.py`，不调用官方代码；由 Torch 算子加 CUDA Graph 组成 |
| FastWAM | torch | flashrt、torch | `flashrt` 是上游代码，只把 Linear 换成原生 GEMM；`torch` 是上游代码原样运行 |

所以"Hy 用 Torch frontend，`backend` 仍然是 `flashrt`"的意思是：Hy 读权重和预处理的代码是用 PyTorch 写的；模型计算用的是 FlashRT 自己重写的版本，而这个版本内部恰好也由 Torch 算子组成。这里"Torch"出现了两次，含义不同：前一个指代码用什么框架写，后一个是 FastWAM 才有的 `backend="torch"`，意思是上游代码原样运行。

### 2.3 只看描述，不加载模型

```python
# CPU_DEMO: describe
import sys
from flash_rt.vla import BackendConfig, describe_backend

config = BackendConfig(
    model="hyvla",
    checkpoint="/local/checkpoint-placeholder",
    hardware="cuda_sm80",
    sampler="euler",
    num_steps=10,
    options={"use_cuda_graph": True, "numerical_profile": "semantic_bf16"},
)
description = describe_backend(config)
assert description["model"] == "hyvla"
assert description["precision"] == "bf16"
assert description["runtime_verified"] is False
assert "torch" not in sys.modules
print(description["action_domain"])
```

路径是占位符，不会去读。`describe_backend` 不读权重，连 Torch 都不 import。`runtime_verified=False` 表示这只是描述，还没真正跑过。

命令行也能看描述，默认就是描述模式：

```bash
python -m flash_rt.vla --model hyvla --checkpoint /local/checkpoint-placeholder --hardware cuda_sm80 --sampler euler --num-steps 10
```

### 2.4 `load_backend` 才真正加载

[`registry.load_backend`](../flash_rt/vla/registry.py) 的流程：

```text
检查 framework、precision、options
  → π0 / π0.5：读 checkpoint 里的动作还原配置
  → 读真实 GPU 型号，和 hardware 核对
  → 选 frontend 类和构造参数
  → 加载模型、分配显存
  → 包成 VLABackend 返回
```

各模型对应的 frontend：

| 模型 | frontend 类 | 文件 |
|---|---|---|
| π0 | `Pi0TorchFrontendRtx` / `Pi0JaxFrontendRtx` | `frontends/torch/pi0_rtx.py` 等 |
| π0.5 | `Pi05TorchFrontendRtx` / `Pi05JaxFrontendRtx` | `frontends/torch/pi05_rtx.py` 等 |
| Hy | `HyVLAFlowFrontend` | `frontends/torch/hyvla_probeflow.py` |
| FastWAM | `FastWAMTorchFrontend` | `frontends/torch/fastwam.py` |

frontend 是按需 `import` 的，所以只做描述的工具不用先装好每个模型的 Torch、tokenizer 和扩展。

另外，`flash_rt.load_model` 是项目原有的通用入口，本文讲的是 `flash_rt.vla.load_backend`，两者参数不同。

<a id="request"></a>
## 三、跟踪一次完整推理

### 3.1 调用示例

下面的代码需要 GPU 和真实权重。在集群上要放到 Slurm 分配的计算节点里跑。

```python
import numpy as np
from flash_rt.vla import BackendConfig, load_backend

config = BackendConfig(
    model="hyvla", checkpoint="/local/hyvla-checkpoint",
    hardware="cuda_sm80", sampler="euler", num_steps=10,
    options={"use_cuda_graph": True, "numerical_profile": "semantic_bf16"},
)

# 形状对应当前 Hy 的测试输入：3 个相机，每个相机 6 帧历史。
# 列表顺序要和 checkpoint 里的相机顺序一致。
observation = {
    "images": [np.zeros((6, 480, 640, 3), dtype=np.uint8) for _ in range(3)],
    "state": np.zeros(20, dtype=np.float32),
}
noise = np.zeros((40, 32), dtype=np.float32)

with load_backend(config) as backend:
    backend.prepare(observation, prompt="pick up the object")
    result = backend.infer(observation, noise=noise)
    print(result["actions"].shape)      # (40, 32)
    print(result["sampler_metadata"])   # 采样器、步数、模型调用次数等
```

全零图像只用来说明形状。测速度和误差要用真实的测试输入。

### 3.2 `prepare()` 做了什么

看 [`VLABackend.prepare`](../flash_rt/vla/backend.py)：

1. 选定 prompt，必要时调用 `set_prompt`（分词、编码文本）。
2. π0 和 π0.5 有 `calibrate_with_real_data`（JAX 版继承自 Torch 版），用这次的 observation 做校准。
3. Hy 和 FastWAM 没有校准入口，`prepare` 会直接跑一次完整推理并丢掉结果，触发 warmup 和 CUDA Graph 录制。

所以 `prepare` 会真的上 GPU 计算。之后 `infer` 遇到 prompt 变了，还会再更新一次。

### 3.3 `infer()` 做了什么

```text
VLABackend.infer
  ├─ 加实例锁，检查实例没有关闭
  ├─ _prepare：选 prompt，需要时更新
  ├─ _invoke：整理 prompt 和 noise，调用 frontend.infer
  ├─ π0 / π0.5：按 checkpoint 配置把 raw_actions 还原成 actions
  ├─ 检查或补上 sampler_metadata
  └─ 加上 backend_metadata，返回
```

`_invoke` 处理了一个历史差异：π0、π0.5、FastWAM 的 `frontend.infer` 有 `noise=` 参数，Hy 则从 observation 里读 noise。`_invoke` 会检查函数签名，有参数就用参数传，没有就放进 observation 的副本里。调用方两处都给 noise 会直接报错。这样外面始终用同一种写法调用。

### 3.4 同一个 prompt 为什么也可能要重新处理

π0.5 可以把 state 编码进 prompt 文本，所以 prompt 字符串没变、state 变了，也要重新 `set_prompt`。统一后端对 π0.5 会额外比较 state。

Hy 的 state 进入 action expert，不在 prompt 里，所以不需要这一步。

### 3.5 一次推理在 GPU 上做了什么

```text
图像 / 文本 / 机器人 state
            │
            ▼
图像预处理，视觉编码，文本 embedding
            │
            ▼
prefix 前向，写入每层 KV cache ─────────┐
                                       │ 本次推理固定不变
初始噪声 → x0                           │
            │                          │
            ├─ 算速度 v(x0, t0) ◀───────┤
            ├─ 更新得到 x1             │
            ├─ 算速度 v(x1, t1) ◀───────┤
            ├─ ...                     │
            └─ 最终动作                 │
                    │
                    ▼
            拷回 CPU，按需还原
```

一次 `infer` = 一次 prefix + N 次速度计算。N 就是模型调用次数。

这张图是 π0、π0.5、Hy 的情况。FastWAM 有一处不同：KV cache 里只有首帧图像，文本和 state 不在里面，而是每一步、每一层通过 cross-attention 重新读入，见 6.4。

<a id="contracts"></a>
## 四、输入和输出

### 4.1 四个模型的输入输出

| 模型 | 输入 | 输出 |
|---|---|---|
| π0 | 图像、连续 state、prompt、初始噪声 | `raw_actions` 是 `[chunk, 32]`；`actions` 按 checkpoint 还原，LIBERO 上是 7 维 |
| π0.5 | 图像、prompt（state 可编码进 prompt）、初始噪声 | 同 π0 |
| Hy | 3 个相机 × 6 帧 uint8 历史图像、20 维归一化 state（补零到 32 维）、`[40, 32]` 噪声 | 归一化的 `[40, 32]`，前 20 维是有效动作 |
| FastWAM | 首帧图像、预先算好的 T5 文本 embedding 和 mask、归一化 proprio（编码成 1 个 token 拼在文本后面）、`[1, H, A]` 噪声 | 归一化的 `[1, H, A]`，H、A 由 checkpoint 配置决定 |

Hy 的 40、32、20 和 π0 的 7 维都来自具体 checkpoint 的 `config.json`，换模型要重新看。

后端不替调用方处理这些事：相机顺序、state 有没有归一化、历史帧怎么取和补齐、动作的单位和含义。这些由 checkpoint 和调用方决定。

### 4.2 actions 怎么从模型输出还原

先看这几个字段：

```text
result["actions"]
result.get("raw_actions")
result["backend_metadata"]["action_domain"]
result["backend_metadata"].get("action_processor")
```

[`processing.py`](../flash_rt/vla/processing.py) 的规则：

- π0、π0.5 的 checkpoint 里如果有 `policy_postprocessor.json`，就按它声明的方式还原。比如声明了 `MEAN_STD`，就用 mean 和 std 还原，目录里同时存着 q01/q99 也不用。
- 没有这个文件时，保持原 frontend 的做法（按分位数还原）。

还原方式是模型输出约定的一部分，用错了动作数值就是错的。

```python
# CPU_DEMO: decode
import numpy as np
from flash_rt.vla.processing import ActionDecoder

decoder = ActionDecoder(
    mode="MEAN_STD", dimension=2,
    first=np.array([10., 20.], dtype=np.float32),  # mean
    second=np.array([2., 4.], dtype=np.float32),   # std
    eps=1e-8,
)
raw = np.array([[0.5, -0.5, 99.]], dtype=np.float32)
decoded = decoder(raw)
assert decoded.tolist() == [[11., 18.]]
assert raw.tolist() == [[0.5, -0.5, 99.]]
print(decoded)
```

第 3 个通道不在声明的 2 维动作里，所以没有出现在还原结果中。原数组也没有被改。

<a id="sampling"></a>
## 五、去噪采样：Euler 和 ProbeFlow

### 5.1 三个量

- **机器人 state**：模型的条件输入。
- **动作变量 x**：从噪声开始，被采样器逐步更新。
- **时间 t**：从 1 走到 0，告诉模型现在处于去噪的哪个阶段。

模型就是一个速度函数 `v(x, t)`：给定当前 x 和 t，输出 x 该往哪走。pipeline 只负责算 v，积分交给采样器。

### 5.2 Euler N 和负号

时间从 1 走到 0，Euler 更新写成：

```text
x_next = x + (t_next − t) × v(x, t)
```

共享采样器换成"进度" `s = 1 − t`（从 0 走到 1），并令 `u = −v`：

```text
x_next = x + Δs × u
```

两种写法是同一个更新。读代码时盯住一点：负号和步长都只能乘一次。

两个参考实现对负号的约定不一样：

| 函数 | 回调返回什么 |
|---|---|
| NumPy `sample_probeflow(noise, velocity, ...)` | `velocity(x, t)` 返回 v，函数内部取负 |
| Torch `sample_probeflow_torch_(state, evaluate, ...)` | `evaluate(t)` 已经返回 −v，函数不再取负 |

把一个接口的回调直接接到另一个上，就会多取或少取一次负号。Hy 的 `_velocity_body` 就是先把模型输出拷到 FP32 buffer，再取一次负，交给 Torch 采样器。

Euler N 的均匀写法：

```text
for i in 0..N-1:
    u = evaluate(time_i)
    x += u / N
```

源码里步长有时被乘进了别的地方。π0 原生 Euler 把 −1/N 直接乘进输出层的矩阵乘；接 ProbeFlow 时要改成只输出速度（系数 −1），积分交给外部采样器。所以改 N 时，要同时检查时间 embedding、输出缩放和积分是不是一起变了。

### 5.3 ProbeFlow（PF）

PF 是一种自适应步数的采样器（[arXiv 2603.17850](https://arxiv.org/html/2603.17850v3)）。思路是先花两次模型调用探路：轨迹直就少走步，弯就多走步。

1. 在起点（纯噪声）调用模型，得到 v1。
2. 沿 v1 走到 `probe_horizon`（默认 0.5），再调用一次，得到 v2。
3. 弯曲度 = 1 − cosine(v1, v2)。两次方向越一致，越接近 0。
4. 按弯曲度选步数（`select_schedule`）：
   - 弯曲度为 0，或 `max_steps` 本身就是 2：走 2 步。
   - 弯曲度 ≥ ε × 档位数：直接走满 `max_steps`。
   - 其他情况：`2 + ⌊弯曲度 / ε⌋ × step_increment`，不超过 `max_steps`。
5. 选了 2 步就直接用两次探路的结果走完。选了更多步，就从原始噪声重新走：v1 复用，v2 丢掉。

默认参数 ε = 0.008、`max_steps` = 20、`step_increment` = 2，档位数是 ⌈(20 − 2) / 2⌉ = 9：

| 弯曲度 | 步数 | 模型调用次数 |
|---:|---:|---:|
| 0 | 2 | 2 |
| 0.02 | 2 + 2 × 2 = 6 | 7 |
| ≥ 0.072 | 20 | 21 |

ε 越小，同样的弯曲度分到的步数越多，结果更准，也更慢。ε 只用来选步数，和后面质量门槛里的 5% 误差是两个参数。

### 5.4 模型调用次数为什么不总等于步数

| 情况 | 模型调用次数 |
|---|---:|
| Euler N 步 | N |
| PF 选 2 步 | 2 |
| PF 选 N 步（N > 2） | N + 1 |

多出来的那次就是被丢掉的 v2。v1 不用重算。

还有一个数：预先准备的时间点个数（`config.timesteps()`）。它是所有可能步数用到的时间点的并集，决定要预先算多少个时间 embedding、录多少张速度 graph。默认 PF 参数下是 64 个时间点，所以 Hy 用默认 PF 时会录 64 张速度 graph。这个数不是一次推理实际走的步数。

两个实现细节：

- `probe_horizon = 0.5` 时，两步分支正好就是均匀的 Euler2。
- Torch 实现会 `clone` v1。原生 pipeline 每次把速度写进同一块输出 buffer，不 clone 的话，第二次调用会把 v1 覆盖掉，弯曲度就算错了。

### 5.5 Euler 和 PF 对比

两者都是采样器，只决定在哪些时间点调用模型、怎么把速度累加成动作，模型本身完全不变。

| | Euler N | ProbeFlow（PF） |
|---|---|---|
| 步数 | 提前定好，所有输入都一样 | 每个输入自己决定 |
| 怎么决定 | 人设定 N | 先探路两次，看两次的速度方向差多少 |
| 模型调用次数 | N | 走 2 步时是 2 次，否则是步数 + 1 |
| 要调的参数 | N | ε 和 `max_steps` |
| CUDA Graph | 整条采样路线都能录 | 只能每个时间点单独录，选步数在 graph 外做 |

**为什么要看轨迹弯不弯**：Euler 每一步都是沿当前速度走一段直线。从噪声到动作的真实轨迹如果接近直线，走 2 步就够准；轨迹越弯，步子迈大了越容易偏，就要多走几步。Euler 对所有输入一视同仁，N 只能按最难的输入来定；PF 的想法是简单的输入少走、难的输入多走。

**这一轮实测**，PF 在三个模型上都没有比"挑好步数的 Euler"更快：

| 模型 | Euler | PF | 原因 |
|---|---|---|---|
| π0 | Euler2：26.1 ms | 26.7 ms | 上限只有 2 步，没有步数可省，反而多了外部积分的开销 |
| π0.5 | Euler10：41.9 ms | ε=0.008 时调用 13 次，46.8 ms；ε=0.002 时调用 21 次，60.8 ms | 误差合格了，但 PF 自动加步，比 Euler10 还慢 |
| FastWAM | Euler13：179.7 ms，误差 4.5% | 默认 ε=0.008（max10）时每个输入都只走 2 步，66.0 ms，相对同引擎 Euler10 误差 26%（job 309097）；ε=0.00006（max20）时调用 13–21 次，200.6 ms，误差 4.6%，P95 262.2 ms | 用 cosine 看，FastWAM 的轨迹显得很直，默认 ε 把每个输入都判成"直"，误差很大；ε 调小到能合格后，有的输入要走满 21 次，中位数和尾延迟都比 Euler13 差 |

FastWAM 这一行里，Euler13 和 ε=0.00006 是同一批 4 个输入、以 Euler20 为参考的对比（job 309386）；ε=0.008 那组来自更早的 job 309097（6 个请求，max10，以 Euler10 为参考）。后来扩大到 12 个输入（job 313779），Euler13 不再合格，FastWAM 退回 Euler20；PF 没有在扩大后的输入上测过。

PF 要发挥作用，前提是不同输入的难度差别很大。在这些固定输入实验中，合适的固定 N 曾比 PF 更有利，但扩大输入范围也推翻过减步候选；不能外推未测任务。数据来自[整合实验记录](reports/2026-11-24-vla-autotune-final.zh-CN.md)。

### 5.6 FastWAM 的时间映射

[`ShiftedSigmaSchedule`](../flash_rt/sampling/schedules.py)：

```text
sigma(u) = shift × u / (1 + (shift − 1) × u)
模型输入的时间 = 1000 × sigma
动作更新 = (sigma_next − sigma) × v
```

模型输入的时间乘了 1000，积分步长却是 sigma 的差，不乘 1000。PF 用在 FastWAM 上时，用逻辑时间 u 当查表的 key，按真实 sigma 的差积分（`sample_probeflow_torch_` 的 `time_map` 参数）。

<a id="models"></a>
## 六、四个模型

### 6.1 π0

入口 [`frontends/torch/pi0_rtx.py`](../flash_rt/frontends/torch/pi0_rtx.py)，计算在 [`models/pi0/pipeline_rtx.py`](../flash_rt/models/pi0/pipeline_rtx.py)，PF 路径在 [`pipeline_rtx_probeflow.py`](../flash_rt/models/pi0/pipeline_rtx_probeflow.py)。

- decoder 的输入是 `[state token; action tokens]`，state token 有自己的 mask。速度和积分只针对 action 部分。
- 精度是 FP16。原生 pipeline 自己管理 CUDA buffer，Torch 采样器用 `native_float_view` 按 FP16 直接读同一块显存，不拷贝。
- PF 路径复用 prefix，单独录每个时间点的 decoder graph，只输出速度，不在 decoder 里积分。

### 6.2 π0.5

入口 [`frontends/torch/pi05_rtx.py`](../flash_rt/frontends/torch/pi05_rtx.py)，计算在 [`models/pi05/pipeline_rtx.py`](../flash_rt/models/pi05/pipeline_rtx.py)，PF 路径在 [`pipeline_rtx_probeflow.py`](../flash_rt/models/pi05/pipeline_rtx_probeflow.py)。

和 π0 的区别：

- state 可以编码进 prompt，所以 prompt 缓存要看 state 有没有变。
- action expert 用 AdaRMS 把时间信息注入每一层，结构和 π0 不同。
- 统一接口固定走 BF16 非量化路径。registry 会检查 FP8/INT8 相关的环境变量，冲突就报错。
- PF 路径不支持 RTC、跨帧复用 KV cache、导出原生 runtime，调用这些会直接报错。
- prefix 有 911 个 token（768 个视觉 + 143 个文本和 state），是耗时的大头，见 9.2。

### 6.3 Hy

入口 [`frontends/torch/hyvla_probeflow.py`](../flash_rt/frontends/torch/hyvla_probeflow.py)，输入处理和权重加载继承自 [`hyvla_thor.py`](../flash_rt/frontends/torch/hyvla_thor.py)，BF16 计算在 [`models/hyvla/pipeline_thor.py`](../flash_rt/models/hyvla/pipeline_thor.py)。

**模型结构和数据形状**（按当前测试输入）：

```text
3 个相机 × 6 帧历史图像，每帧 480×640 uint8
  │ resize + pad 到 224×224，归一化到 [-1, 1]
  ▼ [3, 6, 3, 224, 224] BF16
① ViT（27 层，宽 1152）
  │ 每帧切成 14×14 = 196 个 patch
  │ 第 3、7、11、15、19、23 层是时空层：同一个 patch 位置沿 6 帧做因果 attention
  │ 第 23 层之后只保留当前帧，后面几层的计算量降到 1/6
  ▼ [3, 196, 1152]
② merger
  │ PROJ1（1152 → 2048）→ 2×2 加权池化（196 → 49）→ GELU → PROJ2
  ▼ [3, 49, 2048]，共 147 个视觉 token
③ 拼 prefix
  │ 2 个起始 token
  │ + 每个相机 58 个：vision_start + 7 行 ×（7 个视觉 token + 1 个换行 token）+ vision_end
  │ + 64 个文本 token
  │ = 2 + 3 × 58 + 64 = 240 个 token，再按 [视觉 | 文本] 重新排序
  ▼
④ VLM（32 层，宽 2048，16 个 Q head、4 个 KV head）
  │ 视觉 token 和文本 token 各用一套权重，attention 一起算（MoT）
  │ 只跑一次，产出每层的 KV cache
  ▼
⑤ action expert（32 层，宽 1024），每次调用模型：
  │ 输入：1 个 state token + 40 个动作 token（x 和时间 embedding 融合）
  │ 每层先读 VLM 同一层的 KV cache，再和自己的 token 一起做 attention
  ▼ 速度 v：[40, 32]
采样器累加，Euler10 走 10 步 → 动作 [40, 32]，前 20 维有效
```

几处细节：

- **2×2 加权池化**：把相邻 4 个 patch 合成 1 个。不是简单平均，而是用一个小网络给 4 个 patch 打分，再加权求和。
- **PROJ1** 就是 merger 的第一个 Linear。之前对齐官方实现时，这里因为 Torch 选了不同的矩阵乘 kernel（`bmm` 和 `mm`）而出现过差异。
- **147 个视觉 token 不是 prefix 长度**。prefix 还有起止 token、换行 token 和文本，共 240 个。
- **KV cache 预先扩展成 16 个 head 存放**。mask 和 GQA 同时开启时，Torch 会被迫走慢速的 Math 后端；代码注释记录过这在 Thor（SM110）上端到端约占 41 ms。预先扩展可以绕开。

**代码怎么拆**：`HyVLAFlowFrontend` 里按顺序读这三个函数。

1. `_prefix_body`：跑 VLM，写 KV cache，并把 state 投影成 state token。
2. `_velocity_body`：调用 `pipe.evaluate_velocity` 算一次速度，转成 FP32 后取负。
3. `_sample_latent`：Euler 或 PF 负责更新 x。

**CUDA Graph 分三类**：

| graph | 数量 | 缓存的 key |
|---|---|---|
| ViT + merger | 1 张 | （相机数，帧数） |
| prefix（VLM + state 投影） | 1 张 | （prefix 长度，视觉 token 数，时间表） |
| 速度 | 每个时间点 1 张，Euler10 是 10 张 | 同上 |

积分和 PF 选步数都在 graph 外面做。每次请求先把 prefix embedding、mask、RoPE、state、noise 全部拷进固定 buffer，再重放。prefix 长度变了会重新录一套。

**两种数值配置**（`numerical_profile`）：

| | `semantic_bf16`（默认） | `official_local_fp32` |
|---|---|---|
| 权重和矩阵乘 | BF16 | BF16 |
| RoPE 表、旋转后的 Q/K | BF16 | FP32 |
| VLM 和 action expert 的 attention | Torch SDPA（Efficient 或 Math） | 按官方写法：FP32 算 QK 和 softmax，概率转 BF16 再乘 V |
| 时间 embedding | FP64 计算后存成 BF16 | 同样 FP64 计算、存成 BF16，但时间先转成 FP32，和官方一致 |
| Euler 时间表 | 每步用公式直接算 | FP32 标量每步累减 |
| Euler 增量 dt × v | FP32 | BF16 |
| ViT attention | Torch SDPA | 不变 |

`official_local_fp32` 只对齐了上表里这几处局部计算，日常优化用默认的 `semantic_bf16`。

这条 frontend 只允许纯 BF16：FP8、FP4、fused 等开关必须是 `False`，传 `True` 直接报错。也就是说，它没有用到仓库里为 Thor、Orin 做过的量化 kernel。

**Torch SDPA 的四种后端**：数学公式相同，都是 `softmax(QKᵀ/√d)·V`，区别在分块和累加顺序，所以 BF16 下末几位结果不同。

| 后端 | 做法 | Hy 里的用法 |
|---|---|---|
| Flash | 分块计算，不生成完整注意力矩阵，最快；不支持任意 mask | ViT 的空间 attention（没有 mask，AUTO 通常选中它） |
| Efficient | 分块计算，支持任意 mask | VLM 和 action expert 的带 mask attention |
| cuDNN | NVIDIA 库里的实现 | 没有用 |
| Math | 先算完整 QKᵀ，再 softmax，再乘 V；最慢、最占显存 | Efficient 用不了时的后备 |

### 6.4 FastWAM

入口 [`frontends/torch/fastwam.py`](../flash_rt/frontends/torch/fastwam.py)，计算在 [`models/fastwam/pipeline.py`](../flash_rt/models/fastwam/pipeline.py)。

```text
FastWAMActionPipeline.infer
  → _initial_state：初始动作噪声
  → _prepare_condition
      ├─ proprio 编码成 1 个 token，拼到文本 context 末尾
      ├─ 首帧图像经 VAE 编码
      ├─ video_expert.prepare
      └─ mot.prefill_video_cache_tensor：video KV cache，只算一次
  → Euler 或 PF
      └─ _denoise_action_with_video_cache：每次只算速度
  → 归一化动作 [1, H, A]，拷回 CPU
```

**条件怎么进入模型**：FastWAM 的主干是 Wan 风格的 DiT，分 video expert 和 action expert 两部分。它和 π、Hy 最大的区别是，文本不和图像 token 一起做 self-attention，而是在每一层通过 cross-attention 读入。

```text
文本 context（T5 embedding + 1 个 proprio token）
   │ cross-attention（每层都读）               │ cross-attention（每一步、每层都读）
   ▼                                          ▼
首帧图像 → VAE → video expert 各层 ──► 每层 self-attention 的 K、V（video KV cache，算一次）
                                              │
动作 x + 时间 t → action expert 各层 ──────────┘ self-attention 时把自己的 K、V 拼在 video cache 后面
                                              ▼
                                           速度 v
```

- **图像，算一次**：`prefill_video_cache_tensor` 把首帧的 video token 跑完 video expert 的每一层，存下每层 self-attention 的 K、V。之后 action expert 每层做 self-attention 时，把自己的 K、V 拼在 video cache 后面。这部分和 π、Hy 的 KV cache 是同一个思路。
- **文本和 state，每步都算**：每次调用 `_denoise_action_with_video_cache`，`action_expert.prepare` 都会重新跑一遍 `text_embedding(context)`（两个 Linear 加 GELU），每一层的 cross-attention 也都重新从 context 算 K、V。
- **T5 一次都不跑**：文本 embedding 由调用方提前算好传进来，加载时不加载 T5。

上游代码（[FastWAM](https://github.com/yuantianyuan01/FastWAM) commit `7faa711`）里对应的位置：

| 做什么 | 位置 |
|---|---|
| proprio 拼到文本后面 | [`fastwam.py:224` `_append_proprio_to_context`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L224) |
| 一次速度计算 | [`fastwam.py:703` `_denoise_action_with_video_cache`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/fastwam.py#L703) |
| 写 video KV cache | [`mot.py:474` `prefill_video_cache_tensor`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L474) |
| action 的 K、V 拼在 video cache 后面 | [`mot.py:560`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/mot.py#L560) |
| 每步重跑 `text_embedding` | [`action_dit.py:324`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/action_dit.py#L324) |
| cross-attention 每次从 context 算 K、V | [`wan_video_dit.py:214`](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/src/fastwam/models/wan22/wan_video_dit.py#L214) |

**一个可以做的优化**：同一次推理里，文本 context 和 proprio 不随去噪步变化，所以 `text_embedding` 的输出和每层 cross-attention 的 K、V 其实只需要算一次。现在每一步都在重算，Euler20 就是重算 20 遍。这一处缓存只是复用相同的计算，输出可以和原来逐位一致。能省多少取决于文本 context 的长度和 action expert 的宽度，还没有实测。

其他要点：

- 每次速度计算调用上游的 `_denoise_action_with_video_cache`，只算速度不积分。不会拿上游完整的 `infer_action` 冒充一次速度计算。
- 加载需要上游源码、模型配置、完整训练 checkpoint 和本地 VAE；`options` 里必须给出 `sigma_shift`、`action_horizon`、`model_config`、`backbone_dir`。
- `backend="torch"`：全部用上游 Torch 算子。`backend="flashrt"`：把 Linear 换成原生 GEMM，VAE、attention、norm、激活仍是上游 Torch。
- 原生 Linear 有两种：`legacy` 每次调用都重新建 cuBLASLt 执行计划，`cached` 每种形状只建一次。FlashRT 原生配 CUDA Graph 时必须用 `cached`。

<a id="graphs"></a>
## 七、三种缓存：KV cache、CUDA Graph、GEMM 执行计划

| 缓存 | 存的是什么 | 每次推理还要做什么 |
|---|---|---|
| KV cache | 条件部分每层 self-attention 的 K、V | 新的图像、prompt、state 来了要重新算 |
| CUDA Graph | GPU 执行顺序和显存地址 | 把新输入写进固定 buffer，GPU 照样全部算一遍 |
| GEMM 执行计划 | 矩阵乘用哪个 kernel、怎么配置 | 每次照样做矩阵乘 |

### 7.1 KV cache

同一次推理里图像和文本不变，所以条件部分的 K、V 算一次，N 次速度计算都读它。下一次推理换了图像或 prompt，就要重新算。

四个模型的条件分别怎么进入模型：

| 模型 | 图像 | 文本 | 机器人 state |
|---|---|---|---|
| π0 | 进 prefix，KV cache 算一次 | 进 prefix，KV cache 算一次 | action expert 里的 state token |
| π0.5 | 进 prefix，KV cache 算一次 | 进 prefix，KV cache 算一次 | 可以编进 prompt 文本 |
| Hy | 进 prefix，KV cache 算一次 | 进 prefix，KV cache 算一次 | action expert 里的 state token，投影只做一次 |
| FastWAM | 首帧进 video expert，KV cache 算一次 | 每一步、每层通过 cross-attention 重新读 | 编码成 1 个 token 拼在文本后面，跟文本一样每步读 |

### 7.2 CUDA Graph

```text
预先分配 buffer → warmup → 录制 → 每次推理写入新输入 → 重放
```

省的是 CPU 逐个下发 kernel 的时间，GPU 上的 GEMM、attention 照样执行。

π0、π0.5 用 [`core/cuda_graph.py`](../flash_rt/core/cuda_graph.py) 封装的 CUDA Runtime graph；Hy 和 FastWAM 用 Torch 的 `torch.cuda.CUDAGraph`，各自组织。

每次推理都要把新的图像条件、prompt、state、noise、mask 写进 buffer。不能因为两次的 tensor 地址一样，就认为内容没变——调用方可能原地改了同一个 tensor。

FastWAM 的速度 graph 要求输入的 shape、stride、dtype、设备固定，不一致直接报错，不会自动重录或退回 eager。

### 7.3 PF 为什么不能整条录成一张 graph

PF 要先探路，再根据这次输入的弯曲度决定走几步。如果把整条采样路线录下来，之后每次重放的都是录制时碰巧选中的那条路线，就不是自适应了。

所以拆成三块：prefix 一张 graph；每个时间点一张速度 graph（FastWAM 是一张可以改时间输入的速度 graph）；选步数和积分在 graph 外做。选步数要把 cosine 读回 CPU，这一小段等待也算进延迟。

### 7.4 GEMM 执行计划缓存

cached GEMM runner 第一次遇到某种矩阵形状时，建一份 cuBLASLt 执行计划（矩阵描述、kernel 选择）和 workspace，之后同形状直接复用。省的是 CPU 准备时间，矩阵乘本身每次都要做。

### 7.5 一个实例不能并发使用

一个 backend 实例持有 x、KV cache、graph buffer 和 GEMM workspace 这些可变状态。`VLABackend` 用锁让调用排队执行。FastWAM 第一次 prepare、录 graph 时，进程里不要有别的线程同时往 GPU 提交任务。

### 7.6 `close()` 做了什么

看 [`backend.py`](../flash_rt/vla/backend.py) 和 [`resources.py`](../flash_rt/vla/resources.py)：

| 模型 | `close()` 做什么 |
|---|---|
| π0 / π0.5 | 同步 graph stream，释放 prompt 缓存和旧 pipeline，销毁原生 graph |
| FastWAM | 等执行 stream 跑完，关闭速度 graph，释放引用 |
| Hy | 没有单独的 `close()`，统一层释放 frontend 引用后，graph 和显存由 Torch 回收 |

关闭后显存数字不一定马上下降，allocator 和 CUDA context 还会占一部分。后端不会去清进程的全局缓存。

<a id="native"></a>
## 八、从一个 Linear 追到 CUDA

FastWAM 的 [`native_linear.py`](../flash_rt/models/fastwam/native_linear.py) 最适合看原生算子是怎么接进来的。

### 8.1 先写清形状

```text
输入 A：[M, K]
权重 W：[N, K]
bias： [N]
输出 D：[M, N]

D = A @ W.T + bias
```

输入原本是 `[B, S, K]` 时，前两维展平成 `M = B × S`，算完再恢复。权重本来就按 `[N, K]` 存，不需要再转置复制一份。

FLOPs 按 `2 × M × K × N` 估算：每个输出元素做 K 次乘加，一次乘加算 2 次浮点运算。

### 8.2 调用路径

```text
上游的 nn.Linear
  → 加载时替换成 FlashRTLinear
  → forward：输入转 BF16、展平，准备 weight 和 bias
  → runner.bf16_nt_bias(指针, M, N, K, stream)
  → PyBind 的 C++ 入口
  → cuBLASLt matmul
```

C++ 部分看 [`csrc/bindings.cpp`](../csrc/bindings.cpp)、[`gemm_runner.cu`](../csrc/gemm/gemm_runner.cu)、[`bf16_nt_plan.cpp`](../csrc/gemm/bf16_nt_plan.cpp)。

两个要注意的地方：

1. 传给 C++ 的是裸指针，Torch 不知道这些 tensor 被哪个 stream 用着，所以 `forward` 里对每个 tensor 调了 `record_stream`，防止显存被提前回收。
2. cached runner 有自己的设备、workspace 和对齐要求，不能随便传一个带偏移的 view 进去。

### 8.3 调用计数有好几种

Python 里 `FlashRTLinear.forward` 被调用的次数、CPU 成功下发 GEMM 的次数、graph 里录下的 Linear 数量、GPU 重放的次数，是四个不同的数。

录完 graph 之后，GPU 重放时会照样执行矩阵乘，但不再经过 Python 的 `forward`。所以"Python 计数是 0"不代表 GPU 没做 Linear。看 FastWAM 的 `execution_metadata.velocity_graph` 时，要看重放次数和录制的工作量。

<a id="performance"></a>
## 九、性能怎么看

### 9.1 一次推理的时间拆分

```text
总耗时 ≈ 输入预处理 + prefix
       + 模型调用次数 × 每次速度计算
       + 选步数和积分 + 拷回 CPU
```

这是帮助思考的近似，不是严格相加的 kernel 计时。用它来问：这项优化到底减少了哪一块？

### 9.2 少一半步数，不一定快一倍

π0.5 的实测：prefix 约 24 ms，每次 action expert 约 1.65 ms。

```text
Euler10：实测 41.6 ms
Euler8 ：实测 38.4 ms，只快约 8%
```

prefix 占了大头，减少步数省下的只是后面那一小块。

### 9.3 CPU 下发 kernel 也是成本

FastWAM 在 Euler2 下，插桩（cProfile/Kineto）的请求里 GPU kernel 时长合计约 52 ms；不插桩的请求中位数是 122.5 ms（Torch）和 154.8 ms（cached）。差值很可能主要来自 CPU 逐个下发 kernel，但这只是机制推断，实验记录里没有确认归因。后面上 CUDA Graph 的结果支持这个推断。

所以 FastWAM 上的优化顺序是：

| 做法 | 结果 |
|---|---|
| 原生 Linear 从 `legacy` 换成 `cached` 执行计划（Euler2） | 202.5 → 154.8 ms，快约 24%，但仍比 Torch 的 122.5 ms 慢 |
| 上 CUDA Graph（Torch，Euler20） | 804.7 → 255.1 ms，快约 3.2 倍，输出完全一致 |

### 9.4 FLOPs 怎么估，为什么不能直接换成耗时

[`autotune/dataflow.py`](../flash_rt/vla/autotune/dataflow.py) 按矩阵形状估算 π0.5 的计算量，只算矩阵乘。

**prefix 的 FFN**：911 个 token，宽 2048，FFN 中间层 16384，gate、up、down 三个矩阵乘：

```text
一个矩阵乘：2 × 911 × 2048 × 16384 ≈ 61.1 G
一层 3 个：≈ 183.4 G
17 层：≈ 3.1 T
```

只算 17 层，因为最后一层的 KV 在 FFN 之前就算好了，这层 FFN 的输出没人用。

**action expert 每次调用**：50 个动作 token，18 层，宽 1024：

```text
QKV、输出投影、FFN：每层约 1.73 G，18 层约 31.1 G
attention 的两个矩阵乘（每个 token 看 911 + 50 个 token）：18 层约 7.1 G
合计：约 38 G
```

计算量差了 80 多倍，耗时只差约 15 倍（24 ms 对 1.65 ms）。action expert 每次只处理 50 个 token，矩阵太小，GPU 算力用不满，时间更多花在读权重和下发 kernel 上。所以 FLOPs 只能比较计算量大小，不能直接换算成耗时。

### 9.5 autotune 怎么计时

[`worker.run_experiment`](../flash_rt/vla/autotune/worker.py) 分别记录加载、prepare 和正式测量。默认 warmup 3 次、正式测 10 次，取中位数。每次正式测量：

```text
从测试输入取出本次请求的数据
GPU 同步
开始计时
backend.infer(...)    # 里面的预处理、上传、prompt 更新、模型计算、拷回 CPU 都算在内
GPU 同步
停止计时
保存结果
```

加载、prepare、warmup 和写结果文件不算在内。`infer` 里如果遇到新 prompt 或新形状触发了额外工作，这部分会算进去。profile 是另外单独跑的请求，不混进正式数据。

### 9.6 小算子 benchmark 和完整模型

单个 GEMM 变快，不一定让整个模型变快：

- 省下的可能只是整次推理里很小的一段。
- 原生替换可能多做了 contiguous 或 dtype 转换。
- 更大的 workspace 或更多 graph 会占显存、拖慢准备。
- Python 调用变少，GPU 算术量不变。

比较性能前，先对齐形状、调用次数、模型调用次数、精度、计时范围和输入。

<a id="quality"></a>
## 十、质量怎么判

### 10.1 误差和 cosine

候选配置（比如 Euler8）和参考配置（比如 Euler10）用相同的输入和初始噪声，比较输出的动作。把一次推理的整个动作块展平成一串数字 A（候选）和 B（参考）：

```text
误差（relative L2）= ‖A − B‖ / ‖B‖
cosine            = (A · B) / (‖A‖ × ‖B‖)
```

- **误差**：整体差了多少，占参考值大小的百分比。
- **cosine**：只看方向像不像，不管大小。1 是完全同向，0.999 大约相当于夹角 2.6°。

两个指标各管一种情况。候选 = 1.05 × 参考时，cosine 还是 1，误差是 5%，只有误差能发现；某几个通道走势变了，cosine 更敏感。误差是随机的时候，cosine ≥ 0.999 大约等于误差 ≤ 4.5%，两个门槛差不多严。

worker 的算法：先取出要评价的通道，每个请求单独算误差和 cosine，再取所有请求里**最大的误差和最小的 cosine**和门槛比，全部过了才算合格。

注意 5% 是整个动作块的误差，不是"每个元素都差 5% 以内"，也不是 PF 的 ε。

### 10.2 Hy 现在的目标

配置在 [`hyvla_tradeoff_quality.json`](../examples/vla_autotune/hyvla_tradeoff_quality.json)：

- 参考：原生 BF16 `semantic_bf16`、Euler10。
- 只比前 20 维有效动作，后面 12 维补零的通道不参与。
- 每个请求误差 ≤ 5%，cosine ≥ 0.999。
- 在合格的配置里选端到端最快的。

之前 Hy 做过和官方实现逐元素对齐的工作，用的是逐元素门槛（每个元素 `|原生 − 官方| ≤ 0.01 + 1% × |官方|`），比误差门槛严得多。那段排查留下的 `hy_trace` 研究工具已从当前入口移除，完整源码仍可按 [清理映射](vla_autotune_cleanup.zh-CN.md) 中的 Git 水位查阅。现在的目标是速度和误差的权衡，不再要求中间结果和官方逐位一致。

但这几件事仍然要对：输入的含义、输出没有 NaN/Inf、每个测试用例都跑到、buffer 每次都刷新、没有重复积分。

<a id="autotune"></a>
## 十一、autotune 怎么跑一轮

先看 [`vla_autotune.md`](vla_autotune.md)，再读代码：

| 模块 | 作用 |
|---|---|
| [`schema.py`](../flash_rt/vla/autotune/schema.py) | 候选配置和结果的数据结构 |
| [`workloads.py`](../flash_rt/vla/autotune/workloads.py) | NPZ 测试输入怎么变成一个个请求 |
| [`worker.py`](../flash_rt/vla/autotune/worker.py) | 一个候选：加载、prepare、warmup、计时、算误差、保存、清理 |
| [`batch.py`](../flash_rt/vla/autotune/batch.py) | 在同一个 Slurm 分配里，每个候选开独立子进程依次跑 |
| [`quality.py`](../flash_rt/vla/autotune/quality.py) | 核对评价范围、通道和门槛是否一致 |
| [`search.py`](../flash_rt/vla/autotune/search.py) | 生成候选；在合格结果里选中位数最快的 |
| [`controller.py`](../flash_rt/vla/autotune/controller.py) | 初始化、提交、等待、记录、汇总 |
| [`ledger.py`](../flash_rt/vla/autotune/ledger.py) | 持久化实验状态 |

### 11.1 一轮的流程

```text
固定源码版本、测试输入、质量门槛和候选列表
  → 参考配置
  → 候选 A
  → 候选 B
  → ...
  → 参考配置再测一次（复测）
  → 收集全部结果
  → controller 核对身份、门槛和复测漂移
  → 选出最快的合格配置，或者不选
```

每个候选在独立子进程里跑，避免上一个候选的模型和 graph 留在显存里影响下一个。

首尾两次参考配置的中位数延迟相差超过 10%，说明这段时间机器状态不稳定，controller 不选 winner。

### 11.2 三种"跑完了"

```text
Slurm job 跑完    ≠ worker 测到了数据
worker 测到数据   ≠ 误差合格
误差合格         ≠ 机器人任务成功
```

只有误差合格的点参与选最快。失败的点不会被当成 0 ms，也不会被选中。

### 11.3 结果不能跨轮次混用

每条结果绑定了这一轮 campaign、完整的候选配置、实际的 job、源码版本和测试输入。名字相同的候选不代表是同一次执行；门槛不同的结果也不能拿来当这一轮的合格结果。

<a id="walkthrough"></a>
## 十二、沿源码走一遍

### 路线 A：公共调用

1. [`BackendConfig.__post_init__`](../flash_rt/vla/config.py)：列出哪些组合合法、哪些报错。
2. [`describe_backend`](../flash_rt/vla/registry.py)：确认它只描述，不加载。
3. [`load_backend`](../flash_rt/vla/registry.py)：找到四个模型的分发位置。
4. [`VLABackend._prepare/_invoke/infer`](../flash_rt/vla/backend.py)：跟着 prompt、state、noise、actions 走一遍。
5. [`ActionDecoder`](../flash_rt/vla/processing.py)：看 actions 怎么从 raw_actions 还原。

**读完应该能**：画出一次请求的调用链，说出哪一步开始上 GPU。

### 路线 B：Hy 的数据流

1. [`HyVLAFlowFrontend.__init__`](../flash_rt/frontends/torch/hyvla_probeflow.py)：支持哪些选项，哪些量化开关被禁用。
2. [`HyVLATorchFrontendThor.infer/predict_actions`](../flash_rt/frontends/torch/hyvla_thor.py)：输入和输出的形状。
3. `_preprocess_images → _vit_merge → _assemble_prefix`：图像怎么变成 prefix。
4. `_make_att_2d/_apply_segment_mask`：mask 决定谁能看见谁，不只是形状。
5. [`_prefix_body/_velocity_body/_sample_latent`](../flash_rt/frontends/torch/hyvla_probeflow.py)：一次 prefix 和多次速度计算怎么拆开。
6. [`pipe.evaluate_velocity`](../flash_rt/models/hyvla/pipeline_thor.py)：action expert 每次具体算了什么。

**读完应该能**：分清机器人 state 和动作变量 x，说出哪些 tensor 整次推理不变、哪些每步都变。

### 路线 C：共享采样器

1. 读 NumPy 版 [`sample_probeflow`](../flash_rt/sampling/probeflow.py)。
2. 手动走一遍 2 步分支和多步分支，数模型调用次数。
3. 读 Torch 版 [`sample_probeflow_torch_`](../flash_rt/sampling/torch_probeflow.py)，标出负号、`clone` 和原地更新。
4. 看 π0、π0.5 的 PF pipeline 怎么把原生 buffer 交给采样器。

**读完应该能**：解释为什么不能把已经积分过的 decoder 输出当速度用，以及为什么模型调用次数不总等于步数。

### 路线 D：原生算子

1. 从 [`FlashRTLinear.forward`](../flash_rt/models/fastwam/native_linear.py) 开始。
2. 在纸上写出 M、N、K 和权重布局。
3. 追到 C++ 的 `bf16_nt_bias`，分清建执行计划、workspace 和真正的矩阵乘。
4. 回到 graph 和 worker，问：这个调用在一次推理里重复了多少次？计时包括了什么？

**读完应该能**：把一个 kernel 放回整次推理里看，而不是读到一个"快"的函数就停。

<a id="exercises"></a>
## 十三、CPU 练习

第二、四节的配置描述和动作还原也是 CPU 练习。下面两个练习看 PF 的选步数逻辑，不加载任何模型。

### 13.1 速度不变：`max_steps=10` 也只走 2 步

```python
# CPU_DEMO: constant_velocity
import numpy as np
from flash_rt.sampling import ProbeFlowConfig, sample_probeflow

noise = np.array([2., 3.])
def velocity(x, t):
    return np.ones_like(x)  # 返回 v；NumPy 版内部会取负

actions, metadata = sample_probeflow(
    noise, velocity, ProbeFlowConfig(epsilon=0.008, max_steps=10),
)
assert np.allclose(actions, [1., 2.])
assert np.array_equal(noise, [2., 3.])
assert metadata["scheduled_steps"] == metadata["nfe"] == 2
print(actions, metadata["timesteps"])
```

两次探路的速度方向完全一样，弯曲度是 0，所以只走 2 步。`max_steps` 只是上限。

### 13.2 速度转向：选 6 步，调用 7 次

```python
# CPU_DEMO: dense_route
import numpy as np
from flash_rt.sampling import ProbeFlowConfig, sample_probeflow

def velocity(x, t):
    return np.array([1., 0.]) if t > 0.75 else np.array([0., 1.])

config = ProbeFlowConfig(epsilon=0.01, max_steps=6)
_, metadata = sample_probeflow(np.zeros(2), velocity, config)
assert metadata["scheduled_steps"] == 6
assert metadata["nfe"] == 7
assert len(config.timesteps()) > metadata["scheduled_steps"]
print(metadata["scheduled_steps"], metadata["nfe"], config.timesteps())
```

对着源码标出：哪一次是被丢掉的 v2，为什么要从原始噪声重新走，哪一次的速度被复用了。最后一行还能看到，预先准备的时间点（8 个）比实际步数（6 步）多。

### 13.3 用测试当接口说明

这些测试都能在 CPU 上跑，适合对照源码读：

| 测试文件 | 看什么 |
|---|---|
| [`test_vla_backend_config.py`](../tests/test_vla_backend_config.py) | 配置、描述，以及不加载模型 |
| [`test_vla_backend_observations.py`](../tests/test_vla_backend_observations.py) | prompt、state、noise，prepare，关闭后的实例 |
| [`test_vla_backend_resources.py`](../tests/test_vla_backend_resources.py) | graph 的归属和释放 |
| [`test_vla_action_processor.py`](../tests/test_vla_action_processor.py) | 动作还原按 checkpoint 声明来 |
| [`test_hyvla_probeflow.py`](../tests/test_hyvla_probeflow.py) | Hy 的速度计算、积分、graph |
| [`test_fastwam_adapter.py`](../tests/test_fastwam_adapter.py) | 首帧条件和实际模型调用次数 |
| [`test_shifted_probeflow.py`](../tests/test_shifted_probeflow.py) | 逻辑时间和真实 sigma |
| [`test_hyvla_tradeoff_objective_example.py`](../tests/test_hyvla_tradeoff_objective_example.py) | Hy 的前 20 维误差门槛 |

<a id="faq"></a>
## 十四、常见误解、排查和自测

### 14.1 常见误解

| 误解 | 实际情况 |
|---|---|
| `describe_backend` 能跑，模型就能用 | 它不加载模型，很多检查要到加载时才做 |
| 用 Torch frontend 就是 Torch 参考实现 | frontend 框架、执行 backend 和参考实现是三件事 |
| `flashrt` 表示每个算子都是原生的 | FastWAM 只换了 Linear，Hy 大部分是 Torch 算子 |
| `prepare` 只是设个标记 | 它会真的跑模型、录 graph |
| 步数减半，延迟减半 | prefix 的时间不随步数减少 |
| 四个模型的文本都只算一次 | FastWAM 的文本走 cross-attention，每一步都重新算 |
| graph 重放后 Linear 计数是 0，就没做 GEMM | Python 计数和 GPU 重放是两回事 |
| tensor 地址和形状都没变，条件就能复用 | 内容可能被原地改了，每次都要刷新 |
| ε 就是最终动作的误差容忍度 | ε 只用来选步数，误差门槛另外定 |
| `actions` 可以直接发给机器人 | 还要核对动作含义、还原方式、单位和安全限位 |
| 误差合格就是任务成功 | 误差门槛只说明和参考输出接近，任务成功要上真机或仿真测 |

### 14.2 遇到问题先分层

```text
创建配置就报错
  → 看 config.py 里哪些组合不合法
描述正常，加载失败
  → 看 registry 的选项、真实 GPU、依赖、checkpoint
加载正常，第一次请求失败或很慢
  → 看 prompt、预处理、prepare、graph 录制、显存
后面的请求好像还在用旧输入
  → 看 π0.5 的 state-in-prompt、buffer 有没有刷新、graph 的 key
动作偏差很大
  → 先查 noise、动作还原方式、时间和积分的负号，再看局部数值
某个配置更快但没被选中
  → 看门槛、评价通道、结果身份、测试用例是否齐全、复测漂移
```

不要从最后一个报错的 CUDA 函数名直接猜模型数学错了，也不要靠放宽误差门槛来掩盖输入或执行上的错误。

### 14.3 自测

1. `BackendConfig`、`load_backend`、`prepare`，哪一步开始真正碰模型和 GPU？
2. π0.5 为什么 prompt 不变、state 变了，也要重新处理 prompt？
3. Hy 的 147 个视觉 token 为什么不等于 prefix 长度（240）？
4. 模型输出的速度、采样器用的进度速度、最终的 actions，三者有什么区别？
5. PF 选 6 步时为什么调用 7 次模型？预先准备的时间点为什么更多？
6. FastWAM 模型输入的时间乘了 1000，为什么积分步长不乘？
7. KV cache、CUDA Graph、GEMM 执行计划分别缓存了什么？
8. 某个 GEMM 原型更快，为什么不能直接推出整个模型更快？
9. π0.5 的 prefix 计算量是每步 action expert 的 80 多倍，为什么耗时只差 15 倍？
10. Hy 不再要求和官方逐位一致，为什么 mask、noise、模型调用次数和 buffer 刷新仍然要对？
11. FastWAM 的 KV cache 里有什么、没有什么？每一步里哪部分计算其实可以只算一次？

能回答这些，再去深入某个 attention 或 GEMM 实现，就知道自己在优化整条数据流里的哪一段。

## 延伸阅读

- [统一后端 API 参考](vla_backend.md)：查参数和支持的组合。
- [autotune 机制](vla_autotune.md)：查实验流程、质量门槛、计时和选择规则。
- [autotune pipeline](vla_autotune_pipeline.zh-CN.md)：工具池、按游选工具、worktree 对比、skill 和“谁主导”的记录。
- [五模型整合报告](reports/2026-11-24-vla-autotune-final.zh-CN.md)：当前历史结论、质量失败与新验收待证项。
- [完整历史逐轮记录](reports/2026-11-24-vla-autotune-final.zh-CN.md#historical-record)：旧实验数值和来源身份。
- [研究过程与认识修正](reports/2026-11-24-vla-autotune-final.zh-CN.md#research-history)：Hy 原 EXACT 主线、反例与随后目标修订。
