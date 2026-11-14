# 从一次 `infer` 理解 FlashRT：VLA 后端源码学习指南

> **用途：学习现有实现，不是重构方案，也不是新的性能验收报告。**
>
> 本文依据本分支 `9804654f` 时的源码梳理，日期为2026-11-13。重点是我们正在使用的
> π0、π0.5、Hy和FastWAM统一执行层，不试图覆盖FlashRT中所有LLM、音频、视频和硬件路径。
> 文中“支持的源码路径”不等于所有环境都已运行认证。

如果你只先记住一句话：

**后端把“模型如何执行”和“动作如何通过多次速度评估生成”分开；autotune再在它们之上比较速度与输出偏差。**

**第一次阅读建议：**先读1、3、4、6节，再运行第14节CPU练习；想结合当前Hy工作理解数据流，
接着读7.3和第13节路线B。第9节C++／CUDA可以最后再看，不必从kernel开始啃。

## 阅读导航

1. [先建立心智模型](#overview)
2. [分层和目录地图](#layers)
3. [配置、描述和加载不是一回事](#configuration)
4. [跟踪一次完整推理](#request)
5. [读懂输入、噪声和输出](#contracts)
6. [Euler与ProbeFlow究竟在做什么](#sampling)
7. [四个模型为什么不能共用同一个head](#models)
8. [CUDA Graph、KV缓存和GEMM计划缓存](#graphs)
9. [一条Linear如何走到C++／CUDA](#native)
10. [性能应该怎样理解](#performance)
11. [质量预算与Hy当前目标](#quality)
12. [autotune如何组织和选择实验](#autotune)
13. [沿源码实际走一遍](#walkthrough)
14. [不需要GPU的学习练习](#exercises)
15. [常见误解、排查路线与自测](#faq)

<a id="overview"></a>
## 1. 先建立正确的心智模型

### 1.1 这里的“后端”不是HTTP服务端

这里主要讨论 `flash_rt.vla`：一个**在进程内调用模型推理的执行接口**。
它本身不负责HTTP路由、机器人通信、任务规划或训练。

可以把它放进下面这幅图：

```text
机器人／数据集／调用方
    │ 提供 observation、prompt、初始 noise
    ▼
统一VLA接口：BackendConfig + VLABackend
    │ 选择模型与执行配置，管理调用和结果契约
    ▼
模型frontend
    │ 加载checkpoint、处理图像／状态／文本、选择具体pipeline
    ▼
模型pipeline + sampler
    │ 条件编码／KV prefill + 多次速度评估 + 动作积分
    ▼
Torch算子／FlashRT扩展／cuBLASLt／attention实现
    │
    ▼
GPU计算 → CPU动作数组 + 采样／执行metadata
```

autotune在这条链的**外部**组织候选、计时、质量比较和结果入账，并不是每个kernel里都在做搜索。

### 1.2 三个不同层次的问题

| 问题 | 应该先看哪里 |
|---|---|
| “我怎样加载模型并调用推理？” | `flash_rt/vla` |
| “这张图像和这个state怎样变成动作？” | 模型frontend和pipeline |
| “这个矩阵乘为什么快／慢？” | 具体算子绑定、shape、执行次数和测量 |

一开始就钻进CUDA kernel，容易不知道它在整个请求里执行了几次、输入从哪里来、输出是否已经做过积分。
先理解数据流，再向下追算子，效率更高。

### 1.3 最重要的几个名词

| 名词 | 在这里的含义 |
|---|---|
| observation | 调用方提供的一次观测；各模型的字段和预处理不同 |
| frontend | 模型与框架／checkpoint的适配器，不是网页前端 |
| pipeline | 把模型的实际算子和缓冲区组织成推理过程 |
| prefix／condition | 当前请求中基本不随去噪步变化的图像／语言等条件 |
| KV cache | 条件侧attention的K/V结果，供本请求后续评估复用 |
| latent／state `x` | 正在迭代更新的动作变量；不要和机器人观测state混淆 |
| raw velocity | 模型在当前latent和时间上的速度预测，尚未完成积分 |
| sampler | 决定何时评估速度、评估几次，以及如何更新latent |
| NFE | Neural Function Evaluations，实际神经速度评估次数 |
| graph capture／replay | 记录并重放一段GPU执行图，不等于缓存输出答案 |
| quality scope | 当前质量指标究竟评价哪些动作域、通道和参考点 |

<a id="layers"></a>
## 2. 分层与目录地图

### 2.1 建议按这五层看

```text
实验层       flash_rt/vla/autotune/
              controller、ledger、worker、batch、quality、search
                    │
统一执行层   flash_rt/vla/
              config、registry、backend、processing、resources
                    │
模型适配层   flash_rt/frontends/torch/ 或 frontends/jax/
              checkpoint、输入、prompt、模型专有配置
                    │
计算组织层   flash_rt/models/<model>/ + flash_rt/sampling/
              prefix、raw velocity、采样、graph、buffer
                    │
算子层       flash_rt/core/、csrc/、已安装的CUDA扩展、Torch后端
```

### 2.2 先读的小文件

| 文件 | 你要从中理解什么 |
|---|---|
| [`vla/config.py`](../flash_rt/vla/config.py) | 一份执行配置包含哪些维度，哪些组合在创建时就拒绝 |
| [`vla/registry.py`](../flash_rt/vla/registry.py) | 模型ID如何映射到frontend、精度和硬件检查 |
| [`vla/backend.py`](../flash_rt/vla/backend.py) | `prepare → infer → close`的公共生命周期 |
| [`vla/processing.py`](../flash_rt/vla/processing.py) | checkpoint声明的动作反归一化如何被遵守 |
| [`sampling/probeflow.py`](../flash_rt/sampling/probeflow.py) | 不依赖GPU的ProbeFlow配置、路由和数值参考 |
| [`sampling/torch_probeflow.py`](../flash_rt/sampling/torch_probeflow.py) | 模型速度回调与实际latent更新之间的接口 |

大frontend通常混有checkpoint转换、量化、CFG、批处理等多个模式。
第一次阅读不必逐行读完，先顺着本次配置实际走的分支。

### 2.3 `backend`、`framework`和`hardware`不是同一个开关

以 `BackendConfig` 为例：

| 字段 | 回答的问题 | 需要注意 |
|---|---|---|
| `model` | 哪个模型家族？ | `pi0.5`会规范化为`pi05` |
| `backend` | 选择哪种执行实现？ | 当前只有FastWAM额外接受`torch`，不是所有模型都可随意切换 |
| `framework` | 使用哪种frontend／权重加载生态？ | Pi的`jax`路径不意味着整个模型在XLA里执行 |
| `hardware` | 预期目标GPU是什么？ | 实际加载时核对真实compute capability，不能用字符串假装架构 |
| `precision` | 统一接口约定的模型精度是什么？ | 不是承诺每个中间操作都采用同一dtype |
| `sampler` | 固定Euler还是自适应ProbeFlow？ | 不改变它们应调用的模型专有速度head |
| `options` | 模型特有选项是什么？ | 支持集合由registry限制，不能任意塞frontend隐藏开关 |

两个尤其容易误解的地方：

- `framework="torch"`不等于`backend="torch"`。Hy可以由Torch frontend执行，配置标签仍为`flashrt`。
- `backend="flashrt"`不等于“所有算子都是手写CUDA”。FastWAM是明确标注的**部分原生Linear替换**，Hy的当前BF16路径也包含大量Torch算子。

<a id="configuration"></a>
## 3. 配置、描述和加载：三个不同阶段

### 3.1 `BackendConfig`只是执行意图

创建配置会规范模型名并检查基本组合，例如：

- Euler不能同时带一份ProbeFlow配置。
- ProbeFlow不能用`num_steps`表达自适应预算；应使用`probeflow.max_steps`。
- 当前统一接口不接受任意量化精度。

但**配置成功不等于checkpoint存在、依赖齐全、所有模型专有参数和GPU都兼容**。
后面还有registry和frontend层的检查。

### 3.2 CPU示例：看描述，不加载模型

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

这里的路径是占位符，没有读取checkpoint。`runtime_verified=False`是刻意提醒：**这是描述，不是认证结果**。

还可以在源码目录执行：

```bash
python -m flash_rt.vla --model hyvla --checkpoint /local/checkpoint-placeholder \
    --hardware cuda_sm80 --sampler euler --num-steps 10
```

这个CLI默认就是描述模式，不能把它打印JSON的耗时当作模型推理延迟。

### 3.3 `load_backend`才进入实际加载

[`registry.load_backend`](../flash_rt/vla/registry.py)大致执行：

```text
检查framework／precision／options
  → 按需要读取Pi动作processor
  → 检查真实CUDA设备和hardware
  → 选择frontend类与构造参数
  → 加载模型／分配资源
  → 包成VLABackend
```

动态导入的目的之一，是让只描述候选的工具不必先加载每个模型的Torch、tokenizer、扩展和GPU状态。

`flash_rt.load_model`是项目原来的一般模型入口；本文追踪的是专门的
`flash_rt.vla.load_backend`。不要认为两者的参数和分发行为可以直接互换。

<a id="request"></a>
## 4. 跟踪一次完整推理

### 4.1 公共调用的骨架

下面是GPU用法的**接口示意**，本次文档工作不执行这段代码。
需要已有权重、依赖和合法GPU计算环境；在当前集群上，模型计算必须位于分配到的Slurm compute step。

```python
# GPU_EXAMPLE: interface sketch only; not executed for this document
import numpy as np
from flash_rt.vla import BackendConfig, load_backend

config = BackendConfig(
    model="hyvla", checkpoint="/local/hyvla-checkpoint",
    hardware="cuda_sm80", sampler="euler", num_steps=10,
    options={"use_cuda_graph": True, "numerical_profile": "semantic_bf16"},
)

# 只用于说明当前三相机、六帧fixture的形状；应换成真实代表性输入。
# 列表顺序必须对应checkpoint相机顺序，不能随意交换左右相机。
observation = {
    "images": [np.zeros((6, 480, 640, 3), dtype=np.uint8) for _ in range(3)],
    "state": np.zeros(20, dtype=np.float32),
}
noise = np.zeros((40, 32), dtype=np.float32)

with load_backend(config) as backend:
    backend.prepare(observation, prompt="pick up the object")
    result = backend.infer(observation, noise=noise)
    print(result["actions"].shape)  # 当前Hy统一infer路径：(40, 32)
    print(result["sampler_metadata"])
```

零图像／零噪声不是机器人任务样本，也不是合适的质量或性能代表集。
输入形状正确只说明接口可能接受，不说明模型行为合理。

### 4.2 `prepare()`不是空操作

读 [`VLABackend.prepare`](../flash_rt/vla/backend.py) 时分两步看：

1. `_prepare`确定有效prompt，必要时调用`set_prompt`，并处理模型的校准准备。
2. 若frontend没有独立`calibrate_with_real_data`接口，显式`prepare`会执行一次完整请求并丢弃其结果，以真正准备运行路径。

所以：

- `prepare`可能触发tokenization、模型计算、warmup和graph准备，不能在CPU dry-run中随便调用。
- 它和正式测量的`infer`不是同一个概念。
- 后续`infer`仍会检查prompt／状态等变化；不是调用过一次`prepare`后所有输入都可无条件复用。

### 4.3 `infer()`里的统一层在做什么

主要链路是：

```text
VLABackend.infer
  ├─ 持有实例锁，检查实例未关闭
  ├─ _prepare：选择prompt，按需更新模型准备状态
  ├─ _invoke：整理prompt／noise，调用frontend.infer
  ├─ 按checkpoint需要修正actions解码
  ├─ 检查／补充sampler metadata
  └─ 加入backend metadata，返回结果
```

注意`_invoke`有一个很实用的适配：

- frontend若接受`noise=`关键字，就用该参数传递。
- frontend若从observation里读noise，就把它放进一份浅拷贝的observation。
- 两处重复提供noise会拒绝，避免不知道哪份实际生效。

这让Pi和Hy不同的历史接口，可以在外部保持同一个调用方式。

### 4.4 为什么同一个prompt字符串也可能需要更新

π0.5可以把state编码进prompt，所以只缓存字符串会错。
统一后端对π0.5还比较state是否变化，必要时重新调用`set_prompt`。

但Hy的state主要进入动作侧，不把“传入了state参数”都理解成state-in-prompt。
这正是统一接口不能抹掉模型差异的一个例子。

### 4.5 一次请求内部的计算

可以先用这个简化图理解，再去对应每个模型：

```text
图像／文本／机器人观测state
            │
            ▼
条件预处理与视觉／语言编码
            │
            ▼
prefix／video KV prefill ───────────────┐
                                      │ 本请求固定条件
初始noise → latent x0                  │
                │                     │
                ├─ evaluate(x0,t0) ◀───┤
                ├─ 更新x1             │
                ├─ evaluate(x1,t1) ◀───┤
                ├─ ...                │
                └─ 最终动作latent     │
                         │            │
                         ▼            │
              CPU动作／可选解码         │
```

一次`infer`往往包含多次神经网络速度评估，**不是一次矩阵乘，也不是一次Transformer forward就结束**。

<a id="contracts"></a>
## 5. 读懂输入、噪声和输出

### 5.1 “统一接口”不意味着“统一机器人预处理”

后端不自动猜测：

- 摄像头语义及排列顺序；
- state是原始单位还是已归一化；
- 历史帧如何采样、补齐和排序；
- 左右手、关节、旋转、夹爪等动作协议；
- 某个数据集的归一化统计是否适合另一台机器人。

这些必须由checkpoint和调用方明确。**shape相同也可能语义完全不同。**

### 5.2 四条路径的边界对照

| 模型 | 主要输入边界 | 输出边界 |
|---|---|---|
| π0 | 现有Pi图像／连续state／prompt契约；显式初始noise | `raw_actions`通常是`[chunk,32]`；公开actions按frontend／checkpoint processor解码，当前LIBERO场景常为7维 |
| π0.5 | 图像／prompt，state可能进入prompt；显式noise | 同样区分normalized raw与解码后的公开actions，不保证所有checkpoint都固定public7 |
| Hy | 当前fixture为三相机六帧uint8历史、20维归一化state、`[40,32]`noise；frontend做resize／pad等 | 统一`infer`给normalized padded `[40,32]`；直接`predict_actions`仍有batch维`[1,40,32]` |
| FastWAM | first-frame输入图像、预计算T5 context／mask、normalized proprio及`[1,H,A]`noise | normalized batched `[1,H,A]`；`H/A`须符合实际checkpoint配置 |

Hy的40／32／20以及Pi的public7是当前模型／fixture的具体约定，不是所有VLA的通用常量。

### 5.3 `actions`不一定已经是机器人指令

先看这几个字段：

```text
result["actions"]
result.get("raw_actions")
result["backend_metadata"]["action_domain"]
result["backend_metadata"].get("action_processor")
```

[`processing.py`](../flash_rt/vla/processing.py)的意义尤其容易漏掉：

- 若Pi checkpoint声明了`policy_postprocessor.json`，统一层遵守它声明的ACTION模式。
- `MEAN_STD`意味着按mean/std还原；不能因为同一目录也存着q01/q99，就擅自改用quantiles。
- 没有该声明文件时，保留原frontend／OpenPI／Orbax的既有契约。

解码是模型输出契约的一部分，不是无关紧要的显示格式。
但即使完成checkpoint解码，也仍要核对真实机器人的单位、限位和安全协议。

### 5.4 CPU例子：normalized动作和解码动作不是一个数组

```python
# CPU_DEMO: decode
import numpy as np
from flash_rt.vla.processing import ActionDecoder

decoder = ActionDecoder(
    mode="MEAN_STD", dimension=2,
    first=np.array([10., 20.], dtype=np.float32),  # mean
    second=np.array([2., 4.], dtype=np.float32),  # std
    eps=1e-8,
)
raw = np.array([[0.5, -0.5, 99.]], dtype=np.float32)
decoded = decoder(raw)
assert decoded.tolist() == [[11., 18.]]
assert raw.tolist() == [[0.5, -0.5, 99.]]
print(decoded)
```

这里第3个通道没有被声明为公开动作，所以没有进入解码输出。
这只是数学示例，mean/std不是某台真实机器人的统计值。

<a id="sampling"></a>
## 6. Euler与ProbeFlow：模型计算和积分必须分开

### 6.1 三个变量不要混用

- 机器人观测state：作为模型条件输入。
- 动作latent `x`：从noise开始，采样器逐步更新。
- 模型时间`t`：告诉模型当前去噪阶段。

连续时间模型可以记成速度场`v_theta(x,t;condition)`。
模型评估返回速度，不应在这个接口里偷偷完成整段采样。

### 6.2 为什么实现里经常看到负号

模型时间从1走向0。直接写Euler可以是：

```text
x_next = x + (t_next - t) * v_theta(x,t)
```

本仓库共享采样器换成正向进度`s=1-t`，令`u=-v_theta`：

```text
x_next = x + delta_s * u(x, 1-s)
```

二者可以描述同一个更新。关键不是哪个符号“看着自然”，而是**负号和步长只能应用一次**。

尤其注意两份参考接口并不一样：

| 接口 | 回调提供什么 |
|---|---|
| NumPy `sample_probeflow(noise, velocity, ...)` | `velocity(x,t)`返回模型时间速度`v_theta`，函数内部取负 |
| Torch `sample_probeflow_torch_(state, evaluate, ...)` | `evaluate(t)`已经返回进度速度`-v_theta`，函数不再替你取负 |

把一个接口的回调原样接到另一个接口，可能多取或少取一次负号。

### 6.3 Euler的N步是什么

固定Euler使用预先确定的时间表，每步评估一次、更新一次。
理想化均匀进度写法是：

```text
for i in 0..N-1:
    u = evaluate(time_i)
    x += u / N
```

但源码中步长可能被折入投影权重、某个GEMM的缩放参数，或者按模型原有表达式计算。
因此修改`N`不能只缩短循环，还要检查时间embedding、输出scale和积分是否同步改变。

π0的转换函数明确区分Euler的`-1/N`与ProbeFlow的`-1`，避免从已经舍入的Euler投影倒推出raw velocity。

### 6.4 ProbeFlow在这里怎样选预算

共享实现的主要步骤是：

1. 在原始noise处评估第一份速度。
2. 前进一个probe区间，在预测latent上再评估一次。
3. 用两份速度的cosine构造启发式曲率信号`1-cosine`。
4. 结合`epsilon`选择可达的采样步数。
5. 选2步时复用两次probe；选更密时间表时，从原noise重新起步，复用第一份速度，但丢弃lookahead作为积分步。

这不是一个经过认证的ODE误差上界，`epsilon=0.008`也**不等于最终动作允许0.8%误差**。
采样路由阈值与最终动作质量预算是不同参数。

### 6.5 为什么NFE可能不是scheduled steps

| 分支 | 实际NFE |
|---|---:|
| 固定Euler N步 | N |
| ProbeFlow选择2步 | 2 |
| ProbeFlow选择密集N步，N>2 | N+1 |

密集分支多出来的一次是用于路由、最后未作为该时间表积分速度使用的lookahead。
“先probe再重启”不意味着第一份速度也必须重算。

`probe_horizon=0.5`时，两probe分支对应均匀Euler-2网格；若改成其他horizon，
两个积分区间并不等长，不能继续简称为同一均匀Euler2。

Torch实现会clone第一份probe速度，因为native evaluator可能复用同一scratch输出。
若只保留一个view，下一次评估可能覆盖“第一份速度”，连路由cosine都会算错。
这属于缓冲区别名正确性，不是额外追求数值精确。

还有第三个数量：**预先准备的时间表并集长度**。
它决定可能需要的embedding／graph条目，不是某次请求真正执行的步数。

### 6.6 FastWAM还有一个时间坐标映射

[`ShiftedSigmaSchedule`](../flash_rt/sampling/schedules.py)实现：

```text
sigma(u) = shift*u / (1 + (shift-1)*u)
model_timestep = time_scale * sigma
action_update = (sigma_next - sigma) * raw_velocity
```

把模型时间乘以1000，不意味着把动作积分步长也乘1000。
FastWAM的共享ProbeFlow适配保留逻辑clock作为查表key，但按真实sigma差积分。
这是一项明确的shifted-grid适配，不是“所有论文默认epsilon都已适用FastWAM”的结论。

<a id="models"></a>
## 7. 四个模型的适配差异

### 7.1 π0：连续state token和FP16动作缓冲

入口：[`frontends/torch/pi0_rtx.py`](../flash_rt/frontends/torch/pi0_rtx.py)。
计算组织：[`models/pi0/pipeline_rtx.py`](../flash_rt/models/pi0/pipeline_rtx.py)。
自适应路径：[`pipeline_rtx_probeflow.py`](../flash_rt/models/pi0/pipeline_rtx_probeflow.py)。

阅读要点：

- decoder包含`[state token; action tokens]`，state query有自己的mask约束。
- 输出velocity／积分只针对action部分，不能把state token也当成一个动作步。
- 这条统一接口采用FP16，不是因为两个dtype都占2字节就能把buffer解释成BF16。
- native pipeline持有CUDA缓冲区和指针；Torch采样器通过明确dtype的view接入。
- ProbeFlow复用prefix和单步decoder评估，不把已经执行Euler更新的完整decoder当作raw field。

### 7.2 π0.5：state-in-prompt与AdaRMS时间调制

入口：[`frontends/torch/pi05_rtx.py`](../flash_rt/frontends/torch/pi05_rtx.py)。
计算组织：[`models/pi05/pipeline_rtx.py`](../flash_rt/models/pi05/pipeline_rtx.py)。
自适应路径：[`pipeline_rtx_probeflow.py`](../flash_rt/models/pi05/pipeline_rtx_probeflow.py)。

与π0相比：

- state可能编码进prompt，因此prompt cache需要关注state。
- action expert包含模型专有的时间／AdaRMS调制，不是把π0的head换个名字。
- 统一后端选择BF16非量化路径；大frontend中其他FP8／FP4／CFG等分支不能自动算进本API支持范围。
- ProbeFlow当前不支持的RTC、跨帧decode-only复用和runtime export等边界会显式拒绝，不能把所有Euler能力无条件搬过来。

文件名里有`rtx`不意味着实际调用可以跳过硬件检查，也不意味着已在所有消费级GPU上验证。

### 7.3 Hy：历史视觉、Merger、prefix与动作Expert

入口：[`frontends/torch/hyvla_probeflow.py`](../flash_rt/frontends/torch/hyvla_probeflow.py)，
继承基础输入／权重逻辑自[`hyvla_thor.py`](../flash_rt/frontends/torch/hyvla_thor.py)。
主要BF16计算在[`models/hyvla/pipeline_thor.py`](../flash_rt/models/hyvla/pipeline_thor.py)。

按当前三相机／六帧fixture，可以把shape链记为：

```text
三份 [6,480,640,3] uint8历史
  → 相机／通道布局整理，resize+pad与归一化
  → [3,6,3,224,224] BF16
  → ViT patch：每帧196个token、宽度1152
  → 空间／时间处理，保留送入Merger的当前帧表示
  → [3,196,1152]
  → Merger的投影和2×2池化
  → [3,49,2048]
  → 图像特殊token＋语言token，形成完整prefix
  → prefix排序／mask／RoPE，prefill K/V
  → state embedding＋多次动作速度评估
  → FP32动作latent，公开infer返回 [40,32]
```

其中`3×49=147`是这个例子的视觉token数，**不等于整个prefix长度**。
prefix还包含行分隔、起止token和语言部分。

`HyVLAFlowFrontend`中的三个函数值得按顺序读：

1. `_prefix_body`：prefill条件K/V，并嵌入机器人state。
2. `_velocity_body`：调用真实`evaluate_velocity`，把raw速度转换为共享进度速度。
3. `_sample_latent`：Euler或ProbeFlow负责动作latent更新。

`semantic_bf16`是当前原生路径；`official_local_fp32`是另一条显式局部算术配置。
后者名字不表示“完整官方模型精确等价”，也不是当前性能优化必须先启用的模式。

### 7.4 FastWAM：first-frame video cache与部分native Linear

入口：[`frontends/torch/fastwam.py`](../flash_rt/frontends/torch/fastwam.py)。
计算组织：[`models/fastwam/pipeline.py`](../flash_rt/models/fastwam/pipeline.py)。

关键调用链：

```text
FastWAMActionPipeline.infer
  → _initial_state：初始动作noise
  → _prepare_condition
      ├─ 图像VAE编码
      ├─ video_expert.prepare
      └─ mot.prefill_video_cache_tensor
  → Euler或ProbeFlow
      └─ _denoise_action_with_video_cache：只评估raw velocity
  → normalized CPU actions [1,H,A]
```

这一适配不调用上游完整`infer_action`来冒充一次速度评估。
加载器需要已有上游源码、resolved model config、完整训练checkpoint和本地VAE；
通常由调用方提供预计算T5 context／mask，不在这条本地加载路径中下载或加载T5。

`backend="torch"`和`backend="flashrt"`共用这条first-frame采样组织。
前者不是独立的上游sampler oracle；后者把受支持的活跃Linear模块换成原生GEMM。
VAE、attention、norm、激活等仍有上游Torch计算，不能称为完整原生移植。

<a id="graphs"></a>
## 8. 不要把三种“缓存”混成一种

### 8.1 KV cache：缓存本请求的条件表示

同一请求中图像／语言条件通常不随动作去噪步改变，因此prefix K/V可以算一次供多次速度评估使用。
但新请求的图像、prompt、state或mask变化后，应刷新相应条件。

“一次请求内复用prefix”不等于“跨相机帧永远复用旧prefix”，也不是LLM那种单纯追加一个新token的场景。

### 8.2 CUDA Graph：缓存执行结构，不缓存动作答案

概念上是：

```text
预先分配buffer → warmup → capture执行结构 → 每请求更新输入 → replay
```

它主要减少CPU侧反复分发许多GPU操作的开销。**GEMM、attention等GPU工作仍然执行**。
它既不让GPU FLOPs消失，也不意味着零开销。

原生Pi使用[`core/cuda_graph.py`](../flash_rt/core/cuda_graph.py)包装的CUDA Runtime graph；
Hy和FastWAM还各有自己的Torch graph组织。不能把一种graph的所有权／清理规则直接套到另一种。

### 8.3 为什么ProbeFlow不能把整条采样路线固定捕获

ProbeFlow必须在本次输入的probe之后才决定路线。
若capture时碰巧走2步，之后不加区分地重放整张2步图，就不再是输入自适应算法。

当前设计通常把以下部分分开：

- 静态prefix graph；
- 按预计算时间key准备的单次velocity graph，或FastWAM可更新timestep的raw-field graph；
- graph外的host路由与动作积分。

Torch实现里cosine的标量读取也有host决策成本；不能把这部分时间从算法计时中隐去。

### 8.4 同shape不等于同内容

固定地址只是graph重放的条件之一。
每次请求还必须把新的图像条件、prompt相关数据、state、noise和mask写入相应buffer。

一个危险的错误是：“上次和这次tensor指针一样，所以context没变”。
调用方完全可能原地更新同一个tensor的内容。

FastWAM的raw-field graph限定固定tensor-tree几何；shape／stride／dtype／device不符时会拒绝，
不偷偷recapture或切回eager。这是边界约束，不是“无限动态shape graph”。

### 8.5 GEMM plan cache：缓存选择工作，不缓存矩阵乘结果

cached GEMM runner复用descriptor、heuristic等执行计划，并复用自己的workspace。
相同shape可以减少host选择开销，但每次新的activation仍会做矩阵乘。

| 缓存 | 复用的是什么 | 不代表什么 |
|---|---|---|
| KV cache | 条件K/V结果 | 新请求的条件可以不刷新 |
| CUDA Graph | GPU执行结构和地址绑定 | 重放时不计算模型 |
| GEMM plan cache | 矩阵乘的执行计划 | activation输出已经算好 |

### 8.6 为什么实例不能随意并发复用

后端／frontend持有可变的latent、K/V、scratch、graph输入和runner workspace。
`VLABackend`用实例锁串行化调用，但这不等于协调了进程里其他线程的所有CUDA工作。

FastWAM特别要求首次prepare／capture时进程CUDA工作处于quiescent状态：
也就是本进程内没有其他线程同时提交无关CUDA工作。
同一后端实例不是一个自动支持任意并发、多请求microbatch的服务引擎。

### 8.7 `close()`到底保证什么

看[`backend.py`](../flash_rt/vla/backend.py)和[`resources.py`](../flash_rt/vla/resources.py)：

- **统一层**在实例锁内调用可选的frontend `close()`，为Pi额外处理原生graph，然后释放自身frontend引用并标记关闭。
- **Pi路径**会同步相关graph stream，处理prompt cache和退休pipeline，并去重销毁原生graph／exec句柄。
- **FastWAM路径**会排空它维护的执行stream（或最后一次eager stream），关闭raw-field graph，再释放相关引用。
- **Hy当前路径**没有单独实现frontend `close()`；统一层释放frontend引用，graph／存储后续依赖Torch对象生命周期。
  不能据此推断它实现了与Pi／FastWAM相同的显式stream drain和句柄销毁流程。

所以，统一`close()`**不是所有模型共享的一道相同CUDA同步屏障**。
这也不是在断言Hy存在新的资源安全缺陷，而是在区分源码确实实现了哪些保证。
失败时不能把“有延迟样本”直接当成完整成功；部分路径保留重试清理能力。

关闭并不保证进程显存数字立刻归零：allocator、CUDA context或其他对象可能仍然持有资源。
当前后端也不会为此擅自flush进程全局缓存。

<a id="native"></a>
## 9. 从一条Linear看到C++／CUDA

学习原生算子边界时，FastWAM的[`native_linear.py`](../flash_rt/models/fastwam/native_linear.py)比较直观。

### 9.1 数学和shape先写清楚

```text
activation A: [M,K]
weight W:     [N,K]
bias:         [N]
output D:     [M,N]

D = A @ W.T + bias
```

当输入原来是`[B,S,K]`时，前面的维度可展平为`M=B*S`，最终再恢复输出shape。
这里权重已经按`[N,K]`保存，不能再无条件做一次完整转置副本。

### 9.2 当前调用路径

```text
普通nn.Linear模块
  → 实例内安装FlashRTLinear
  → forward整理activation／weight／bias
  → runner.bf16_nt_bias(data_ptrs, M,N,K, stream)
  → PyBind C++入口
  → cuBLASLt matmul
```

参考：[`csrc/bindings.cpp`](../csrc/bindings.cpp)、
[`gemm_runner.cu`](../csrc/gemm/gemm_runner.cu)、
[`bf16_nt_plan.cpp`](../csrc/gemm/bf16_nt_plan.cpp)。

有两个特别重要的细节：

1. 原始指针不会自动告诉Torch这些tensor被哪个stream使用，所以wrapper显式记录stream／生命周期关系。
2. cached runner有自己的设备、workspace和受限对齐契约；不能把任意offset view当成合法base buffer，并指望缓存算法照样适用。

这些是执行正确性条件，不是为了追求过高数值精确。

### 9.3 “调用计数”也有层次

Python Linear的`forward`计数、成功的CPU GEMM enqueue计数、graph中编码的逻辑Linear工作和GPU replay次数不是一回事。

capture后，GPU replay可以重复执行matmul，而不再经过Python模块`forward`。
因此“Python计数为0”不表示“GPU没有做Linear”。
看FastWAM的`execution_metadata.velocity_graph`时，应结合replay和encoded-work字段，而非只看一个host计数器。

<a id="performance"></a>
## 10. 怎样理解速度，而不是被毫秒数误导

### 10.1 一次请求不只包含action head

一个便于思考的近似分解是：

```text
T_request ≈ T_input/condition + T_prefix
          + NFE × T_velocity
          + T_routing/integration + T_output/dispatch
```

这不是严格可相加的GPU kernel计时公式；实际有stream、依赖、host提交和同步。
它的用途是帮助你问：“这项优化究竟减少了哪一部分？”

### 10.2 少一半NFE，不一定快一倍

假设固定前置成本20ms，每次velocity成本2ms：

```text
10NFE：20 + 10×2 = 40ms
 5NFE：20 +  5×2 = 30ms
```

速度提升是`40/30≈1.33×`，不是2×。
当prefix占主导时，继续减少action NFE的收益自然受限。

### 10.3 当前worker测量的边界

[`worker.run_experiment`](../flash_rt/vla/autotune/worker.py)分别记录load、prepare和正式样本。
正式样本大致是：

```text
从workload取得本请求的host数据拷贝
sync
开始计时
backend.infer(...)  # 其内部触发的预处理／上传／prompt更新／模型／CPU输出都计入
sync
结束计时
计时后保存结果
```

初始load、显式prepare、warmup和NPZ文件IO不在steady样本里。
但是，**不能因此声称计时内永远没有准备工作**：若`infer`遇到新的prompt或shape而触发实际工作，它仍在这个边界内。

profile是额外请求，不应把它的耗时混进正式样本。
CPU wall-clock、CUDA event区间和kernel时长总和也不是相同指标。

### 10.4 为什么小算子benchmark不能代替完整模型

- GEMM选择开销减少，可能只节约了很小一段请求时间。
- 原生替换可能多做contiguous或dtype转换。
- 更大的workspace／graph bank会影响内存和准备成本。
- 更少的Python调用，不意味着更少的GPU算术。
- FLOPs约`2MNK`是计算量估算，不是实测时间。
- “权重尺寸×字节数”是逻辑数据量，不等于本次实际HBM流量。

理解性能前，先明确shape、调用次数、NFE、精度、测量边界与输入是否相同。
不要预设FlashRT标签一定胜过Torch标签。

<a id="quality"></a>
## 11. 质量预算、精度和“正确”是三件事

### 11.1 三种检查分别回答不同问题

| 检查 | 问题 | 不能代替什么 |
|---|---|---|
| 执行契约 | 输入／shape／time／NFE／buffer／资源是否正确？ | 不能证明动作适合任务 |
| 数值代理 | 固定输入下，候选输出偏离参考多少？ | 不能直接当作机器人成功率 |
| 闭环任务评价 | 机器人是否在代表性任务中成功且安全？ | 不能仅用几个固定数组代替 |

FP16、BF16是数据与运算配置；它们不是“模型质量分数”。
比如Hy权重／投影是BF16，但动作latent和部分操作可以是FP32。

### 11.2 当前Hy目标已经转为性能—质量权衡

按2026-11-13用户要求，当前配置
[`hyvla_tradeoff_quality.json`](../examples/vla_autotune/hyvla_tradeoff_quality.json)定义：

- 对固定原生BF16 `semantic_bf16` Euler10执行参考比较；
- 评价normalized public20动作通道，不让12个padding通道否决候选；
- 每请求／回放relative L2≤0.05、cosine≥0.999；
- 在完整覆盖、质量通过和对照稳定的点中选择更快的端到端配置。

以参考数组`B`和候选`A`记：

```text
relative_L2 = ||A-B||₂ / max(||B||₂, 1e-12)
cosine = <A,B> / (||A||₂ ||B||₂)  # 零范数时实现另有明确处理
```

worker先选定公开通道，再对每个请求的动作chunk展平计算。
多请求中取最坏relative L2和最小cosine，不是用平均值掩盖某个难例。

这里的5%预算不是“每个元素最多差5%”，也不是ProbeFlow的`epsilon`。
参考点自身被标为reference角色，不代表它是独立官方正确性oracle。

### 11.3 为什么历史EXACT研究不应成为现在的性能前提

`examples/vla_autotune/hy_trace/`中的冻结算子、局部交接和LSE脚本用于特定问题的研究验证，
并不自动进入正常`load_backend → infer`路径。

它们可帮助解释数值现象，但当前主目标不再要求先把旧runtime或所有中间量逐位闭合。
旧official/padded门及失败记录保留，不能用新scope事后改判。

仍必须遵守输入语义、有限输出、完整case覆盖、正确生命周期和source身份；
“不过度追精确”不等于允许漏输入、重复积分、旧cache污染或任意放宽预算。

<a id="autotune"></a>
## 12. autotune在执行后端之上做什么

先看[`vla_autotune.md`](vla_autotune.md)，再读这些代码：

| 模块 | 作用 |
|---|---|
| [`schema.py`](../flash_rt/vla/autotune/schema.py) | Candidate与结果身份的结构 |
| [`workloads.py`](../flash_rt/vla/autotune/workloads.py) | NPZ字段如何明确映射成请求 |
| [`worker.py`](../flash_rt/vla/autotune/worker.py) | 一个候选加载、prepare、warmup、测量、保存raw与cleanup |
| [`batch.py`](../flash_rt/vla/autotune/batch.py) | 同一allocation内用独立子进程串行执行各点 |
| [`quality.py`](../flash_rt/vla/autotune/quality.py) | 精确核对评价scope／通道／阈值契约 |
| [`search.py`](../flash_rt/vla/autotune/search.py) | 枚举候选，在已通过质量的结果里选median最快点 |
| [`controller.py`](../flash_rt/vla/autotune/controller.py) | 初始化、提交、等待、入账与汇总 |
| [`ledger.py`](../flash_rt/vla/autotune/ledger.py) | 可持久化的实验状态与身份 |

### 12.1 一次campaign的最小故事

```text
冻结source、workload、质量scope与候选
  → reference
  → candidate A
  → candidate B
  → reference bookend
  → 获取完整结果与raw
  → controller核对身份、预算与漂移
  → 选择合格点，或者不给winner
```

独立子进程减少不同候选之间的模型／graph驻留相互污染。
bookend检查的是比较过程中参考延迟和输出是否仍可信；当前controller在参考median漂移超过10%等情况下不给winner。

### 12.2 三个“成功”不要混淆

```text
Slurm完成        ≠ worker测量成功
worker保存测量   ≠ 质量代理通过
质量代理通过    ≠ 机器人任务成功
```

某个候选失败不能被当成0ms，也不能被选中。
campaign尚未完成或缺乏可信reference/bookend时，不能挑一个看起来最乐观的值。

### 12.3 为什么不能随便复用另一次结果

v2结果绑定campaign instance、完整Candidate、实际job、source和workload。
同名candidate不表示同一次执行；同scope名字但用了更松的阈值，也不能冒充当前预算下的passed。

这不是额外的数值精确要求，而是防止拿错证据。

<a id="walkthrough"></a>
## 13. 沿源码实际走一遍

### 路线A：第一次读，先只看公共调用

1. [`BackendConfig.__post_init__`](../flash_rt/vla/config.py)：画出合法／非法配置。
2. [`describe_backend`](../flash_rt/vla/registry.py)：确认它只描述，不加载。
3. [`load_backend`](../flash_rt/vla/registry.py)：找到四模型分发位置。
4. [`VLABackend._prepare/_invoke/infer`](../flash_rt/vla/backend.py)：追prompt、state、noise和actions。
5. [`ActionDecoder`](../flash_rt/vla/processing.py)：看公开输出怎样从raw得到。

**这一遍的产出：**能画出一次请求的调用链，能解释哪个阶段开始接触GPU。

### 路线B：选Hy理解模型数据流

1. [`HyVLAFlowFrontend.__init__`](../flash_rt/frontends/torch/hyvla_probeflow.py)：支持哪些选项，禁用哪些量化开关。
2. [`HyVLATorchFrontendThor.infer/predict_actions`](../flash_rt/frontends/torch/hyvla_thor.py)：公开输入与输出shape。
3. `_preprocess_images → _vit_merge → _assemble_prefix`：图像到视觉／语言prefix。
4. `_make_att_2d/_apply_segment_mask`：mask不只是shape，它决定语义。
5. [`_prefix_body/_velocity_body/_sample_latent`](../flash_rt/frontends/torch/hyvla_probeflow.py)：一次prefill与多次评估怎样拆开。
6. [`pipe.evaluate_velocity`](../flash_rt/models/hyvla/pipeline_thor.py)：动作Expert真正计算了什么。

**这一遍的产出：**分清机器人state与动作latent，知道哪些张量本请求固定、哪些每步变化。

### 路线C：再理解共享采样

1. 先读NumPy[`sample_probeflow`](../flash_rt/sampling/probeflow.py)。
2. 手工走一遍2步分支和密集分支，数evaluate次数。
3. 再读Torch[`sample_probeflow_torch_`](../flash_rt/sampling/torch_probeflow.py)，标出负号、clone和原地更新。
4. 最后看Pi两个ProbeFlow pipeline怎样给采样器提供native buffer view。

**这一遍的产出：**能解释为什么不能把已积分decoder当作raw velocity，以及为什么NFE不总等于steps。

### 路线D：最后再向下读原生算子

1. 从[`FlashRTLinear.forward`](../flash_rt/models/fastwam/native_linear.py)开始。
2. 纸上写出M/N/K和权重布局。
3. 追到C++ `bf16_nt_bias`，区分plan选择、workspace和真正matmul。
4. 再回到graph和worker，问这个调用在请求内重复多少次、测量包含什么。

**这一遍的产出：**能把一个kernel放回完整请求，而不是只读到某个“快”的函数就结束。

<a id="exercises"></a>
## 14. 不需要GPU的练习

前面的配置描述和动作解码也是CPU练习。下面再给两个采样练习。
它们只是数值小例子，不加载任何真实模型。

### 14.1 常量速度：为什么max_steps=10仍只执行2次

```python
# CPU_DEMO: constant_velocity
import numpy as np
from flash_rt.sampling import ProbeFlowConfig, sample_probeflow

noise = np.array([2., 3.])
def velocity(x, t):
    return np.ones_like(x)  # 返回v_theta；NumPy参考内部会转换成进度速度

actions, metadata = sample_probeflow(
    noise, velocity, ProbeFlowConfig(epsilon=0.008, max_steps=10),
)
assert np.allclose(actions, [1., 2.])
assert np.array_equal(noise, [2., 3.])
assert metadata["scheduled_steps"] == metadata["nfe"] == 2
print(actions, metadata["timesteps"])
```

两次速度方向完全一致，所以路由到2步。
配置max_steps只是上限，不要求每次都走到上限。

### 14.2 改变速度方向：密集6步为什么用了7次评估

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

请在源码中标出：哪次是lookahead，为什么密集分支重启，哪份速度被复用。

### 14.3 用已有测试当“可运行的接口说明”

先看这些小范围测试，不必一开始运行整仓GPU测试：

| 测试文件 | 适合学习的行为 |
|---|---|
| [`test_vla_backend_config.py`](../tests/test_vla_backend_config.py) | 配置／描述和不加载模型的边界 |
| [`test_vla_backend_observations.py`](../tests/test_vla_backend_observations.py) | prompt／state／noise、prepare和closed实例 |
| [`test_vla_backend_resources.py`](../tests/test_vla_backend_resources.py) | graph所有权与清理 |
| [`test_vla_action_processor.py`](../tests/test_vla_action_processor.py) | ACTION模式不是随意选统计值 |
| [`test_hyvla_probeflow.py`](../tests/test_hyvla_probeflow.py) | Hy raw-field／积分／graph连接 |
| [`test_fastwam_adapter.py`](../tests/test_fastwam_adapter.py) | first-frame条件与实际NFE |
| [`test_shifted_probeflow.py`](../tests/test_shifted_probeflow.py) | logical clock与physical sigma |
| [`test_hyvla_tradeoff_objective_example.py`](../tests/test_hyvla_tradeoff_objective_example.py) | 新public20质量预算及旧门隔离 |

CPU边界fixture能说明接口和控制流，**不能证明真实CUDA kernel、checkpoint或机器人任务质量**。
某些旧文件头的“未执行”注释是开发当时的描述；当前执行证据应看对应验证记录，而不是从注释推断。

<a id="faq"></a>
## 15. 常见误解与排查路线

### 15.1 十个需要立即纠正的误解

| 误解 | 正确理解 |
|---|---|
| `describe_backend`能跑，所以模型已支持 | 它不加载模型，描述也不完成所有运行检查 |
| 用Torch frontend就是Torch reference backend | 框架、执行backend与独立参考实现不是同一个维度 |
| `flashrt`表示每个算子全原生 | 必须看实际执行覆盖；FastWAM明确是部分替换 |
| `prepare`只是设置标记 | 它可能进行真实计算和graph准备 |
| 步数减半，延迟必然减半 | 固定条件／prefix成本不随NFE等比例减少 |
| Graph replay后没有Linear调用计数，就没有GEMM | host计数与重放的GPU逻辑工作不同 |
| 同指针同shape就可复用旧条件 | 内容可能已被原地更新，必须按请求刷新 |
| `epsilon`就是最终action容差 | 一个是速度路由启发式，一个是最终输出质量预算 |
| 统一`actions`可直接发给机器人 | 动作域、解码、单位和安全协议仍需核对 |
| 质量代理过门就是任务成功 | 固定输入数值代理不是闭环机器人评价 |

### 15.2 遇到问题先分层

```text
配置创建失败
  → 看config合法组合
描述正常、加载失败
  → 看registry选项／真实硬件／依赖／checkpoint
加载正常、首请求失败或很慢
  → 看prompt／预处理／prepare／capture／内存
后续请求像在用旧输入
  → 看state-in-prompt、条件copy、graph buffer刷新和cache key
动作偏差异常
  → 先核对noise／动作域／processor／时间与积分符号，再看局部数值
某点更快但不能入账
  → 看scope／预算／身份／raw覆盖／cleanup／bookend
```

不要从最后一个失败的CUDA函数名直接猜模型数学有错，也不要用放宽精度预算掩盖输入／执行错误。

### 15.3 自测：读完后应能回答什么

1. `BackendConfig`、`load_backend`、`prepare`哪一步开始接触真实模型和GPU？
2. 为什么π0.5相同prompt、不同state可能要重新处理prompt？
3. Hy的147个视觉token为什么不等于完整prefix长度？
4. raw velocity、进度速度和最终actions的区别是什么？
5. ProbeFlow选择6步时，为什么NFE可能是7，时间表bank长度还可能更大？
6. FastWAM的模型时间乘1000，为何积分步长不也乘1000？
7. KV cache、graph和GEMM plan cache分别缓存什么？
8. 为什么某个GEMM原型更快，不能直接推出完整模型更快？
9. 为什么Hy可以不追求中间EXACT，但仍必须检查mask、noise、NFE和buffer生命周期？
10. 哪些CPU例子能帮助理解接口，哪些结论一定要真实GPU或闭环任务才能给出？

能回答这些问题后，再深入某个attention／GEMM实现，就能知道自己正在优化整条数据流中的哪一段。

## 本篇文档的核验边界

- 文中的4个`CPU_DEMO`示例及描述CLI已在本次文档核验中实际执行，不加载真实模型、权重或GPU。
- GPU接口示意仅做语法和源码契约核对，**本次没有执行GPU推理或新性能实验**。
- 相关既有CPU接口测试为218通过、1跳过；跳过原因是本地未构建`flash_rt_kernels`，不是GPU路径已通过。
- 全仓另有既有收集／依赖问题，不能据这组接口测试宣称整个仓库绿色。
- 文档是源码学习材料；真实checkpoint、硬件、几何、模型质量和机器人安全仍要用各自的验证证据回答。

## 进一步阅读

- [统一后端API参考](vla_backend.md)：查参数和支持边界。
- [autotune机制](vla_autotune.md)：查实验身份、质量scope、测量与选择。
- [实际迭代记录](vla_autotune_experiments.md)：查哪些结论有实测证据及其范围。
- [中文阶段报告](reports/2026-11-09-vla-autotune-report.zh-CN.md)：理解历史结果与限制。
- [研究过程记录](reports/2026-11-12-vla-autotune-research-process.zh-CN.md)：理解假设、反例与目标修订，不必当作初学源码的阅读顺序。
