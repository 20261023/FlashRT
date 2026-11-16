# DM0.5：项目方法快于固定FlashRT与ApxInf

**本轮固定输入范围内目标已达成。** 只优化项目方法，没有优化、替换或重命名两条对照。
正式配对作业`335362`为COMPLETED 0:0、3分53秒；同一A800／8CPU allocation、同一
Slurm step0，串行独立进程。主会话已从全部保存数组独立复算结果、源码和noise身份。

## 1. 同卡完整public policy结果

固定B1、三张448×448原始PIL图像、当前state／prompt、host provided-noise、BF16、
Euler10、无历史、H50/internal32/public14。每case5warmup＋10formal，四case全保留。
计时包含CPU预处理／tokenization、H2D、完整模型、D2H及绝对FP32 CPU `[50,14]`
重建，排除文件解码／网络。不使用prepared-GPU、GPU-only或异步enqueue时间代替。

| 路径 | Median ms | Mean ms | P95 ms | 完整正式输出 |
|---|---:|---:|---:|---:|
| 原始FlashRT mixed，冻结实现未改动 | 206.3908680 | 206.5768056 | 207.56426970 | 40 |
| ApxInf，冻结实现未改动 | 151.1550295 | 151.29895675 | 152.51795445 | 40 |
| **我们的项目compiled方法** | **96.1658565** | **96.386913325** | **97.87547085** | **40** |
| ApxInf bookend，未改动 | 150.7088110 | 150.79157885 | 151.72859245 | 40 |

- 相对原始FlashRT，延迟下降**53.40595%**，延迟比**2.14620×**。
- 相对初始ApxInf，下降36.37932%；即使用同作业**更快的ApxInf bookend**，仍下降
  **36.19095%**，延迟比**1.56718×**。median和P95都优于两条对照。
- ApxInf首尾median漂移**−0.295206%**，首尾输出逐元素相同，优化优势远大于漂移。
- 历史334487的项目194.2449／FlashRT206.0562，以及329543的ApxInf149.3522 ms
  全部保留；新回测没有覆盖旧点。这里的原始FlashRT是本fork已接入的mixed DM基线，
  ApxInf是之前已实跑的独立原生移植，不冒称上游原有DM内置支持或论文发布值。

## 2. 没有通过改质量或计时口径取得优势

四条路径全部160份完整输出均有限、同case重复逐元素一致。主会话读取每条路径的
`noise_fp32.npy`，并从作者保存BF16 tensor storage独立还原，确认所有host noise数值相同。
权重、相机数量／尺寸、历史、BF16及E10未变，无量化／减步／排除案例。

| 对作者329331保存输出 | 最坏relative L2 | 最低cosine | 最大绝对差 |
|---|---:|---:|---:|
| 原始FlashRT | 0.0086970135 | 0.9999650775 | 0.0156019926 |
| ApxInf／bookend | 0.0093243955 | 0.9999608552 | 0.0156019926 |
| **项目compiled** | **0.0091867243** | **0.9999653347** | **0.0126748681** |

候选相对同卡FlashRT最坏L2／最低cos为0.0098358635／0.9999557181；相对ApxInf为
0.0121681631／0.9999260550，均满足原public14 **L2≤0.05且cos≥0.999**范围。
不称逐元素等价、质量更优、机器人闭环成功或通用release验收。

## 3. 方法：编译融合与数据流，而非再次宣称graph收益

原项目已经是whole-model graph；本次增加的是：

1. QKV与MLP gate/up在加载阶段打包，降低重复投影调用和中间布局操作。
2. 去除项目Torch路径不必要的attention Q/K/V/O槽位搬运，直接tensor SDPA。
3. vision／prefix／action三类可复用layer函数用Inductor编译，融合norm、RoPE、
   gate、激活与残差；权重和time modulation作为tensor参数，不做逐层／十步整数特化。
4. action使用真正KV4-head的FlashAttention GQA；带任意mask的prefix仍保留KV扩展，
   不谎称所有位置都消除了repeat。
5. `fullgraph=True,dynamic=False`，默认Inductor GEMM选择；关闭编译器内部CUDA graph，
   每个精确prefix预热后由已有frontend进行整模capture。无suppress-errors或静默eager fallback。

实际生成代码共**5个graph module**：vision1个、prefix882／876各1个、action KV932／926
各1个。生成调用显示vision／action为`aten._scaled_dot_product_flash_attention`，action
保持KV4 heads；masked prefix为`aten._scaled_dot_product_efficient_attention`。
主会话核对生成代码archive中的5份源摘要与记录一致，没有运行或公开私有生成代码。

### 分阶段观测的有限解释

独立事件诊断，仅prefix882、3次graph replay的median，不属于正式public policy样本：

| GPU interval | 原始FlashRT ms | 项目compiled ms |
|---|---:|---:|
| vision | 18.958336 | 19.048449 |
| prefix，不含vision | 51.077118 | 33.533953 |
| 十次velocity合计 | 123.767809 | 31.680512 |
| Euler／other | 0.122885 | 0.123904 |
| whole-model | 193.926147 | 84.382721 |

组合方法的主要实测改善位于velocity，prefix也改善，vision基本持平。没有做ablation，
因此不能把收益逐项分摊给融合、packing或GQA。事件区间含依赖／事件开销，不是GPU
kernel时长求和；诊断不包含完整CPU路径，也不能当作正式96.17 ms的替代数字。

## 4. 冷成本、身份和失败保留

- 候选load41.5804 s、FlashRT26.6475 s；候选preparation16.5874 s包含在load里，不能重复相加。
- 独立初始空Inductor／Triton cache，882／876的compile-plus-first-forward分别
  7.7712／3.8029 s，不称纯compiler计时。候选capture合计1.0147 s，FlashRT3.2663 s。
- Torch peak allocated16,289,728,000字节含setup／诊断pool；APX由Rust分配，Torch的
  0计数是未覆盖而不是零显存，不能据此比较显存。
- 首轮335329为FAILED1:0、3分52秒，私有诊断遗漏inference mode并触发相关guard问题；
  部分APX参考输出保留，但不是成功配对campaign。R2只修正私有诊断／事件和setup记录，
  运行时数学源码没有变化。恰好两个GPU作业，无第三次allocation或额外测试。
- 候选执行内容摘要`eb901a1d6cb35be4bb9d2ecad49c4011fd0794052a21e1a73853d173e87946d8`；
  主会话核对563份清单文件与当前源码一致。freeze HEAD是e9ede857，不包含候选未提交
  内容，所以以内容manifest而不是HEAD单独标识实跑。
- 原FlashRT内容摘要仍为`e16bb373…106b`，APX extension仍为`575b96f8…74d52`；
  正式结果中的加载身份及主会话预核对均匹配，未改比较对象、checkpoint、生产`.so`或环境。

## 5. 使用方式与范围

```python
from flash_rt.vla import BackendConfig, load_backend

config = BackendConfig(
    model="dm05", checkpoint="/local/DM05-base", backend="torch",
    hardware="cuda_sm80", precision="bf16", num_steps=10,
    options={"source_dir": "/local/pinned-opendm", "num_views": 3,
             "use_cuda_graph": True, "max_graph_profiles": 2,
             "project_optimization": "compiled"},
)
```

此flag默认仍为`none`；FlashRT backend拒绝`compiled`，防止对照被偷偷改变。
用法见[DM0.5说明](../dm05_usage.md)。按用户要求没有pytest、smoke、operator/parity
probes或额外评审；证据是固定四案例正式推理和已保存输出算术，不扩展成任意输入／
eager-parity／部署或机器人成功保证。旧配置仍可调用，原始测量点和失败记录均保留。
