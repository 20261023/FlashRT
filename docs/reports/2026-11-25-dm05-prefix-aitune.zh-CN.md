# DM0.5 前缀层 AITune 算子观测

## 热点和部署

作业 `360006` 的 compiled + graph E10 参考，在独立的 97.9 ms profile 中花了 33.5 ms 做前缀预填充，约占 34%。该路径调用 34 次相同的已编译层函数。每层权重独立，除隐状态结果外，层函数还把 K 和 V 写入后续动作专家读取的缓存。比较对象是实际的 Inductor fullgraph 层。视觉 Sage 作业处理不同的调用点。

AITune 固定在 `1d3a4236`。隔离环境的安装回执记录了 91 个离线 wheel、Torch 2.11.0+cu128、Triton 3.6.0 与 TensorRT 10.15.1.29。已核对该版本的 inspect、wrap、tune、save、load 入口与 TensorRT 配置。既有记录证明了安装和 CPU 导入，尚缺在 DM0.5 实际前缀上的 TensorRT 性能与数值证据。本次直接使用本地 TensorRT 和固定权重；外部 LLM provider 凭据不参与这条路径。

## 作业与通过条件

A800 算子作业 `364193` 从 `360006` 的原基线快照 `e28a422b4a3dace7765d` 和四个原请求执行已编译前缀。采集器校验源码和工作负载摘要，在第 0 层真实调用处保存权重、输入隐状态、位置编码、任意前缀掩码、输出隐状态和写回的 K/V，并用三轮 CUDA event 观测 34 层。原始张量保留在私有目录。

同一卡上仅调优第 0 层与首个请求的 prefix length，batch 固定为 1。以 `OneBackendStrategy` 运行单一 TensorRT 后端，保留 BF16 与缓存写入结果，关闭内部 CUDA Graph、TF32 和量化，不允许调优失败后退回 eager。调优前以采集的实际输出验证独立模块方程；成功后保存并回读 .ait，对同长度请求分别复算隐状态及 K/V 的最坏 L2 和最低 cosine。

局部门槛为三个输出均满足 relative L2 ≤ 0.05、cosine ≥ 0.999，30 次 CUDA event 观测中的候选中位数较两次 Inductor 对照均至少降低 10%，首尾漂移至多 10%。该门的范围仅含一个权重实例与一种长度。其他长度、其他层权重、图内回放和四请求动作双门仍需单独证据。

截至 2026-11-25 11:06 UTC，已回读 `364193` 的提交回执和 gpu_a800 作业身份，状态为 PENDING。当前保留 compiled + graph E10；模型级 v3 候选须等算子结果、可移植 artifact 及覆盖范围核实之后决定。
