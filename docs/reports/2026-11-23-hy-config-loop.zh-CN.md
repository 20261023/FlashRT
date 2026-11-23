# Hy 首轮 A800 配置扫描

作业 `356212` 在一张 A800 上以 semantic BF16 比较七个固定输入。每个配置预热七次，每个输入正式计时五次，共 280 次。项目用同轮 Euler10 首测与复测评估计时漂移，在归一化 public20 动作上要求最坏 relative L2 ≤ 0.05 且最低 cosine ≥ 0.999。

| 配置 | 中位延迟（ms） | 项目精度门槛 |
|---|---:|---|
| Euler10 + Graph 首测 | 120.50 | 基线 |
| Euler6 + Graph | 102.45 | 未通过 |
| Euler8 + Graph | 116.92 | 通过 |
| Euler9 + Graph | 118.67 | 通过 |
| ProbeFlow ε=0.002 + Graph | 124.05 | 通过 |
| ProbeFlow ε=0.008 + Graph | 111.99 | 未通过 |
| Euler10 关闭 Graph | 287.98 | 通过 |
| Euler10 + Graph 复测 | 121.23 | 通过 |

控制器选出 Euler8 + Graph。中位延迟较首测下降 2.97%，首尾漂移为 +0.60%。相对项目 Euler10，最坏 relative L2 为 0.032544，最低 cosine 为 0.999496。与作业 `326641` 保存的作者 BF16 Euler10 输出配对，七个输入中最坏 relative L2 为 0.035629，最低 cosine 为 0.999390，均出现在 `noise` 输入。精度结论覆盖本轮固定输入。
