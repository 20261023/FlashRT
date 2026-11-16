# ApxInf 外部研究实现与源码增量

这里保存本轮实际执行过的外部 ApxInf 模型候选的**源码增量与身份说明**，不向
FlashRT registry 增加后端，也不宣称公开 ApxInf 上游已经内置这些模型。

2026-11-16更新：DM0.5独立原生候选也已完成正式执行，job329543完整public policy
中位数149.3522 ms、40份输出在原四案例代理范围内。
[DM0.5源码增量](dm05-sm80/README.md)与FastWAM快照分别提供，不相互混合；
结果、provided-noise／作者RNG差异及较慢冷启动见下方大表。

DM0.5包中的23份运行时代码保持实测身份；12份后补文档／helper／许可副本明确为
未执行。它同样依赖私有Hy foundation，不是干净上游的自包含重建包。该外部ApxInf
源码包**不是另已实跑的FlashRT项目内DM0.5路径**，不得将两者的结果或身份混用；
项目内Torch／FlashRT mixed基线的入口与用法见[DM0.5说明](../../../docs/dm05_usage.md)。

## FastWAM SM80，实验性

源码包：[fastwam-sm80/](fastwam-sm80/README.md)；实际结果见
[五模型延迟大表](../../../docs/reports/2026-11-14-vla-latency-comparison.zh-CN.md)。

- A800、BF16、224×448 当前帧、cached-T5 128、proprio8、H32／A7、Euler20／shift5。
- 作业329256完成完整native graph与公共policy，各80份动作；公共policy中位数
  215.3437 ms。保存输出对作者compiled20参考的最坏relative L2为0.01161、最低
  cosine为0.999933，在原八案例数值代理范围内，不是闭环或部署验收。
- 当前比已选项目compiled约138 ms慢，不修改默认或已选方法。
- 公开源码增量SHA-256：
  `79baaac216e816fddd7f693c9a29b37d086886bb37408664909e724a7fc6b450`。

### 基础依赖限制

**本补丁不是干净上游`7f37642`的自包含重建包。** 它基于已归档、包含未提交改动的
私有Hy foundation；完整基础源码及复用的native库不在本目录。缺少该基础时，不应
直接套用到其他ApxInf版本，也不能将“提供了源码增量”理解为公开环境已经可独立复建。
基础和native库身份在`SOURCE_MANIFEST.json`、`BUILD_LICENSE_NOTES.md`中明确列出。

### 实测与后补材料的身份不同

补丁31个文件中，24个运行时代码文件与329256保存的实测源码字节相同；另外7个
文档／参数化helper是测量之后补写，**没有执行、重建或验证patch应用**。
原实测增量包SHA与公开补丁SHA不是同一身份，不能混称。

主会话独立核对了这24份源码摘要、交付文件摘要及实际日志中的驱动身份；仅进行了
机械披露过滤，没有把这些文件检查冒充代码、安全、数值或发布验收。按照本轮用户要求，
没有新增测试、smoke、额外评审或GPU作业。

目录只包含文本源码补丁、许可证说明和聚合统计；没有权重、环境、二进制、原始输入
或逐次动作。应用者必须保留ApxInf根许可证、FastWAM MIT及相应NOTICE；本目录不授予
权重再分发许可。研究policy明确保持`deployable=False`，不作为机器人动作部署接口。
