# 四模型延迟对比：本项目已选方法与官方 Native

**状态：结果前协议。** 本轮尚未产生新的四模型延迟，下表“待实测”不能由旧实验值替换。

## 1. 术语与测量目标

- **官方 Native**只指作者原论文／原仓库的执行栈，**不是FlashRT**。
- 本项目FastWAM Torch graph虽使用Torch，也不是作者原始`infer_action`执行入口，必须另列。
- 优先引用作者论文／仓库的可核验数字；未找到发布值的Hy补跑作者实现。
- 不预设本项目更快，不把不同GPU、输入、步数或计时边界的数字直接相除成“加速比”。

## 2. 已核实选择与待执行点

| 模型 | 本轮固定方法 | 已有选择依据与限制 | 新A800延迟 |
|---|---|---|---|
| π0 | FlashRT FP16 Euler2，3view、chunk50、Orbax/JAX权重loader | `307979`保留Euler2；loader不是Native JAX执行；未证明Euler2等价默认Euler10质量 | 待实测 |
| π0.5 | FlashRT BF16 Euler8，3view、chunk50 | `308244`四请求数值代理范围内选择，不是泛化／机器人质量保证 | 待实测 |
| FastWAM | 本项目Torch raw-field graph Euler20，shift5、H32、2cam224×224、cached-T5 | `313779`扩大8案例拒绝Euler13/14/16，保留Euler20；不能用旧四案例Euler13冒充当前最优 | 待实测 |
| Hy | FlashRT semantic-BF16 Euler10＋graph，3cam×6帧，模型chunk40、raw32/public20 | 新public20权衡目标尚无实测搜索胜点；此项明确是稳定基线，不称全局最优 | 待实测 |

## 3. 官方已发布的数字

NR表示来源未报告，不能用当前默认配置补填历史条件。

| 模型／作者版本 | 官方 Native发布延迟 | GPU | 已知条件／范围 | 相对本轮A800 |
|---|---:|---|---|---|
| π0原论文 | **73 ms** on-board；86 ms off-board | RTX4090 | 3cam、H50、Euler10；统计量／dtype NR；off-board额外13ms网络 | 条件不同，仅作背景，不算加速比 |
| π0.5，作者RTC论文vanilla行 | **76 ms** | RTX4090 | BF16、H50、5步；5warmup后10次mean；仅GPU神经计算；不是RTC97ms | 硬件／步数／边界不同，仅作背景 |
| FastWAM论文 | **190 ms** | RTX5090D V2 32GB | H32、10步、T5计入；作者澄清无compile/graph，实机384×320而非本项目LIBERO224×448 | 硬件／输入／文本／步数不同，仅作背景 |
| FastWAM后续官方仓库compiled默认 | **110 ms**（此前190ms） | RTX4090 | 包含text/VAE；具体checkpoint／步数／统计量等NR | 不倒填条件，不算加速比 |
| 同一官方仓库compiled默认 | **210 ms**（此前470ms） | H20 | 同上，不能与论文190ms合并成一条测量 | 不算A800加速比 |
| Hy官方论文／仓库／模型卡 | **NR** | NR | 未找到完整单次推理ms；任务完成秒数／servo Hz不是推理延迟 | 本轮补跑官方实现 |

来源：

1. [π0原论文Appendix A-D／Table I](https://arxiv.org/html/2410.24164v4#A1.SS4)。
2. [作者RTC论文Appendix A.3／Table 1](https://arxiv.org/html/2506.07339v2#A1.SS3)。π0.5该行是vanilla，不是RTC、BID或网络端到端数字；精确checkpoint／私有runtime未证明等于公开OpenPI。
3. [FastWAM论文v2](https://arxiv.org/html/2603.16666v2#S4.SS1)、[作者对190ms的澄清](https://github.com/yuantianyuan01/FastWAM/issues/34#issuecomment-4250694653)。
4. [FastWAM固定提交README](https://github.com/yuantianyuan01/FastWAM/blob/7faa71108368fbb3b6885649f112af607427a2d4/README.md)。官方默认可以包含作者自己的compile优化，Native不等于一律eager。
5. [Hy论文v2](https://arxiv.org/html/2606.14409v2)、[官方源码af57](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/tree/af57e7507ec5964b52fdf6296741e553cbcd3288)。

## 4. 本轮固定计时协议

- 单A800、单allocation、串行独立进程，30分钟硬上限；每个固定配置reference＋bookend，不重新搜索。
- 本项目π0／π0.5／FastWAM／Hy分别保留1／4／8／7个既有case；每进程8次warmup，正式50／40／80／70次，均衡覆盖。
- 计时为host observation／固定noise到完整CPU动作chunk，包含`infer`触发的输入处理／上传／模型／输出；load、prepare、warmup单列。
- FastWAM明确使用预计算T5，不包含现场文本编码；与作者含T5的发布值不假称同口径。
- 共用worker若增加profile请求，其位于正式样本之外，不拿profile耗时作为正式延迟。
- 保留完整延迟样本、动作、case ID、实际NFE、source、GPU和环境身份；核验覆盖、有限性、cleanup与reference/bookend漂移。
- 质量门在本轮只检查固定配置的bookend/replay稳定性，不借reference角色宣称新的算法质量选优或官方质量等价。

## 5. Hy官方 Native自测协议

现有作者Torch2.7.1／FA2 2.7.4.post1环境及完整checkpoint保持只读。
从作者Git提交af57导出未修改的`hy_vla`源码，并核对导出字节和实际导入位置。

计时调用原作者`prepare_images`、`prepare_state`、`prepare_language`，然后调用原作者
`policy.model.sample_actions(..., noise=固定noise)`，最后返回FP32 CPU raw `[40,32]`。
它与本项目Hy的host uint8历史／state／prompt／noise到CPU raw chunk边界相对应。
这是计时与输入转换wrapper，**没有替换作者的神经／采样实现，没有构造FlashRT frontend／pipeline／扩展**。

不采用缓存action queue命中的`select_action/get_action`时间；不将仅算预处理后`sample_actions`的时间冒充完整请求。
`forward_evaluate`本身不接受固定noise，所以明确记录上述原API组合，而不谎称调用了该入口。
一个额外的未计时请求核对Euler10调用次数，观察器在warmup／正式计时前移除。
每次仍执行完整视觉／prefix／动作采样；raw32与public20区别保留，RoboTwin模型chunk40是20relative＋20absolute位置，并非40个不同未来时刻。

官方Hy先测reference，四个本项目配置随后执行，最后补官方Hy bookend。
一个模型／官方环境失败不阻断其余独立模型的测量；失败项保留NR与错误，整批不能冒称完整成功。

## 6. 本轮结果

等待真实调度、完整产物及独立数组／统计审计后填写。当前没有新的延迟、加速比或质量晋升结论。
