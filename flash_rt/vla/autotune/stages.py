"""Pipeline stage map (游) and diagnostic time attribution.

Stage ids name where time goes inside one VLA request. MODEL_STAGES binds
them to source seams that exist in this tree; PROFILE_KEYS binds the
diagnostic CUDA-event interval keys written by ``profiling.profile_request``
to the stages each interval covers. ``attribute`` reads a worker result and
reports interval time per stage group plus the unattributed residual.

Interval time is a diagnostic, never a formal latency sample: GPU event
intervals can include stream-idle gaps, and the residual (wall minus
intervals) is reported exactly as computed, negative values included.
Stdlib only; importing this module loads no model.
"""

from __future__ import annotations

import argparse
import ast
from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

from ..model_spec import get_spec, list_models


# --------------------------------------------------------------------------
# Bottleneck playbook rows, verbatim from the earlier design
# (docs/bottleneck-playbook.md §1 and §5). Keys are the short strings that
# StageInfo.playbook references.

BOUND_ROWS = {
    '§1 计算吞吐': {
        'evidence': '关键算子持续执行、相关计算管线接近当前形状可达上限、内存/供给并非主要等待',
        'misread': '“FLOPs 多”不等于已经吃满 Tensor Core',
        'search_space': 'GEMM 后端、tile、精度、算子融合、减少计算量'},
    '§1 内存带宽/容量': {
        'evidence': '实际传输量、带宽、缓存、分配/换入换出与关键路径相关',
        'misread': '显存占用高不等于带宽饱和；空闲显存多也不等于没有带宽瓶颈',
        'search_space': '权重/KV 精度、数据布局、访存/缓存、批量、驻留与 offload'},
    '§1 launch/依赖延迟': {
        'evidence': '很多短 kernel，CPU launch/同步或串行依赖之间存在关键路径空洞',
        'misread': 'GPU-util 低不能直接推出 Python 慢，也可能是等待输入/通信',
        'search_space': 'compile、CUDA Graph、融合、批量、减少 host 往返'},
    '§1 CPU/输入/服务排队': {
        'evidence': 'GPU 之前/之间缺乏工作，CPU 栈、队列年龄或 I/O 与延迟对应',
        'misread': 'GPU kernel 都快仍然可能有很慢的请求响应',
        'search_space': '预处理、线程/进程、序列化、异步流水、调度、队列'},
}

SYMPTOM_ROWS = {
    'TTFT 很高、生成后较快': {
        'evidence': 'queue/processor/vision/prefill 各阶段边界；prompt 与视觉 token 数',
        'hypothesis': '排队或输入处理，视觉编码，长 prefix，首次 compile',
        'next_observation': '先读现有请求时间戳、processor shapes、编译日志',
        'candidates': '分别调队列、输入预算、prefix cache、prefill backend/分块',
        'invalid_when': 'TTFT 主要在网络或队列时，换 attention kernel 收益小；prefix 不共享则 cache 失效'},
    'decode 每 token 慢': {
        'evidence': '有效 token batch、上下文、weight/KV 字节、短 kernel 间隙',
        'hypothesis': '权重带宽、KV attention、小 GEMM、launch、通信中的一种或组合',
        'next_observation': '读取现有 token/step 日志和形状；核对 dtype 与真实后端',
        'candidates': 'batching、量化、KV 优化、融合/graph、speculation、并行度',
        'invalid_when': 'batch 已大或通信占优时，权重量化未必主导；草稿成本高时 speculation 净负收益'},
    '图片/帧数增加导致很慢': {
        'evidence': '图片解码/resize/vision、视觉 token、language prefix 分别增长',
        'hypothesis': '输入 I/O、视觉塔、语言侧序列膨胀均可能',
        'next_observation': '读取 processor 配置及实际 grid/token，核对多相机路径',
        'candidates': 'resize/tile/帧采样、encoder batching/cache、视觉 token 剪枝、prefill',
        'invalid_when': '细小物体/快速运动依赖被删信息；无有效缓存复用；token 数降而固定开销占优'},
    'denoising 总耗时高': {
        'evidence': '实际网络调用数、各 step 时间、CFG 路径、VAE 与 offload',
        'hypothesis': '重复 transformer/UNet、条件分支、拷贝、scheduler',
        'next_observation': '代码核对 timestep 循环、分支、cache 与 offload 配置',
        'candidates': 'compile、融合、低精度、attention、少步采样、缓存/外推/提前停止',
        'invalid_when': 'VAE/加载占主导；step 间特征不稳定；误差放大；检查器开销超过被省成本'},
    'step 少了但端到端几乎没变': {
        'evidence': 'prefix/预处理/网络/VAE 占比，剩余成本和跳步判定成本',
        'hypothesis': '优化非主导段，或节省被判定/同步抵消',
        'next_observation': '从既有区间求关键路径；核对是否仍执行被“跳过”的工作',
        'candidates': '把搜索转向主导段，减少判定/同步，调整 prefix 或执行周期',
        'invalid_when': '不能仅凭 denoiser 调用数下降就选 winner'},
    'VLA expert 很小却不快': {
        'evidence': '一次 chunk 内多个短 kernel、host/device 标量交互、重复 mask/投影',
        'hypothesis': 'launch/控制流、重复构造、低占用小 GEMM',
        'next_observation': '先审 sample loop、dtype/copy、graph break，区分 eager/compiled',
        'candidates': 'expert 编译/图执行、融合、预计算不变量、静态形状、后端',
        'invalid_when': '动态分支阻碍图捕获；编译成本无法摊销；固定 horizon 与任务要求冲突'},
    'Probe/跳步方案无收益': {
        'evidence': '实际 expert 调用数、probe/check 成本、prefix 占比',
        'hypothesis': 'check 与主路径重复计算，节省不在关键路径',
        'next_observation': '读候选执行记录和源码调用路径；标出强制 full step',
        'candidates': '降低 check 频率、共享中间量、仅在适用阶段跳步',
        'invalid_when': '高动态/接触阶段需要频繁恢复；缓存/外推误差累积；prefix 本就占主导'},
    'model_ms 好看但机器人仍卡': {
        'evidence': '完整 client RTT、pack/unpack、CPU transforms、拷贝、action queue',
        'hypothesis': '计时边界漏项、网络串行、控制线程等待',
        'next_observation': '核对 timer 放置；读取完整请求起止与 action 来源 ID',
        'candidates': '减少转换/序列化、驻留输入、流水化、降低拷贝、队列策略',
        'invalid_when': '模型本身占绝大多数时改 CPU 无显著总收益；流水造成动作更旧'},
    'GPU 时间线存在长空洞': {
        'evidence': 'CPU 栈/线程、CUDA API、copy、输入到达、collective/event 等待',
        'hypothesis': 'CPU 供给、I/O、同步、跨进程或另一个 stream 的依赖',
        'next_observation': '先关联已有 CPU/GPU 时间线，不以 util 单点归因',
        'candidates': '消除重复同步、输入预取、线程/进程调度、异步 copy/流水',
        'invalid_when': '空洞由正常等待新观测造成；overlap 引入争用或不满足数据依赖'},
    '小 tensor 的 .cpu/.item/NumPy 很贵': {
        'evidence': '该调用前有大量未完成 GPU 工作，API 等待区间',
        'hypothesis': '“拷贝耗时”包含前序计算完成等待',
        'next_observation': '读源码中隐式同步点及前驱 kernel；关联时间线',
        'candidates': '留在 device、合并读回、降低日志/分支跨设备频率',
        'invalid_when': '只是把等待移到别处，不减少关键路径；控制端必须及时读取动作'},
    'warmup 后偶发大尖峰': {
        'evidence': 'shape/guards、compile cache miss、allocator、CPU 抢占、I/O',
        'hypothesis': '重编译、缓存失效、页错误、队列抖动',
        'next_observation': '读既有 TORCH_LOGS、shape 分布与服务日志',
        'candidates': 'shape bucket、稳定图边界/缓存、减少临时分配、隔离输入工作',
        'invalid_when': '只报 p50 掩盖 tail；bucket padding 增加工作；冷启动目标不允许高编译成本'},
}

PLAYBOOK = {**BOUND_ROWS, **SYMPTOM_ROWS}


# --------------------------------------------------------------------------
# Tool ids referenced as stage defaults. The registry that defines them lives
# in flash_rt/vla/autotune/tools/; this module only names them.

REFERENCED_TOOL_IDS = frozenset({
    'sampler.euler_steps', 'sampler.probeflow', 'launch.cuda_graph',
    'gemm.fastwam_linear_backend', 'prefix.vision_pool', 'prefix.state_prompt_mode',
    'precision.hy_numerical_profile', 'quant.fp8', 'quant.nvfp4', 'quant.int8_pi05',
    'fusion.hy_rope_qknorm_kvwrite', 'cache.fastwam_text_crossattn_kv',
    'kernel.forgekernel', 'kernel.kernelagent', 'engine.apxinf',
    'profile.stage_intervals', 'profile.pi05_dataflow', 'profile.ncu_kernel',
    'analysis.lever_pricing', 'baseline.official_impl', 'compile.aitune',
    'compile.inductor',
})

STAGE_KINDS = ('compute', 'host', 'count', 'kernel', 'precision')
UNATTRIBUTED = 'unattributed'


@dataclass(frozen=True)
class StageInfo:
    stage_id: str
    title: str
    kind: str
    description: str
    playbook: tuple = ()
    default_tools: tuple = ()

    def __post_init__(self):
        if self.kind not in STAGE_KINDS:
            raise ValueError(f'Stage {self.stage_id!r} kind must be one of {STAGE_KINDS}')
        unknown = [row for row in self.playbook if row not in PLAYBOOK]
        if unknown:
            raise ValueError(f'Stage {self.stage_id!r} references unknown playbook rows {unknown}')
        tools = [tool for tool in self.default_tools if tool not in REFERENCED_TOOL_IDS]
        if tools:
            raise ValueError(f'Stage {self.stage_id!r} references unknown tool ids {tools}')

    @property
    def id(self):
        return self.stage_id

    def to_dict(self, *, rows=False):
        value = {'id': self.stage_id, 'title': self.title, 'kind': self.kind,
                 'description': self.description, 'playbook': list(self.playbook),
                 'default_tools': list(self.default_tools)}
        if rows:
            value['playbook_rows'] = {row: dict(PLAYBOOK[row]) for row in self.playbook}
        return value


@dataclass(frozen=True)
class ModelStage:
    stage_id: str
    seams: tuple = ()
    profiling_keys: tuple = ()
    notes: tuple = ()
    kind: str | None = None
    title: str | None = None

    def to_dict(self):
        return {'stage_id': self.stage_id, 'seams': list(self.seams),
                'profiling_keys': list(self.profiling_keys), 'notes': list(self.notes)}


_STAGE_LIST = (
    StageInfo('host.input', 'Host input', 'host',
              'Observation preprocessing (resize/normalize/stack), tokenization and prompt '
              'embedding, state/noise staging and H2D copies before the model runs.',
              ('图片/帧数增加导致很慢', 'model_ms 好看但机器人仍卡', 'GPU 时间线存在长空洞',
               '§1 CPU/输入/服务排队'),
              ('profile.stage_intervals', 'baseline.official_impl')),
    StageInfo('vision.encode', 'Vision encoder', 'compute',
              'Vision tower: SigLIP/ViT for pi0/pi05/Hy; first-frame VAE encode for FastWAM.',
              ('图片/帧数增加导致很慢', 'TTFT 很高、生成后较快'),
              ('profile.stage_intervals', 'profile.ncu_kernel', 'kernel.forgekernel',
               'kernel.kernelagent', 'compile.aitune')),
    StageInfo('vision.project', 'Vision projection', 'compute',
              'Projector / merger between vision tower and prefix (pi0/pi05 multi-modal '
              'projector and optional token pooling; Hy merger PROJ1/pool/PROJ2).',
              ('图片/帧数增加导致很慢', 'TTFT 很高、生成后较快'),
              ('prefix.vision_pool', 'profile.stage_intervals')),
    StageInfo('prefix.prefill', 'Prefix prefill', 'compute',
              'VLM / prefix transformer producing the per-layer KV cache that every action '
              'evaluation reads (FastWAM: 30-layer first-frame video-expert prefill).',
              ('TTFT 很高、生成后较快', '图片/帧数增加导致很慢', 'step 少了但端到端几乎没变'),
              ('prefix.vision_pool', 'prefix.state_prompt_mode', 'fusion.hy_rope_qknorm_kvwrite',
               'profile.pi05_dataflow', 'engine.apxinf', 'kernel.forgekernel', 'compile.aitune',
                 'compile.inductor')),
    StageInfo('denoise.velocity', 'Action-expert evaluation', 'compute',
              'ONE action-expert (velocity field) evaluation; total cost is per-call time x NFE.',
              ('VLA expert 很小却不快', 'denoising 总耗时高'),
              ('launch.cuda_graph', 'gemm.fastwam_linear_backend', 'cache.fastwam_text_crossattn_kv',
               'fusion.hy_rope_qknorm_kvwrite', 'profile.pi05_dataflow', 'kernel.forgekernel',
                 'kernel.kernelagent', 'compile.aitune', 'compile.inductor')),
    StageInfo('denoise.schedule', 'Denoising schedule (NFE)', 'count',
              'Number of action-expert calls chosen by the sampler (Euler N steps or ProbeFlow '
              'probe + adaptive schedule). A count, not a time interval.',
              ('step 少了但端到端几乎没变', 'Probe/跳步方案无收益', 'denoising 总耗时高'),
              ('sampler.euler_steps', 'sampler.probeflow')),
    StageInfo('denoise.routing', 'Routing and integration', 'host',
              'Host routing and integration between evaluations: ProbeFlow cosine .item() '
              'decision, Euler x += dt * v updates, finite checks.',
              ('小 tensor 的 .cpu/.item/NumPy 很贵', 'Probe/跳步方案无收益'),
              ('sampler.probeflow', 'profile.stage_intervals')),
    StageInfo('host.launch', 'Kernel launch / dispatch', 'host',
              'CPU kernel-launch and dispatch gaps in eager execution; cross-cutting, removed '
              'by CUDA graph capture.',
              ('GPU 时间线存在长空洞', 'VLA expert 很小却不快', '小 tensor 的 .cpu/.item/NumPy 很贵',
               'warmup 后偶发大尖峰', '§1 launch/依赖延迟'),
              ('launch.cuda_graph', 'gemm.fastwam_linear_backend', 'prefix.state_prompt_mode',
                'compile.inductor')),
    StageInfo('host.output', 'Host output', 'host',
              'Device-to-host copy of the action chunk and CPU action decoding '
              '(unnormalize, slicing, checkpoint action processor).',
              ('model_ms 好看但机器人仍卡', '小 tensor 的 .cpu/.item/NumPy 很贵'),
              ('profile.stage_intervals',)),
    StageInfo('kernel.gemm', 'GEMM kernels', 'kernel',
              'Linear / matmul kernels across all compute stages (cross-cutting kernel class).',
              ('§1 计算吞吐', '§1 内存带宽/容量', 'decode 每 token 慢'),
              ('gemm.fastwam_linear_backend', 'profile.ncu_kernel', 'analysis.lever_pricing',
               'kernel.forgekernel', 'kernel.kernelagent')),
    StageInfo('kernel.attention', 'Attention kernels', 'kernel',
              'Attention kernels (prefix self-attention, expert attention over prefix KV, '
              'cross-attention).',
              ('§1 计算吞吐', '§1 内存带宽/容量', 'decode 每 token 慢'),
              ('profile.ncu_kernel', 'kernel.forgekernel', 'kernel.kernelagent')),
    StageInfo('kernel.norm_elementwise', 'Norm / elementwise kernels', 'kernel',
              'RMSNorm/LayerNorm, RoPE, QK-norm, bias/residual/activation and KV-write kernels.',
              ('§1 内存带宽/容量', '§1 launch/依赖延迟', 'VLA expert 很小却不快'),
              ('fusion.hy_rope_qknorm_kvwrite', 'profile.ncu_kernel', 'kernel.forgekernel',
               'kernel.kernelagent')),
    StageInfo('kernel.conv', 'Convolution kernels', 'kernel',
              'Convolution kernels (FastWAM VAE encoder; patch embeddings run as im2col + GEMM).',
              ('§1 计算吞吐', '§1 内存带宽/容量'),
              ('profile.ncu_kernel', 'kernel.forgekernel', 'compile.aitune')),
    StageInfo('precision.weights', 'Weight precision', 'precision',
              'Quantization / weight and activation precision (cross-cutting; exactness changes).',
              ('§1 内存带宽/容量', '§1 计算吞吐', 'decode 每 token 慢'),
              ('precision.hy_numerical_profile', 'quant.fp8', 'quant.nvfp4', 'quant.int8_pi05')),
)

STAGES = {item.stage_id: item for item in _STAGE_LIST}
STAGE_IDS = tuple(STAGES)


# --------------------------------------------------------------------------
# Interval key -> stages covered, per model (keys from profiling.profile_request).

def profile_keys_for(model, *, spec_dir=None):
    return {key: tuple(covered)
            for key, covered in get_spec(model, spec_dir=spec_dir)['profiling']['keys'].items()}


class _DeclaredTable(Mapping):
    def __init__(self, value):
        self.value = value

    def __getitem__(self, model):
        try:
            return self.value(model)
        except ValueError as error:
            raise KeyError(model) from error

    def __iter__(self):
        return iter(list_models())

    def __len__(self):
        return len(list_models())


MODEL_STAGES = _DeclaredTable(lambda model: stages_for(model))
PROFILE_KEYS = _DeclaredTable(profile_keys_for)
MODELS = list_models()


# --------------------------------------------------------------------------
# Model stage maps. Seams are 'relative/path.py:function' or ':Class.method'.
def stage(stage_id):
    if stage_id not in STAGES:
        raise ValueError(f'Unknown stage id {stage_id!r}; known: {list(STAGE_IDS)}')
    return STAGES[stage_id]


def stages_for(model, *, spec_dir=None):
    try:
        spec = get_spec(model, spec_dir=spec_dir)
    except ValueError as error:
        raise ValueError(f'Unknown model {model!r}: {error}') from error
    items = []
    for item in spec['stages']:
        known = STAGES.get(item['stage_id'])
        kind = item.get('kind', known.kind if known else None)
        if kind not in STAGE_KINDS:
            raise ValueError(f'Model {model!r} has unknown stage {item["stage_id"]!r}; '
                             f'declare kind from {STAGE_KINDS}')
        if known and kind != known.kind:
            raise ValueError(f'Stage {item["stage_id"]!r} kind contradicts the shared stage definition')
        keys = tuple(key for key, covered in spec['profiling']['keys'].items()
                     if item['stage_id'] in covered)
        items.append(ModelStage(
            item['stage_id'], tuple(item['seams']), keys, tuple(item['notes']),
            kind=kind, title=item.get('title', known.title if known else item['stage_id'])))
    return tuple(items)


# --------------------------------------------------------------------------
# Seam verification (AST, no imports of model code).

def repo_root():
    return Path(__file__).resolve().parents[3]


def parse_seam(seam):
    if not isinstance(seam, str) or seam.count(':') != 1:
        raise ValueError(f'Seam must be "relative/path.py:name", got {seam!r}')
    path, name = seam.split(':')
    if not path.endswith('.py') or path.startswith('/') or '..' in Path(path).parts:
        raise ValueError(f'Seam path must be a relative .py path: {seam!r}')
    parts = name.split('.')
    if not 1 <= len(parts) <= 2 or not all(part.isidentifier() for part in parts):
        raise ValueError(f'Seam name must be "function" or "Class.method": {seam!r}')
    return path, tuple(parts)


def seam_defined(seam, root=None):
    path, parts = parse_seam(seam)
    source = Path(root or repo_root()) / path
    if not source.is_file():
        return False
    tree = ast.parse(source.read_text(), filename=str(source))
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    top = {node.name: node for node in tree.body if isinstance(node, definitions)}
    if len(parts) == 1:
        return parts[0] in top
    owner = top.get(parts[0])
    if not isinstance(owner, ast.ClassDef):
        return False
    return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == parts[1]
               for node in owner.body)


def all_seams(*, spec_dir=None):
    return sorted({seam for model in list_models(spec_dir=spec_dir)
                   for item in stages_for(model, spec_dir=spec_dir) for seam in item.seams})


def missing_seams(root=None, *, spec_dir=None):
    return [seam for seam in all_seams(spec_dir=spec_dir) if not seam_defined(seam, root)]


# --------------------------------------------------------------------------
# Attribution.

def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


def _record_model(record, model, *, spec_dir=None):
    candidate = record.get('candidate')
    found = None
    if candidate is not None:
        backend = candidate.get('backend') if isinstance(candidate, dict) else None
        if not isinstance(backend, dict) or not isinstance(backend.get('model'), str):
            raise ValueError('record.candidate.backend.model must be a string')
        found = backend['model']
    if model is not None and found is not None and model != found:
        raise ValueError(f'model {model!r} does not match record model {found!r}')
    model = model or found
    if model is None:
        raise ValueError('Model unknown: record has no candidate.backend.model and none was given')
    try:
        return get_spec(model, spec_dir=spec_dir)['model']
    except ValueError as error:
        raise ValueError(f'No stage map for model {model!r}: {error}') from error


def _record_sampler(record, metadata):
    candidate = record.get('candidate')
    if isinstance(candidate, dict) and isinstance(candidate.get('backend'), dict):
        sampler = candidate['backend'].get('sampler', 'euler')
    elif isinstance(metadata, dict):
        sampler = metadata.get('sampler')
    else:
        sampler = None
    return sampler


def _empty(model, notes, wall_ms=None, nfe=None):
    return {'model': model, 'wall_ms': wall_ms, 'groups': [], 'per_call_ms': {}, 'nfe': nfe,
            'unattributed_ms': None, 'coverage': 'none', 'residual_stages': [], 'notes': notes,
            'operator_probes': [], 'coverage_scope': 'declared_compute_stages',
            'attributed_fraction': None, 'nonadditive_diagnostics': [],
            'coverage_evidence': None}


def attribute(record, *, model=None, spec_dir=None):
    """Attribute the profiled request's wall time to stage groups.

    ``record`` is a worker result dict (``record['stage_metrics']`` as written
    by profiling.profile_request). Coverage is 'none' without stage metrics,
    with a profiling failure, or with no intervals. Nothing is clamped.
    """
    if not isinstance(record, dict):
        raise ValueError('record must be a worker result dict')
    candidate = record.get('candidate')
    if spec_dir is None and isinstance(candidate, dict):
        backend = candidate.get('backend')
        if isinstance(backend, dict):
            spec_dir = backend.get('spec_dir')
    model = _record_model(record, model, spec_dir=spec_dir)
    spec = get_spec(model, spec_dir=spec_dir)
    model_stages = stages_for(model, spec_dir=spec_dir)
    declared_stage_ids = {item.stage_id for item in model_stages}
    notes = []
    if 'profiling_failure' in record:
        return _empty(model, [f'profiling failed: {record["profiling_failure"]}'])
    metrics = record.get('stage_metrics')
    if metrics is None:
        return _empty(model, ['record has no stage_metrics (profiling disabled, failed or not measured)'])
    if not isinstance(metrics, dict):
        raise ValueError('stage_metrics must be a dict')
    if metrics.get('measurement_kind') != 'diagnostic_cuda_event_intervals':
        raise ValueError('stage_metrics.measurement_kind must be diagnostic_cuda_event_intervals')
    if metrics.get('separate_from_latency_samples') is False:
        raise ValueError('stage_metrics must be separate_from_latency_samples')
    if 'notes' in metrics:
        if not isinstance(metrics['notes'], list) or not all(
                isinstance(note, str) for note in metrics['notes']):
            raise ValueError('stage_metrics.notes must be a list of strings')
        notes.extend(metrics['notes'])
    wall_ms = _finite_number(metrics.get('wall_ms'), 'stage_metrics.wall_ms', positive=True)
    intervals = metrics.get('interval_ms')
    if not isinstance(intervals, dict):
        raise ValueError('stage_metrics.interval_ms must be a dict of interval lists')
    sums = metrics.get('interval_sum_ms')
    if sums is not None and (not isinstance(sums, dict) or set(sums) != set(intervals)):
        raise ValueError('stage_metrics.interval_sum_ms keys must equal interval_ms keys')

    sampler_metadata = metrics.get('sampler_metadata')
    nfe = None
    if sampler_metadata is None:
        notes.append('stage_metrics has no sampler_metadata: nfe unknown, no cross-check')
    elif not isinstance(sampler_metadata, dict):
        raise ValueError('stage_metrics.sampler_metadata must be a dict or null')
    else:
        nfe = sampler_metadata.get('nfe')
        if nfe is None and spec['head']['type'] != 'flow_matching':
            notes.append('Non-flow head: velocity NFE is not applicable')
        elif isinstance(nfe, bool) or not isinstance(nfe, int) or nfe < 1:
            raise ValueError('stage_metrics.sampler_metadata.nfe must be a positive integer')
    sampler = _record_sampler(record, sampler_metadata)

    mapping = profile_keys_for(model, spec_dir=spec_dir)
    provenance = metrics.get('interval_provenance', {})
    if not isinstance(provenance, dict) or (provenance and set(provenance) != set(intervals)):
        raise ValueError('stage_metrics.interval_provenance keys must equal interval_ms keys')
    probe_keys, diagnostic_keys = set(), set()
    for key, detail in provenance.items():
        if not isinstance(detail, dict):
            raise ValueError(f'interval_provenance[{key!r}] must be an object')
        covered = detail.get('stages')
        if not isinstance(covered, list) or not covered or any(
                stage_id not in STAGES and stage_id not in declared_stage_ids for stage_id in covered):
            raise ValueError(f'interval_provenance[{key!r}] has an unknown stage or empty stages')
        if detail.get('device') not in ('cpu', 'cuda') or detail.get('clock') not in (
                'cuda_event', 'perf_counter_ns'):
            raise ValueError(f'interval_provenance[{key!r}] requires a known device and clock')
        if (detail['device'] == 'cpu') != (detail['clock'] == 'perf_counter_ns'):
            raise ValueError(f'interval_provenance[{key!r}] device and clock disagree')
        if 'additive' in detail and type(detail['additive']) is not bool:
            raise ValueError(f'interval_provenance[{key!r}].additive must be bool')
        if detail.get('scope') == 'operator_probe':
            if detail.get('additive') is not False:
                raise ValueError('Operator probes must be explicitly non-additive')
            probe_keys.add(key)
        if key in mapping and tuple(covered) != mapping[key]:
            raise ValueError(f'interval_provenance[{key!r}] stages differ from the model declaration')
        if detail.get('additive') is False and key not in probe_keys:
            diagnostic_keys.add(key)
    unknown = sorted(set(intervals) - set(mapping) - probe_keys)
    if unknown:
        raise ValueError(f'Interval keys {unknown} have no stage mapping for model {model!r}; '
                         f'known: {list(mapping)}')
    groups, probes, diagnostics = [], [], []
    excluded = probe_keys | diagnostic_keys
    ordered = ([key for key in intervals if key in mapping] if provenance else
               [key for key in mapping if key in intervals])
    ordered += [key for key in intervals if key not in mapping]
    for key in ordered:
        values = intervals[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f'stage_metrics.interval_ms[{key!r}] must be a nonempty list')
        values = [_finite_number(value, f'stage_metrics.interval_ms[{key!r}]') for value in values]
        total = math.fsum(values)
        if sums is not None:
            stored = _finite_number(sums[key], f'stage_metrics.interval_sum_ms[{key!r}]')
            if abs(stored - total) > 1e-6 * max(1.0, abs(total)):
                raise ValueError(f'interval_sum_ms[{key!r}]={stored} differs from sum(interval_ms)={total}')
        group = {'stages': list(provenance[key]['stages'] if key in probe_keys else mapping[key]),
                 'interval_key': key, 'ms': total, 'calls': len(values)}
        if key in provenance:
            group['provenance'] = dict(provenance[key])
        if key in probe_keys:
            probes.append(group)
        elif key in diagnostic_keys:
            diagnostics.append(group)
        else:
            groups.append(group)

    # A legacy parent and its split children cannot both be additive. Equal
    # stage sets may describe sequential subregions (e.g. condition preparation
    # and prefill), so only strict containment is rejected here.
    for index, parent in enumerate(groups):
        for child in groups[index + 1:]:
            left, right = set(parent['stages']), set(child['stages'])
            if left < right or right < left:
                raise ValueError(f'Overlapping parent/child intervals would double-count: '
                                 f'{parent["interval_key"]!r}, {child["interval_key"]!r}')

    spans = metrics.get('interval_spans_ms')
    if spans is not None:
        if not isinstance(spans, dict) or set(spans) != set(intervals) or not provenance:
            raise ValueError('interval_spans_ms requires provenance and all interval_ms keys')
        by_clock = {}
        for key, ranges in spans.items():
            if not isinstance(ranges, list) or len(ranges) != len(intervals[key]):
                raise ValueError(f'interval_spans_ms[{key!r}] must match its interval count')
            for index, pair in enumerate(ranges):
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError(f'interval_spans_ms[{key!r}] must contain [start, stop] pairs')
                start, stop = [_finite_number(value, f'interval_spans_ms[{key!r}]') for value in pair]
                duration = stop - start
                # CUDA elapsed_time returns float32 milliseconds; differences
                # of epoch-relative timestamps need microsecond tolerance.
                tolerance = max(0.002, 1e-6 * max(start, stop, 1.0))
                if duration < 0 or abs(duration - intervals[key][index]) > tolerance:
                    raise ValueError(f'interval_spans_ms[{key!r}] differs from its interval duration')
                if key not in excluded and duration:
                    by_clock.setdefault(provenance[key]['clock'], []).append((start, stop, key))
        for ranges in by_clock.values():
            ordered_ranges = sorted(ranges)
            for left, right in zip(ordered_ranges, ordered_ranges[1:]):
                tolerance = max(0.002, 1e-6 * max(left[1], right[1], 1.0))
                if right[0] < left[1] - tolerance:
                    raise ValueError(f'Overlapping request intervals would double-count: '
                                     f'{left[2]!r}, {right[2]!r}')

    if not groups:
        notes.append('no additive request intervals recorded; non-additive diagnostics/probes do not establish coverage')
        empty = _empty(model, notes, wall_ms=wall_ms, nfe=nfe)
        empty['operator_probes'] = probes
        empty['nonadditive_diagnostics'] = diagnostics
        return empty

    per_call = {}
    by_key = {group['interval_key']: group for group in groups}
    if 'velocity' in by_key:
        velocity = by_key['velocity']
        per_call['denoise.velocity'] = velocity['ms'] / velocity['calls']
        if nfe is not None and velocity['calls'] != nfe:
            notes.append(f'velocity interval count {velocity["calls"]} != sampler nfe {nfe}')
    elif 'full' in by_key:
        notes.append("'full' interval covers vision, prefix and all denoising steps "
                     '(Euler + CUDA graph): per-call velocity time is not separable')
    for key in ('full', 'prefix', 'condition_vae_video_prefill'):
        if key in by_key and by_key[key]['calls'] != 1:
            notes.append(f'interval {key!r} recorded {by_key[key]["calls"]} calls, expected 1 per request')

    expected = metrics.get('expected_interval_keys', spec['profiling']['expected_keys'].get(sampler))
    if expected is not None and (
            not isinstance(expected, (list, tuple))
            or any(not isinstance(key, str) or key not in mapping for key in expected)):
        raise ValueError('expected_interval_keys must contain declared interval keys')
    missing = [key for key in expected or () if key not in intervals]
    if expected is None:
        notes.append(f'sampler {sampler!r} unknown: expected interval keys not checked')
    elif missing:
        notes.append(f'expected interval keys missing for {model}/{sampler}: {missing}')

    attributed = math.fsum(group['ms'] for group in groups)
    residual = wall_ms - attributed
    if residual < 0:
        notes.append(f'interval sum {attributed:.6g} ms exceeds wall {wall_ms:.6g} ms by '
                     f'{-residual:.6g} ms; reported unclamped (event intervals can include '
                     'stream-idle or overlapping time)')
    covered = {stage_id for group in groups for stage_id in group['stages']}
    compute = [item.stage_id for item in model_stages if item.kind == 'compute']
    residual_stages = [item.stage_id for item in model_stages
                       if item.stage_id not in covered and item.kind in ('compute', 'host')]
    coverage = 'full' if all(stage_id in covered for stage_id in compute) and not missing else 'partial'
    if metrics.get('diagnostic_graph_recapture') and (
            metrics.get('capture_parity') or {}).get('status') != 'bit_exact':
        coverage = 'partial'
        notes.append('Diagnostic recapture lacks bit-exact public output parity evidence')
    notes.append('coverage describes declared compute stages; CPU IO and residual time are reported separately')
    if excluded:
        notes.append(f'non-additive diagnostics/probes excluded from wall coverage: {sorted(excluded)}')
    notes.append(f'unattributed_ms contains: {residual_stages}')
    return {'model': model, 'wall_ms': wall_ms, 'groups': groups, 'per_call_ms': per_call,
            'nfe': nfe, 'unattributed_ms': residual, 'coverage': coverage,
            'residual_stages': residual_stages, 'notes': notes, 'operator_probes': probes,
             'coverage_scope': 'declared_compute_stages',
            'attributed_fraction': attributed / wall_ms, 'nonadditive_diagnostics': diagnostics,
            'coverage_evidence': {
                'additive_keys': [group['interval_key'] for group in groups],
                'excluded_keys': sorted(excluded), 'expected_keys': expected,
                'missing_keys': missing, 'attributed_ms': attributed, 'wall_ms': wall_ms,
                'residual_ms': residual, 'fraction': attributed / wall_ms,
                'within_wall': residual >= 0,
                'span_validation': 'same_clock_nonoverlap_checked' if spans is not None else 'unavailable_legacy_record',
                'clock_domains': sorted({detail['clock'] for detail in provenance.values()}),
            }}


def amdahl_ceiling(share, local_speedup):
    """End-to-end speedup ceiling when a fraction ``share`` speeds up locally."""
    for value, name in ((share, 'share'), (local_speedup, 'local_speedup')):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if not 0 <= share <= 1:
        raise ValueError('share must lie in [0, 1]')
    if local_speedup <= 0:
        raise ValueError('local_speedup must be positive')
    return 1.0 / ((1.0 - share) + share / local_speedup)


def rank(attribution):
    """Stage groups and the residual by time, largest first."""
    if not isinstance(attribution, dict) or 'groups' not in attribution:
        raise ValueError('rank expects the dict returned by attribute()')
    wall = attribution.get('wall_ms')
    if wall is None:
        return []
    wall = _finite_number(wall, 'wall_ms', positive=True)
    items = [{'stages': list(group['stages']), 'ms': group['ms'], 'share': group['ms'] / wall}
             for group in attribution['groups']]
    residual = attribution.get('unattributed_ms')
    if residual is None:
        residual = wall - math.fsum(item['ms'] for item in items)
    items.append({'stages': [UNATTRIBUTED], 'ms': residual, 'share': residual / wall})
    return sorted(items, key=lambda item: -item['ms'])


# --------------------------------------------------------------------------
# CLI.

def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m flash_rt.vla.autotune.stages',
                                     description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    listing = commands.add_parser('list', help='List stage ids, or one model stage map')
    listing.add_argument('--model')
    listing.add_argument('--spec-dir')
    show = commands.add_parser('show', help='Show one stage with its playbook rows')
    show.add_argument('stage_id')
    attribution = commands.add_parser('attribute', help='Attribute a worker result.json')
    attribution.add_argument('result_json')
    attribution.add_argument('--model', help='Required only when the record has no candidate')
    attribution.add_argument('--spec-dir', help='Explicit extra model declaration directory')
    attribution.add_argument('--rank', action='store_true', help='Also print the ranked groups')
    seams = commands.add_parser('check-seams', help='Verify every seam exists in this tree')
    seams.add_argument('--spec-dir')
    args = parser.parse_args(argv)
    try:
        if args.command == 'list':
            if args.model is None:
                output = {'stages': [item.to_dict() for item in _STAGE_LIST]}
            else:
                items = stages_for(args.model, spec_dir=args.spec_dir)
                output = {'model': args.model,
                          'profile_keys': {key: list(value) for key, value in
                                           profile_keys_for(args.model, spec_dir=args.spec_dir).items()},
                          'stages': [{**item.to_dict(), 'title': item.title, 'kind': item.kind}
                                     for item in items]}
        elif args.command == 'show':
            output = stage(args.stage_id).to_dict(rows=True)
        elif args.command == 'attribute':
            try:
                record = json.loads(Path(args.result_json).read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(f'Cannot read result JSON {args.result_json}: {error}') from error
            output = attribute(record, model=args.model, spec_dir=args.spec_dir)
            if args.rank:
                output = {'attribution': output, 'rank': rank(output)}
        else:
            missing = missing_seams(spec_dir=args.spec_dir)
            output = {'seams': len(all_seams(spec_dir=args.spec_dir)), 'missing': missing}
            if missing:
                print(json.dumps(output, indent=2, ensure_ascii=False))
                print(f'error: {len(missing)} seam(s) missing', file=sys.stderr)
                return 1
    except ValueError as error:
        print(f'error: {error}', file=sys.stderr)
        return 2
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
