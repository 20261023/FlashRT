"""Allocated block0 norm/QKV/spatial-attention controls, not policy acceptance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

_KIND = 'diagnostic_frozen_block0_operators_not_recorded_block0_or_action_acceptance'
_SDPA_BACKENDS = ('FLASH_ATTENTION', 'EFFICIENT_ATTENTION', 'CUDNN_ATTENTION', 'MATH')
# Exact host-dispatch rejection strings in Torch2.7.1/2.11 attention.cpp and
# cuda/sdp_utils.cpp. Never reinterpret an arbitrary CUDA/cuDNN failure as this.
_NO_DISPATCH = frozenset((
    'No available kernel. Aborting execution.',
    'No viable backend for scaled_dot_product_attention was found.',
    'No viable backend for scaled_dot_product_attention was found. This is likely due to turning off both the math kernel and the fused kernels.',
))


def validate_reference_report(report, *, job_id, checkpoint, official_trace):
    checkpoint = Path(checkpoint).expanduser().resolve()
    stat = checkpoint.stat()
    if (report.get('kind') != _KIND or report.get('engine') != 'official'
            or report.get('job_id') != job_id or report.get('checkpoint') != str(checkpoint)
            or report.get('checkpoint_stat') != {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            or report.get('official_trace') != str(Path(official_trace).expanduser().resolve())):
        raise ValueError('Frozen QKV reference checkpoint/input provenance differs from this allocated control')


def split_qkv(qkv, *, heads):
    """Return BNHD views in the official flash_attn_func argument order."""
    if (qkv.ndim != 3 or type(heads) is not int or heads < 1
            or qkv.shape[-1] < 3*heads or qkv.shape[-1] % (3*heads)):
        raise ValueError('Invalid QKV/head geometry')
    batch, tokens, width = qkv.shape
    return qkv.reshape(batch, tokens, 3, heads, width//(3*heads)).unbind(2)


def attention_arm(qkv, *, heads, group_size, flash_fn=None, observer=None):
    """Hold QKV values fixed; vary only camera grouping or attention backend."""
    import torch
    import torch.nn.functional as F
    split_qkv(qkv, heads=heads)
    if type(group_size) is not int or group_size < 1 or qkv.shape[0] % group_size:
        raise ValueError('Camera group must divide the frozen QKV batch')
    outputs = []
    for group in qkv.split(group_size, 0):
        q, k, v = split_qkv(group, heads=heads)
        scale = q.shape[-1]**-.5
        if flash_fn is not None:
            if observer is not None:
                observer('flash', q, k, v)
            output = flash_fn(q, k, v, softmax_scale=scale, causal=False)
        else:
            q, k, v = (value.permute(0, 2, 1, 3) for value in (q, k, v))
            if observer is not None:
                observer('sdpa', q, k, v)
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.,
                is_causal=False, scale=scale).permute(0, 2, 1, 3)
        outputs.append(output.reshape(group.shape[0], group.shape[1], -1))
    return torch.cat(outputs, 0)


def sdpa_flags():
    import torch
    cuda = torch.backends.cuda
    return {'flash': cuda.flash_sdp_enabled(), 'efficient': cuda.mem_efficient_sdp_enabled(),
            'cudnn': cuda.cudnn_sdp_enabled(), 'math': cuda.math_sdp_enabled()}


def _layout(value):
    return {'shape': list(value.shape), 'live_stride': list(value.stride()),
            'storage_offset': value.storage_offset(), 'dtype': str(value.dtype), 'device': str(value.device)}


def _observed_inputs(kind, q, k, v):
    return {'api': kind, **{name: _layout(value) for name, value in (('q', q), ('k', k), ('v', v))}}


def _sdpa_preflight(qkv, *, heads, group_size, name):
    import torch
    import warnings
    if name == 'MATH':
        return {'supported': True, 'checker': 'math_no_fused_preflight', 'warnings': []}
    cuda = torch.backends.cuda
    checker = {'FLASH_ATTENTION': cuda.can_use_flash_attention,
               'EFFICIENT_ATTENTION': cuda.can_use_efficient_attention,
               'CUDNN_ATTENTION': cuda.can_use_cudnn_attention}[name]
    tensors = [value.permute(0, 2, 1, 3) for value in split_qkv(qkv[:group_size], heads=heads)]
    params = cuda.SDPAParams(*tensors, None, 0., False, False)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always')
        supported = bool(checker(params, debug=True))
    return {'supported': supported, 'checker': checker.__name__,
            'warnings': [str(item.message) for item in messages]}


def probe_sdpa_arm(qkv, *, heads, group_size, name, preflight=None, synchronize=None):
    """One-hot diagnostic selection, never implicit fallback or hidden GPU faults."""
    import torch
    from torch.nn.attention import sdpa_kernel, SDPBackend
    if name not in _SDPA_BACKENDS:
        raise ValueError('Unknown SDPA backend')
    split_qkv(qkv, heads=heads)
    if type(group_size) is not int or group_size < 1 or qkv.shape[0] % group_size:
        raise ValueError('Camera group must divide the frozen QKV batch')
    preflight = preflight or _sdpa_preflight
    synchronize = synchronize or (lambda: torch.cuda.synchronize(qkv.device))
    with sdpa_kernel(getattr(SDPBackend, name)):
        observed = []
        metadata = {'backend': name, 'group_size': group_size, 'selected_flags': sdpa_flags(),
                    'observed_sdpa_inputs': observed}
        metadata['preflight'] = preflight(qkv, heads=heads, group_size=group_size, name=name)
        if not metadata['preflight']['supported']:
            return None, {**metadata, 'status': 'unsupported'}
        try:
            output = attention_arm(qkv, heads=heads, group_size=group_size,
                observer=lambda kind, q, k, v: observed.append(_observed_inputs(kind, q, k, v)))
        except RuntimeError as error:
            message = str(error).strip()
            if message not in _NO_DISPATCH:
                raise
            return None, {**metadata, 'status': 'unsupported', 'dispatch_error': message}
        # Synchronization faults are execution errors even if their wording
        # resembles a host dispatch rejection. They must fail the process.
        synchronize()
        return output, {**metadata, 'status': 'executed'}


def summarize_dispatch_events(events, *, region_name=None):
    from collections import Counter
    operators, kernels, annotations = Counter(), Counter(), Counter()
    for event in events:
        is_cuda = str(event.device_type).endswith('CUDA')
        # Kineto can mirror a record_function region on CUDA. A device-type
        # label alone does not make that annotation an executed GPU kernel.
        if getattr(event, 'is_user_annotation', False) or event.name == region_name:
            if is_cuda:
                annotations[event.name] += 1
            continue
        if is_cuda:
            kernels[event.name] += 1
        elif event.name.startswith('aten::_scaled_dot_product'):
            operators[event.name] += 1
    return {'sdpa_operator_counts': dict(operators), 'cuda_kernel_names_in_region': dict(kernels),
            'ignored_cuda_annotation_names': dict(annotations),
            'cuda_activity_observed': bool(kernels),
            'kernel_identity': 'observed_region_names_not_one_to_one_attribution' if kernels else 'unverified_no_cuda_activity',
            'semantics': 'Names/counts only; no profiler timing or zero-GPU-work claim'}


def _profile_auto_sdpa(qkv, *, heads, group_size):
    import torch
    region = 'frozen_auto_sdpa_no_timing_acceptance'
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                          torch.profiler.ProfilerActivity.CUDA],
                                record_shapes=False, with_stack=False, profile_memory=False) as profile:
        with torch.profiler.record_function(region):
            attention_arm(qkv, heads=heads, group_size=group_size)
            torch.cuda.synchronize(qkv.device)
    return summarize_dispatch_events(profile.events(), region_name=region)


def _sibling(name):
    spec = importlib.util.spec_from_file_location('hy_attention_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('official', 'native'), required=True)
    parser.add_argument('--official-trace', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Required native input: same-job official frozen-attention output')
    parser.add_argument('--probe-sdpa-backends', action='store_true',
                        help='Extra frozen one-hot backend controls and name-only auto dispatch profiling; no product changes')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    common = _sibling('diagnose')
    common.require_step()  # before Torch, checkpoint reads or output creation
    if args.engine == 'native' and args.reference is None:
        raise ValueError('Native operator process requires the official frozen-QKV reference')
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file
    if tuple(torch.cuda.get_device_capability()) != (8, 0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('Frozen attention controls require the recorded A800 target')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    report = json.loads((args.official_trace/'result.json').read_text())
    if report.get('engine') != 'official' or report.get('oracle', {}).get('exact') is not True:
        raise ValueError('Frozen input requires an exactly reproduced accepted official trace')
    args.output.mkdir(parents=True, exist_ok=False)
    rebuild_view = _sibling('frozen_ops').rebuild_view
    outputs, metrics, layouts = {}, {}, {}
    flash_backend = None
    backend_probes, auto_dispatch = {}, None
    attention_inputs = {}
    def input_observer(name):
        rows = attention_inputs.setdefault(name, [])
        return lambda kind, q, k, v: rows.append(_observed_inputs(kind, q, k, v))
    def keep(name, value):
        layouts[name] = _layout(value)
        outputs[name] = value.detach().cpu().contiguous().clone()
        return value
    def compare(name, a, b):
        row = common.difference(common._array(a), common._array(b))
        row['unequal_elements'] = int((a.detach().cpu() != b.detach().cpu()).sum())
        metrics[name] = row
    with torch.inference_mode(), safe_open(str(args.official_trace/'trace.safetensors'), framework='pt', device='cpu') as frozen:
        position = frozen.get_tensor('vision.position_embedding').cuda()
        patches = [frozen.get_tensor(f'vision.patch_embed.{i}').cuda() for i in range(3)]
        if tuple(position.shape) != (1, 196, 1152) or any(tuple(value.shape) != (6, 196, 1152) for value in patches):
            raise ValueError('Frozen patch/position geometry differs from block0 contract')
        # Reconstruct the NCHW-conv -> flatten/transpose output layout. These
        # strides are source-derived controls, NOT a newly observed block0 trace.
        camera_x = [rebuild_view(value, (225792, 1, 196)) + position for value in patches]
        x = keep('block0_input', torch.cat(camera_x, 0))
        with safe_open(str(args.checkpoint), framework='pt', device='cpu') as checkpoint:
            prefix = 'model.dual_tower.vlm.model.visual.vision_tower.blocks.0.'
            tensors = {name: checkpoint.get_tensor(prefix + name).to('cuda', torch.bfloat16)
                       for name in ('norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias')}
        nw, nb, qw, qb = (tensors[name] for name in ('norm1.weight', 'norm1.bias', 'attn.qkv.weight', 'attn.qkv.bias'))
        if nw.shape != (1152,) or nb.shape != (1152,) or qw.shape != (3456, 1152) or qb.shape != (3456,):
            raise ValueError('Block0 checkpoint norm/QKV geometry differs from the observed contract')
        hn = keep('norm1_per_camera', torch.cat([F.layer_norm(value, (1152,), nw, nb, 1e-6) for value in camera_x]))
        # Match the native combined pre-norm layout, without changing values.
        combined_x = rebuild_view(x, (225792, 1, 196))
        hb = keep('norm1_combined', F.layer_norm(combined_x, (1152,), nw, nb, 1e-6))
        compare('norm1_combined_vs_per_camera', hb, hn)
        qa = keep('qkv_per_camera', torch.cat([F.linear(value, qw, qb) for value in hn.split(6, 0)]))
        qb_output = keep('qkv_combined_frozen_norm', F.linear(hn, qw, qb))
        compare('qkv_combined_vs_per_camera_same_norm_values', qb_output, qa)
        if args.engine == 'official':
            import flash_attn
            flash_attn_func = flash_attn.flash_attn_func
            flash_backend = {'version': getattr(flash_attn, '__version__', 'unknown'),
                             'module': str(Path(flash_attn.__file__).resolve())}
            qkv = qa
            expected = keep('attention_flash_per_camera', attention_arm(qkv, heads=16, group_size=6,
                flash_fn=flash_attn_func, observer=input_observer('flash_group6')))
            flash_batch = keep('attention_flash_combined', attention_arm(qkv, heads=16, group_size=18,
                flash_fn=flash_attn_func, observer=input_observer('flash_group18')))
            compare('flash_combined_vs_flash_per_camera_same_qkv', flash_batch, expected)
        else:
            prior = json.loads((args.reference/'result.json').read_text())
            validate_reference_report(prior, job_id=os.environ['SLURM_JOB_ID'],
                                      checkpoint=args.checkpoint, official_trace=args.official_trace)
            flash_backend = prior.get('flash_backend')
            with safe_open(str(args.reference/'operator_outputs.safetensors'), framework='pt', device='cpu') as reference:
                if not torch.equal(x.cpu(), reference.get_tensor('block0_input')):
                    raise ValueError('Frozen QKV reference reconstructed input values differ')
                qkv = reference.get_tensor('qkv_per_camera').cuda()
                expected = reference.get_tensor('attention_flash_per_camera').cuda()
            if qkv.shape != (18, 196, 3456) or qkv.dtype != torch.bfloat16:
                raise ValueError('Saved official QKV geometry/dtype differs from admitted reference')
            compare('native_recomputed_qkv_vs_frozen_official_per_camera', qa, qkv)
        q, k, v = split_qkv(qkv, heads=16)
        for name, value in (('frozen_q', q), ('frozen_k', k), ('frozen_v', v)):
            layouts[name] = _layout(value)
        initial_sdpa_flags = sdpa_flags()
        sa = keep('attention_sdpa_per_camera', attention_arm(qkv, heads=16, group_size=6,
            observer=input_observer('auto_sdpa_group6')))
        sb = keep('attention_sdpa_combined', attention_arm(qkv, heads=16, group_size=18,
            observer=input_observer('auto_sdpa_group18')))
        compare('sdpa_per_camera_vs_frozen_flash_same_qkv', sa, expected)
        compare('sdpa_combined_vs_frozen_flash_same_qkv', sb, expected)
        compare('sdpa_combined_vs_sdpa_per_camera_same_qkv', sb, sa)
        if args.probe_sdpa_backends:
            # Observe AUTO before any forced selector context. Profiling is
            # separate from all numerical arms and exports no duration table.
            auto_dispatch = _profile_auto_sdpa(qkv, heads=16, group_size=6)
            for backend_name in _SDPA_BACKENDS:
                arms = {}
                for group_size in (6, 18):
                    name = f'attention_sdpa_{backend_name.lower()}_group{group_size}'
                    value, detail = probe_sdpa_arm(qkv, heads=16, group_size=group_size, name=backend_name)
                    backend_probes[name] = detail
                    if value is None:
                        continue
                    arms[group_size] = keep(name, value)
                    compare(name + '_vs_frozen_flash', value, expected)
                    compare(name + '_vs_auto', value, sa if group_size == 6 else sb)
                    if torch.equal(value, expected):
                        repeated, confirmation = probe_sdpa_arm(qkv, heads=16, group_size=group_size, name=backend_name)
                        if repeated is None or not torch.equal(value, repeated):
                            raise RuntimeError('Exact FA2 candidate did not reproduce on its confirmation arm')
                        keep(name + '_repeat', repeated)
                        detail['exact_confirmation'] = confirmation
                if 6 in arms and 18 in arms:
                    compare(f'{backend_name.lower()}_group18_vs_group6', arms[18], arms[6])
            if sdpa_flags() != initial_sdpa_flags:
                raise RuntimeError('Diagnostic SDPA selectors did not restore initial backend flags')
    torch.cuda.synchronize()
    save_file(outputs, str(args.output/'operator_outputs.safetensors'))
    result = {'kind': _KIND,
        'engine': args.engine, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'job_id': os.environ['SLURM_JOB_ID'],
        'official_trace': str(args.official_trace.resolve()), 'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_stat': {
            'bytes': args.checkpoint.stat().st_size, 'mtime_ns': args.checkpoint.stat().st_mtime_ns},
        'metrics': metrics, 'layouts': layouts, 'model_constructed': False, 'performance_measurement': False,
        'flash_backend': flash_backend,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'math_sdp_fp16_bf16_reduction_allowed': torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        'cudnn_version': torch.backends.cudnn.version(), 'initial_sdpa_flags': initial_sdpa_flags,
        'sdpa_backend_probes': backend_probes, 'auto_sdpa_dispatch': auto_dispatch,
        'observed_attention_inputs': attention_inputs,
        'caveat': 'Reconstructed block0 input, not a recorded block0 boundary. Native attention consumes frozen official QKV. Within-runtime arms isolate grouping/backend; no full-policy parity or performance claim.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'engine': args.engine, 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
