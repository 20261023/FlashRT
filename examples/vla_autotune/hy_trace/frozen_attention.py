"""Allocated block0 norm/QKV/spatial-attention controls, not policy acceptance."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

_KIND = 'diagnostic_frozen_block0_operators_not_recorded_block0_or_action_acceptance'


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


def attention_arm(qkv, *, heads, group_size, flash_fn=None):
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
            output = flash_fn(q, k, v, softmax_scale=scale, causal=False)
        else:
            output = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
                v.permute(0, 2, 1, 3), dropout_p=0., is_causal=False, scale=scale).permute(0, 2, 1, 3)
        outputs.append(output.reshape(group.shape[0], group.shape[1], -1))
    return torch.cat(outputs, 0)


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
    def keep(name, value):
        layouts[name] = {'shape': list(value.shape), 'live_stride': list(value.stride()), 'dtype': str(value.dtype)}
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
            expected = keep('attention_flash_per_camera', attention_arm(qkv, heads=16, group_size=6, flash_fn=flash_attn_func))
            flash_batch = keep('attention_flash_combined', attention_arm(qkv, heads=16, group_size=18, flash_fn=flash_attn_func))
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
            layouts[name] = {'shape': list(value.shape), 'live_stride': list(value.stride()), 'dtype': str(value.dtype)}
        sa = keep('attention_sdpa_per_camera', attention_arm(qkv, heads=16, group_size=6))
        sb = keep('attention_sdpa_combined', attention_arm(qkv, heads=16, group_size=18))
        compare('sdpa_per_camera_vs_frozen_flash_same_qkv', sa, expected)
        compare('sdpa_combined_vs_frozen_flash_same_qkv', sb, expected)
        compare('sdpa_combined_vs_sdpa_per_camera_same_qkv', sb, sa)
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
        'caveat': 'Reconstructed block0 input, not a recorded block0 boundary. Native attention consumes frozen official QKV. Within-runtime arms isolate grouping/backend; no full-policy parity or performance claim.'}
    (args.output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'engine': args.engine, 'metrics': metrics}), flush=True)


if __name__ == '__main__':
    main()
