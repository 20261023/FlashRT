"""Allocated frozen early-vision operators; no model or action-gate changes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path


def rebuild_view(values, strides):
    """Reconstruct a validated non-overlapping live layout with identical values."""
    import torch
    shape = tuple(values.shape)
    strides = tuple(strides)
    if (len(strides) != len(shape) or any(type(x) is not int or x < 0 for x in strides)
            or any(size < 1 for size in shape)):
        raise ValueError('Malformed or negative stride/shape')
    span = 1
    for stride, size in sorted((stride, size) for stride, size in zip(strides, shape) if size > 1):
        if stride < span:
            raise ValueError('Overlapping stride layout cannot be used as a frozen control')
        span += stride * (size - 1)
    result = torch.empty_strided(shape, strides, dtype=values.dtype, device=values.device)
    result.copy_(values)
    return result


def projection_inputs(values, official_strides, native_metadata):
    """Hold frozen values fixed while reconstructing the two observed layouts."""
    import torch
    if (not values or len(values) != len(official_strides)
            or any(value.ndim != 4 or value.shape[0] != 1 for value in values)):
        raise ValueError('Invalid official merger input geometry')
    combined = torch.cat(values, 0).contiguous()
    batch, tokens, width = native_metadata['shape']
    side = math.isqrt(tokens)
    if side*side != tokens or (batch, side, side, width) != tuple(combined.shape):
        raise ValueError('Native and official merger input geometry does not align')
    bs, ns, ds = native_metadata['live_stride']
    official = [rebuild_view(value, stride) for value, stride in zip(values, official_strides)]
    native = rebuild_view(combined, (bs, side*ns, ns, ds))
    return official, native, combined


def resize_position(position, output_hw, *, promote):
    """Compare only interpolation dtype, preserving learned token/channel order."""
    import torch.nn.functional as F
    if position.ndim != 3 or position.shape[0] != 1:
        raise ValueError('Unexpected learned position geometry')
    grid = math.isqrt(position.shape[1])
    if grid < 1 or grid*grid != position.shape[1]:
        raise ValueError('Unexpected learned position geometry')
    width = position.shape[2]
    pe = position[0].t().contiguous().view(1, width, grid, grid)
    if promote:
        pe = pe.float()
    result = F.interpolate(pe, output_hw, mode='bilinear', align_corners=False).to(position.dtype)
    return result.view(width, math.prod(output_hw)).t().contiguous()[None]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-trace', type=Path, required=True)
    parser.add_argument('--native-report', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Readonly model.safetensors; selected three tensors only')
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location('hy_frozen_operator_common', Path(__file__).with_name('diagnose.py'))
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    common.require_step()  # before Torch, CUDA or checkpoint reads
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from safetensors.torch import save_file
    if tuple(torch.cuda.get_device_capability()) != (8, 0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('This revision-bound operator experiment requires the recorded A800 target')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    metadata = json.loads((args.official_trace / 'result.json').read_text())
    if metadata.get('engine') != 'official' or metadata.get('oracle', {}).get('exact') is not True:
        raise ValueError('Frozen operators require an exactly reproduced official reference')
    native = json.loads(args.native_report.read_text())
    live = native['tensor_metadata']['vision.merger_input']
    outputs, results = {}, {}
    def metric(a, b):
        row = common.difference(common._array(a), common._array(b))
        row['unequal_elements'] = int((a.detach().cpu() != b.detach().cpu()).sum())
        return row
    def observe(name, fn):
        # CPU operator names reveal mm/bmm/addmm/add lowering. These timings
        # are intentionally not exported as an uninstrumented benchmark.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                     record_shapes=False, with_stack=False, profile_memory=False) as profiler:
            value = fn()
        torch.cuda.synchronize()
        outputs[name] = value.detach().cpu().contiguous()
        wanted = {'aten::linear', 'aten::matmul', 'aten::mm', 'aten::bmm', 'aten::addmm',
                  'aten::add_', 'aten::add', 'aten::clone', 'aten::contiguous', 'aten::reshape'}
        return value, {event.key: event.count for event in profiler.key_averages() if event.key in wanted}
    with torch.inference_mode(), safe_open(str(args.official_trace / 'trace.safetensors'), framework='pt', device='cpu') as frozen:
        images = frozen.get_tensor('prepared.images').cuda()
        with safe_open(str(args.checkpoint), framework='pt', device='cpu') as checkpoint:
            prefix = 'model.dual_tower.vlm.model.visual.vision_tower.'
            weight = checkpoint.get_tensor(prefix + 'patch_embed.proj.weight').to('cuda', torch.bfloat16)
            bias = checkpoint.get_tensor(prefix + 'patch_embed.proj.bias').to('cuda', torch.bfloat16)
            position = checkpoint.get_tensor(prefix + 'pos_embed').to('cuda', torch.bfloat16)
        if tuple(images.shape) != (3, 6, 3, 224, 224) or tuple(weight.shape) != (1152, 3, 16, 16):
            raise ValueError('Patch convolution geometry differs from the accepted frozen contract')
        patch_expected = torch.cat([frozen.get_tensor(f'vision.patch_embed.{i}') for i in range(3)]).cuda()
        patch_a, ops_a = observe('patch_per_camera', lambda: torch.cat([
            F.conv2d(camera, weight, bias, stride=16).flatten(2).transpose(1, 2) for camera in images]))
        patch_b, ops_b = observe('patch_combined_batch', lambda:
            F.conv2d(images.reshape(18, 3, 224, 224), weight, bias, stride=16).flatten(2).transpose(1, 2))
        results['patch_conv'] = {'per_camera_vs_reference': metric(patch_a, patch_expected),
                                 'combined_vs_reference': metric(patch_b, patch_expected),
                                 'combined_vs_per_camera': metric(patch_b, patch_a),
                                 'cpu_ops_per_camera': ops_a, 'cpu_ops_combined': ops_b}
        pos_a, _ = observe('position_direct_bf16', lambda: resize_position(position, (14, 14), promote=False))
        pos_b, _ = observe('position_fp32_then_bf16', lambda: resize_position(position, (14, 14), promote=True))
        pos_expected = frozen.get_tensor('vision.position_embedding').cuda()
        results['position_resize'] = {'bf16_vs_reference': metric(pos_a, pos_expected),
                                     'fp32_vs_reference': metric(pos_b, pos_expected),
                                     'fp32_vs_bf16': metric(pos_b, pos_a)}
        projection_weight = frozen.get_tensor('vision.proj1_weight').cuda()
        projection_bias = frozen.get_tensor('vision.proj1_bias').cuda()
        originals = [frozen.get_tensor(f'vision.proj1_input.{i}').cuda() for i in range(3)]
        strides = [frozen.get_tensor(f'vision.proj1_input_strides.{i}').tolist() for i in range(3)]
        official_views, native_view, combined = projection_inputs(originals, strides, live)
        projection_expected = torch.cat([frozen.get_tensor(f'vision.proj1.{i}') for i in range(3)]).cuda()
        proj_a, ops_a = observe('proj1_official_views', lambda: torch.cat([
            F.linear(value, projection_weight, projection_bias) for value in official_views]))
        proj_b, ops_b = observe('proj1_native_view_batch', lambda: F.linear(native_view, projection_weight, projection_bias))
        proj_c, ops_c = observe('proj1_contiguous_batch_control', lambda: F.linear(combined, projection_weight, projection_bias))
        results['merger_proj1'] = {'official_vs_reference': metric(proj_a, projection_expected),
            'native_layout_vs_reference': metric(proj_b, projection_expected),
            'contiguous_control_vs_reference': metric(proj_c, projection_expected),
            'native_layout_vs_official': metric(proj_b, proj_a), 'contiguous_vs_official': metric(proj_c, proj_a),
            'official_input_strides': [list(value.stride()) for value in official_views],
            'native_input_stride': list(native_view.stride()), 'contiguous_control_stride': list(combined.stride()),
            'cpu_ops_official': ops_a, 'cpu_ops_native': ops_b, 'cpu_ops_contiguous_control': ops_c}
    save_file(outputs, str(args.output / 'operator_outputs.safetensors'))
    report = {'kind': 'diagnostic_frozen_operators_not_action_or_performance_acceptance', 'label': args.label,
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
              'job_id': os.environ['SLURM_JOB_ID'], 'official_trace': str(args.official_trace),
              'checkpoint_stat': {'bytes': args.checkpoint.stat().st_size, 'mtime_ns': args.checkpoint.stat().st_mtime_ns},
              'results': results, 'model_constructed': False, 'action_gate_changed': False,
              'caveat': 'Compare arms within each runtime. Reference mismatch can reflect environment/kernel differences, not the controlled variable alone.'}
    (args.output / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'label': args.label, 'results': results}), flush=True)


if __name__ == '__main__':
    main()
