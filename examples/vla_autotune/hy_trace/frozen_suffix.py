"""Allocated frozen suffix / Expert layer0 Q controls, never a policy override."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace


def build_suffix(state, latent, time, w):
    import torch
    import torch.nn.functional as F
    state_emb = F.linear(state.to(torch.bfloat16), w._state_w, w._state_b)[:, None]
    action = F.linear(latent.to(torch.bfloat16), w._ain_w, w._ain_b)
    time_input = torch.cat((action, time.expand_as(action)), dim=2)
    hidden = F.silu(F.linear(time_input, w._atmlp_in_w, w._atmlp_in_b))
    fused = F.linear(hidden, w._atmlp_out_w, w._atmlp_out_b)
    return {'state_embedding': state_emb, 'action_embedding': action, 'time_input': time_input,
            'time_hidden': hidden, 'suffix': torch.cat((state_emb, fused), dim=1)}


def local_rms(hidden, weight, epsilon=1e-5):
    import torch
    dtype = hidden.dtype
    value = hidden.float()
    variance = value.pow(2).mean(-1, keepdim=True)
    value = value * torch.rsqrt(variance + epsilon)
    return weight * value.to(dtype)


def project_qkv(hidden, q, k, v, *, fused):
    import torch
    if hidden.ndim != 3 or hidden.shape[0] != 1:
        raise ValueError('Frozen Expert projection requires the recorded batch1 suffix')
    x = hidden[0]
    if fused:
        return (x @ torch.cat((q, k, v), dim=0).t())[None]
    return torch.cat((x @ q.t(), x @ k.t(), x @ v.t()), dim=-1)[None]


def rope_tables(inverse, positions):
    import torch
    with torch.autocast(device_type=inverse.device.type, enabled=False):
        expanded = inverse[None, :, None].float().expand(positions.shape[0], -1, 1)
        frequencies = (expanded @ positions[:, None, :].float()).transpose(1, 2)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos()[:, None], embedding.sin()[:, None]


def normalized_query(qkv, *, query_width, head_dim, cos, sin, weight):
    import torch
    batch, length = qkv.shape[:2]
    q = qkv[..., :query_width].reshape(batch, length, query_width//head_dim, head_dim).transpose(1, 2)
    first, second = q.chunk(2, dim=-1)
    q = q * cos + torch.cat((-second, first), dim=-1) * sin
    return local_rms(q, weight)


def _time_embedding(time, width, device):
    import torch
    fraction = torch.linspace(0., 1., width//2, dtype=torch.float64, device=device)
    period = 4e-3 * (4. / 4e-3) ** fraction
    scaling = 1. / period * 2 * math.pi
    value = torch.tensor([time], dtype=torch.float32, device=device)
    argument = scaling[None] * value[:, None]
    return torch.cat((torch.sin(argument), torch.cos(argument)), dim=1).to(torch.bfloat16)


def admit_sources(official, native, *, variant, official_job, native_job, checkpoint):
    extra = native.get('experimental_projection', {})
    proof = extra.get('precondition', {})
    if (official.get('engine') != 'official' or official.get('job_id') != official_job
            or official.get('variant') != variant or official.get('oracle', {}).get('exact') is not True
            or native.get('engine') != 'native' or native.get('job_id') != native_job or native.get('variant') != variant
            or native.get('execution_mode') != 'diagnostic_eager' or native.get('config', {}).get('use_cuda_graph') is not False
            or native.get('artifact_complete') is not True or native.get('diagnostic_cleanup', {}).get('passed') is not True
            or extra.get('fold_spatial') is not True or proof.get('checkpoint') != checkpoint
            or proof.get('operator_exact') is not True or proof.get('job_id') != native_job
            or proof.get('source_revision') != extra.get('source_revision')):
        raise ValueError('Frozen suffix historical source/fold/checkpoint identity admission failed')


def publish(folder, tensors, report):
    from safetensors.torch import save_file
    save_file({name: value.detach().cpu().contiguous().clone() for name, value in tensors.items()},
              str(folder/'operator_outputs.safetensors'))
    temporary = folder/'result.json.tmp'
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(folder/'result.json')


def _sibling(name):
    spec = importlib.util.spec_from_file_location('hy_suffix_'+name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--engine', choices=('official', 'native'), required=True)
    for name in ('checkpoint', 'official', 'native', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--official-job', default='308690')
    parser.add_argument('--native-job', default='313766')
    args = parser.parse_args(argv)
    common = _sibling('diagnose')
    common.require_step()  # before Torch/checkpoint IO/output creation
    import torch
    from safetensors import safe_open
    if tuple(torch.cuda.get_device_capability()) != (8, 0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('Frozen suffix controls require the recorded A800 target')
    if not torch.__version__.startswith('2.7.1' if args.engine == 'official' else '2.11.0'):
        raise RuntimeError('Use the recorded engine-specific Torch runtime')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    ablation = _sibling('proj1_camera_ablation')
    checkpoint = ablation._checkpoint_stat(args.checkpoint)
    selected = {name: 'model.'+key for name, key in (
        ('_state_w', 'state_proj.weight'), ('_state_b', 'state_proj.bias'),
        ('_ain_w', 'action_in_proj.weight'), ('_ain_b', 'action_in_proj.bias'),
        ('_atmlp_in_w', 'action_time_mlp_in.weight'), ('_atmlp_in_b', 'action_time_mlp_in.bias'),
        ('_atmlp_out_w', 'action_time_mlp_out.weight'), ('_atmlp_out_b', 'action_time_mlp_out.bias'),
        ('_ln', 'dual_tower.expert.model.layers.0.input_layernorm_v.weight'),
        ('_q', 'dual_tower.expert.model.layers.0.self_attn.q_proj_v.weight'),
        ('_k', 'dual_tower.expert.model.layers.0.self_attn.k_proj_v.weight'),
        ('_v', 'dual_tower.expert.model.layers.0.self_attn.v_proj_v.weight'),
        ('_qnorm', 'dual_tower.vlm.model.language_model.model.layers.0.self_attn.query_layernorm.weight'))}
    with safe_open(checkpoint['path'], framework='pt', device='cpu') as saved:
        w = SimpleNamespace(**{name: saved.get_tensor(key).to('cuda', torch.bfloat16) for name, key in selected.items()})
    if (w._state_w.shape != (1024,32) or w._ain_w.shape != (1024,32)
            or w._atmlp_in_w.shape != (1024,2048) or w._atmlp_out_w.shape != (1024,1024)
            or w._ln.shape != (1024,) or w._q.shape != (2048,1024)
            or w._k.shape != (512,1024) or w._v.shape != (512,1024) or w._qnorm.shape != (128,)):
        raise ValueError('Selected checkpoint suffix/Expert geometry differs from the recorded contract')
    args.output.mkdir(parents=True, exist_ok=False)
    admitted_all = True
    for variant in ('baseline','state'):
        official_report = json.loads((args.official/variant/'result.json').read_text())
        native_report = json.loads((args.native/variant/'result.json').read_text())
        admit_sources(official_report, native_report, variant=variant, official_job=args.official_job,
                      native_job=args.native_job, checkpoint=checkpoint)
        folder = args.output/variant
        folder.mkdir()
        outputs, metrics = {}, {}
        def compare(name, actual, expected):
            metrics[name] = common.difference(common._array(actual), common._array(expected))
            metrics[name]['unequal_elements'] = int((actual != expected).sum().item())
        with torch.inference_mode(), safe_open(str(args.official/variant/'trace.safetensors'), framework='pt', device='cpu') as official, safe_open(str(args.native/variant/'trace.safetensors'), framework='pt', device='cpu') as native:
            for name in ('prepared.state', 'step0.latent_before'):
                if not torch.equal(official.get_tensor(name), native.get_tensor(name)):
                    raise ValueError('Historical frozen suffix input values differ')
            state = official.get_tensor('prepared.state').cuda()
            latent = official.get_tensor('step0.latent_before').cuda()
            time = _time_embedding(float(official.get_tensor('step0.time').reshape(-1)[0]), 1024, 'cuda')
            native_time = native.get_tensor('step0.time_embedding').cuda()
            compare('time_vs_observed_native', time, native_time)
            cos, sin = rope_tables(official.get_tensor('language.rope_inv_freq').cuda(), official.get_tensor('suffix.position_ids').cuda())
            compare('cos_vs_observed_native', cos, native.get_tensor('suffix.cos').cuda())
            compare('sin_vs_observed_native', sin, native.get_tensor('suffix.sin').cuda())
            outputs.update(build_suffix(state, latent, time, w))
            expected_suffix = official.get_tensor('suffix.embeddings').cuda()
            compare('suffix_vs_recorded_official', outputs['suffix'], expected_suffix)
            normalized = local_rms(outputs['suffix'], w._ln)
            normalized_flat = local_rms(outputs['suffix'][0], w._ln)[None]
            outputs['normalized_suffix'] = normalized
            outputs['normalized_suffix_flat'] = normalized_flat
            compare('input_norm_batched_vs_flat', normalized, normalized_flat)
            admitted = all(row['exact'] for row in metrics.values())
            admitted_all = admitted_all and admitted
            if admitted:
                for name, fused in (('separate',False),('fused',True)):
                    qkv = project_qkv(normalized, w._q, w._k, w._v, fused=fused)
                    qnorm = normalized_query(qkv, query_width=2048, head_dim=128, cos=cos, sin=sin, weight=w._qnorm)
                    outputs[name+'_qkv'] = qkv
                    outputs[name+'_query_norm'] = qnorm
                    compare(name+'_query_norm_vs_recorded', qnorm, official.get_tensor('query_norm.layer0.1').cuda())
                compare('fused_vs_separate_qkv', outputs['fused_qkv'], outputs['separate_qkv'])
                compare('fused_vs_separate_query_norm', outputs['fused_query_norm'], outputs['separate_query_norm'])
        report = {'kind':'frozen_step0_suffix_expert0_query_controls_not_policy_gate',
            'status':'diagnostic_complete' if admitted else 'input_admission_failed', 'input_admitted':admitted,
            'engine':args.engine,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,'variant':variant,
            'checkpoint':checkpoint,'selected_checkpoint_keys':list(selected.values()),'model_constructed':False,
            'historical_official':{'job_id':args.official_job,'path':str((args.official/variant).resolve())},
            'historical_native':{'job_id':args.native_job,'path':str((args.native/variant).resolve())},
            'metrics':metrics,'performance_measurement':False,'policy_qualification':'unchanged_failed6of7',
            'tensor_metadata':{name:{'shape':list(value.shape),'live_stride':list(value.stride()),
                'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}
                for name,value in {**outputs,'input_state':state,'input_latent':latent,
                    'time_embedding':time,'rope_cos':cos,'rope_sin':sin}.items()},
            'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'caveat':'STEP0 only, recorded suffix and layer0 normalized-query boundary; no attention/whole-Expert/full-policy equivalence claim'}
        publish(folder,outputs,report)
        print(json.dumps({'engine':args.engine,'variant':variant,'input_admitted':admitted,'metrics':metrics}),flush=True)
    return 0 if admitted_all else 1


if __name__ == '__main__':
    raise SystemExit(main())
