"""Diagnostic frozen native inputs; no policy precision or solver changes."""
from __future__ import annotations


def admit_historical_reference(report, *, expected_job, variant):
    if (report.get('engine') != 'official' or report.get('job_id') != expected_job
            or report.get('variant') != variant or report.get('oracle', {}).get('exact') is not True):
        raise ValueError('Historical official reference identity/oracle admission failed')


def publish_after_cleanup(pending, state, save):
    if state.get('cleanup', {}).get('passed') is not True or state.get('execution') != 'diagnostic_eager':
        raise ValueError('Folded diagnostic cannot publish completed traces before eager cleanup')
    for item in pending:
        item['report'].update(diagnostic_cleanup=state['cleanup'], execution_mode=state['execution'],
                              performance_measurement=False)
        save(item)


def admit_candidate_prior(record, saved, data, *, variant, expected_job, checkpoint, runtime, fixture):
    """Require the actual historical folded candidate, including its raw replay."""
    import numpy as np
    extra = record.get('experimental_merger_proj1', {})
    proof = extra.get('precondition', {})
    if (record.get('job_id') != expected_job or record.get('fixture') != fixture
            or fixture.get('manifest', {}).get('variant') != variant
            or record.get('checkpoint') != checkpoint or record.get('runtime') != runtime
            or record.get('runtime_checks_passed') is not True or extra.get('arm') != 'camera_proj1'
            or extra.get('fold_spatial') is not True or extra.get('effective_projection') != 'per_camera_hwd'
            or extra.get('cleanup', {}).get('passed') is not True
            or record.get('graph_eager_check', {}).get('passed') is not True
            or proof.get('operator_exact') is not True or proof.get('fold_spatial') is not True
            or proof.get('job_id') != expected_job or proof.get('source_revision') != extra.get('source_revision')):
        raise ValueError('Folded candidate prior identity/cleanup/proof admission failed')
    for old, current in (('noise', 'noise'), ('input_state', 'state'), ('input_prompt', 'prompt'),
                         ('oracle_actions32', 'expected_actions32')):
        if saved[old].dtype != np.asarray(data[current]).dtype or not np.array_equal(saved[old], data[current]):
            raise ValueError(f'Folded candidate prior input mismatch: {old}')
    actions = saved['actions32']
    if (saved['replay_actions32'].shape != (2, *actions.shape)
            or any(not np.array_equal(actions, item) for item in saved['replay_actions32'])
            or not np.array_equal(actions, saved['graph_actions32'])
            or not np.array_equal(actions, saved['eager_actions32'])):
        raise ValueError('Folded candidate prior raw replay/graph evidence is inconsistent')
    return {'admitted': True, 'historical_job_id': expected_job,
            'source_revision': extra['source_revision'], 'variant': variant,
            'original_oracle_gate_passed': record.get('official_oracle', {}).get('same_action_gate_passed')}


def reference_euler_step(latent, velocity, *, num_steps=10):
    """CPU characterization of the recorded CUDA BF16 increment, not a solver."""
    import torch
    if (type(num_steps) is not int or num_steps < 1 or latent.dtype != torch.float32
            or velocity.dtype != torch.bfloat16 or latent.shape != velocity.shape
            or not torch.isfinite(latent).all() or not torch.isfinite(velocity).all()):
        raise ValueError('Recorded integration needs matching finite FP32 latent/BF16 velocity and positive steps')
    step = torch.tensor(-1. / num_steps, dtype=torch.bfloat16, device=velocity.device)
    return latent + step * velocity


def verify_reference_integration(reference, *, num_steps=10):
    import torch
    if type(num_steps) is not int or num_steps < 1:
        raise ValueError('Invalid recorded integration step count')
    time = torch.tensor(1., dtype=torch.float32)
    dt = torch.tensor(-1. / num_steps, dtype=torch.float32)
    for index in range(num_steps):
        recorded_time = reference[f'step{index}.time']
        if recorded_time.dtype != torch.float32 or recorded_time.numel() != 1 or recorded_time.item() != time.item():
            raise ValueError('Recorded time differs from the original FP32 recursive schedule')
        result = reference_euler_step(reference[f'step{index}.latent_before'],
            reference[f'step{index}.velocity'], num_steps=num_steps)
        expected = reference[f'step{index+1}.latent_before'] if index+1 < num_steps else reference['actions']
        if not torch.equal(result, expected):
            raise ValueError(f'Recorded integration does not reproduce observed next state at step{index}')
        time += dt
    return {'exact': True, 'steps': num_steps,
            'recipe': 'characterized BF16 scalar times BF16 raw velocity, then FP32 addition',
            'scope': 'CPU replay admission of historical tensors, not a production math override'}


def snapshot_velocity_inputs(args):
    import torch
    if len(args) != 9:
        raise ValueError('Expected nine native velocity arguments')
    latent, time, state, mask, cos, sin, k, v, prefix = args
    if (latent.dtype != torch.float32 or time.dtype != torch.bfloat16 or state.dtype != torch.bfloat16
            or mask.dtype != torch.bool or cos.dtype != torch.float32 or sin.dtype != torch.float32
            or k.dtype != torch.float32 or v.dtype != torch.bfloat16 or k.ndim != 5 or k.shape != v.shape
            or type(prefix) is not int or prefix < 1 or k.shape[-2] != prefix+1+latent.shape[1]):
        raise ValueError('Unexpected official-local native velocity snapshot shape/dtype')
    result = {name: value.detach().clone() for name, value in
              (('state_emb', state), ('smask', mask), ('scos', cos), ('ssin', sin))}
    for name, value in (('kbuf', k), ('vbuf', v)):
        result[name] = torch.zeros_like(value)
        result[name][..., :prefix, :].copy_(value[..., :prefix, :])
    result['S_p'] = prefix
    return result


def evaluate_controls(frontend, pipe, reference, native_trace, snapshot, *, common, num_steps=10):
    """Own-trajectory replay then two KV conditions on the SAME official inputs."""
    import torch
    verify_reference_integration(reference, num_steps=num_steps)
    prefix = snapshot['S_p']
    k0, v0 = snapshot['kbuf'], snapshot['vbuf']
    def native_pair():
        return k0.clone(), v0.clone()
    output, rows = {}, []
    def call(latent, embedding, pair):
        if latent.dtype != torch.float32 or embedding.dtype != torch.bfloat16 or not torch.isfinite(embedding).all():
            raise ValueError('Frozen evaluator requires FP32 latent and finite BF16 time embedding')
        readonly = (latent, embedding, snapshot['state_emb'], snapshot['smask'], snapshot['scos'], snapshot['ssin'])
        before = tuple(value.clone() for value in readonly)
        k, v = pair
        before_k, before_v = k[..., :prefix, :].clone(), v[..., :prefix, :].clone()
        result = pipe.evaluate_velocity(latent, embedding, snapshot['state_emb'], snapshot['smask'],
            snapshot['scos'], snapshot['ssin'], k, v, prefix)
        if any(not torch.equal(value, prior) for value, prior in zip(readonly, before)):
            raise ValueError('Raw evaluator mutated a read-only latent/time/condition input')
        if not torch.equal(k[..., :prefix, :], before_k) or not torch.equal(v[..., :prefix, :], before_v):
            raise ValueError('Raw evaluator mutated frozen prefix KV')
        if result.shape != latent.shape or result.dtype != torch.bfloat16 or not torch.isfinite(result).all():
            raise ValueError('Unexpected raw velocity shape/dtype/finite state')
        return result
    replay_pair = native_pair()
    for step in range(num_steps):
        latent = native_trace[f'step{step}.latent_before'].to(frontend.device)
        embedding = native_trace[f'step{step}.time_embedding'].to(frontend.device)
        result = call(latent, embedding, replay_pair)
        if not torch.equal(result.detach().cpu(), native_trace[f'step{step}.velocity']):
            raise ValueError(f'Native own-trajectory replay differs at step{step}')
        output[f'native_replay.step{step}.velocity'] = result.detach().cpu().clone()
    conditions = {'native_kv': native_pair()}
    official = []
    for name, template in (('k', k0), ('v', v0)):
        expanded = common.expand_official_kv(common._array(reference['prefix.'+name]), snapshot['perm'],
            frontend.n_heads, template.shape[-2]-prefix)
        item = torch.from_numpy(expanded).to(device=frontend.device, dtype=template.dtype)
        if item.shape != template.shape:
            raise ValueError('Official KV expansion does not match the admitted native geometry')
        official.append(item)
    conditions['official_kv_sorted'] = tuple(official)
    for step in range(num_steps):
        latent = reference[f'step{step}.latent_before'].to(frontend.device)
        embedding = frontend._sinusoidal_time(float(reference[f'step{step}.time'].reshape(-1)[0])).to(
            device=frontend.device, dtype=torch.bfloat16)
        output[f'frozen_inputs.step{step}.latent'] = latent.detach().cpu().clone()
        output[f'frozen_inputs.step{step}.time_embedding'] = embedding.detach().cpu().clone()
        for condition, pair in conditions.items():
            result = call(latent, embedding, pair)
            output[f'{condition}.step{step}.velocity'] = result.detach().cpu().clone()
            rows.append({'step': step, 'condition': condition,
                **common.difference(common._array(result), common._array(reference[f'step{step}.velocity']))})
        rows.append({'step': step, 'condition': 'kv_effect_native_vs_official',
            **common.difference(common._array(output[f'native_kv.step{step}.velocity']),
                                common._array(output[f'official_kv_sorted.step{step}.velocity']))})
    return output, rows, {'native_replay_exact': True, 'prefix_preserved': True,
        'independent_scratch': True, 'interpretation': 'teacher-forced official trajectory, conditional KV intervention only'}
