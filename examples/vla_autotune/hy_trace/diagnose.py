"""Allocation-only Hy boundary observations; not a policy or latency benchmark."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager
import importlib.util
import inspect
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np


VARIANTS = ('baseline', 'state', 'noise')


def require_step(env=None, hostname=None):
    env = os.environ if env is None else env
    host = (hostname or socket.gethostname()).split('.')[0]
    if (not env.get('SLURM_JOB_ID', '').isdigit() or not env.get('SLURM_STEP_ID', '').isdigit()
            or host.startswith(('ln', 'login')) or host != env.get('SLURMD_NODENAME', '').split('.')[0]
            or env.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1')):
        raise RuntimeError('Diagnostic model execution requires an allocated GPU srun compute step')


def _perm(perm, rows):
    perm = np.asarray(perm)
    if perm.shape != (rows,) or perm.dtype.kind not in 'iu' or not np.array_equal(np.sort(perm), np.arange(rows)):
        raise ValueError('Expected a complete prefix permutation')
    return perm


def expand_official_kv(cache, perm, query_heads, suffix_rows):
    """[L,B,P,KV,D] -> sorted [L,B,Q,P+suffix,D], no live tensor mutation."""
    cache = np.asarray(cache)
    if cache.ndim != 5 or query_heads < 1 or query_heads % cache.shape[3] or suffix_rows < 0:
        raise ValueError('Invalid official KV geometry')
    perm = _perm(perm, cache.shape[2])
    prefix = np.repeat(cache[:, :, perm].transpose(0, 1, 3, 2, 4), query_heads // cache.shape[3], axis=2)
    out = np.zeros((*prefix.shape[:3], prefix.shape[3] + suffix_rows, prefix.shape[4]), dtype=cache.dtype)
    out[..., :prefix.shape[3], :] = prefix
    return out


def canonical_native_kv(cache, perm, kv_heads):
    cache = np.asarray(cache)
    if cache.ndim != 5 or kv_heads < 1 or cache.shape[2] % kv_heads:
        raise ValueError('Invalid native KV geometry')
    perm = _perm(perm, len(perm))
    if cache.shape[3] < len(perm):
        raise ValueError('Native cache does not cover prefix')
    repeat = cache.shape[2] // kv_heads
    prefix = cache[..., :len(perm), :]
    compact = prefix[:, :, ::repeat]
    if not np.array_equal(np.repeat(compact, repeat, axis=2), prefix):
        raise ValueError('Native repeated head groups disagree')
    return compact[:, :, :, np.argsort(perm)].transpose(0, 1, 3, 2, 4)


def observer_functions(path):
    # Reuse only observers/load-report validation. Never run the old module's
    # hashing, manifest generation, private fixture generator or model driver.
    source = ast.parse(Path(path).read_text())
    wanted = {'validate_load_report', 'diagnostic_capture'}
    definitions = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise ValueError('Reference observer source lacks the expected functions')
    scope = {'contextmanager': contextmanager}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    return scope['validate_load_report'], scope['diagnostic_capture']


def wrap_method(stack, owner, name, factory):
    missing = object()
    own = owner.__dict__.get(name, missing)
    original = getattr(owner, name)
    setattr(owner, name, factory(original))
    def restore():
        if own is missing:
            delattr(owner, name)
        else:
            setattr(owner, name, own)
    stack.callback(restore)


def difference(a, b):
    a, b = np.asarray(a).astype(np.float64), np.asarray(b).astype(np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Diagnostic comparison requires matching finite tensors')
    delta = a - b
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return {'shape': list(a.shape), 'exact': bool(np.array_equal(a, b)),
            'max_abs': float(abs(delta).max()),
            'relative_l2': float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12)),
            'cosine': float(a.ravel() @ b.ravel() / denom) if denom else float(np.array_equal(a, b))}


def _array(tensor):
    return tensor.detach().cpu().float().numpy()


class ReferenceInputMismatch(ValueError):
    pass


def admitted_frozen_evaluation(reference, native, data, report, variant, evaluate):
    invariant_names = ('prepared.images', 'prefix.embeddings', 'prefix.pad_mask',
                       'prefix.modality_mask', 'prefix.k', 'prefix.v')
    invariants = report.get('prefix_invariance', {})
    if (report.get('variant') != variant or report.get('oracle', {}).get('exact') is not True
            or any(invariants.get(name) is not True for name in invariant_names)):
        raise ReferenceInputMismatch('Official reference label/oracle/prefix invariance admission failed')
    comparisons = [('actions', np.asarray(data['expected_actions32'])[None]),
                   ('step0.latent_before', np.asarray(data['noise'])[None])]
    for key in ('prepared.images', 'prepared.state', 'prepared.tokens', 'prepared.token_mask',
                'prefix.pad_mask', 'prefix.modality_mask'):
        if key not in native:
            raise ReferenceInputMismatch(f'Missing native input identity: {key}')
        if key not in reference or native[key].dtype != reference[key].dtype:
            raise ReferenceInputMismatch(f'Official reference input dtype differs: {key}')
        comparisons.append((key, _array(native[key])))
    for key, expected in comparisons:
        if key not in reference or not np.array_equal(_array(reference[key]), expected):
            raise ReferenceInputMismatch(f'Official reference does not match current input/oracle: {key}')
    # Numeric prefix embeddings/KV are deliberately NOT required to match:
    # their contribution is exactly what this controlled experiment investigates.
    return evaluate()


def validate_trace_options(args):
    folded = getattr(args, 'folded_proj1', False)
    capture_attention = getattr(args, 'capture_eager_attention', False)
    probe_order=getattr(args,'probe_suffix_key_order',False)
    order_options=('order_preflight_report','order_baseline','order_baseline_job')
    if probe_order and (not folded or args.engine!='native' or any(getattr(args,key,None) is None for key in order_options)):
        raise ValueError('Suffix order probe requires folded native mode, current order proof and historical sorted baseline')
    if not probe_order and any(getattr(args,key,None) is not None for key in order_options):
        raise ValueError('Suffix order options cannot silently affect a non-intervention trace')
    if capture_attention and (folded or args.engine != 'official' or any(getattr(args,key,None) is None
            for key in ('source_revision','reference','reference_job'))):
        raise ValueError('Official eager attention capture requires explicit historical reference/source identity')
    if folded and (args.engine != 'native' or any(getattr(args, key, None) is None for key in
            ('preflight_report', 'source_revision', 'prior_job', 'reference_job', 'reference', 'prior'))):
        raise ValueError('Folded native trace requires current proof and historical prior/reference identity')
    if not folded and not capture_attention and any(getattr(args, key, None) is not None for key in
                         ('preflight_report', 'source_revision', 'prior_job', 'reference_job')):
        raise ValueError('Folded trace identity options cannot silently affect a legacy trace')
    if capture_attention and any(getattr(args,key,None) is not None for key in ('preflight_report','prior_job')):
        raise ValueError('Folded candidate proof options cannot affect official attention capture')
    variants = getattr(args, 'variants', None) or (('baseline', 'state') if folded or capture_attention else VARIANTS)
    if len(variants) != len(set(variants)) or any(value not in VARIANTS for value in variants):
        raise ValueError('Invalid or repeated diagnostic variant')
    if probe_order and any(value not in ('baseline','state') for value in variants):
        raise ValueError('Suffix order probe variants are limited to admitted baseline/state')
    return tuple(variants)


def _sibling(name):
    spec = importlib.util.spec_from_file_location('hy_trace_' + name, Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(folder, tensors, report, *, atomic_report=False):
    from safetensors.torch import save_file
    save_file({name: value.contiguous() for name, value in tensors.items()}, str(folder / 'trace.safetensors'))
    path = folder/'result.json'
    text = json.dumps(report, indent=2, allow_nan=False) + '\n'
    if atomic_report:
        temporary = folder/'result.json.tmp'
        temporary.write_text(text)
        temporary.replace(path)
    else:
        path.write_text(text)


def publish_folded_trace(item):
    from safetensors.torch import save_file
    save_file({name: value.contiguous() for name, value in item['frozen'].items()}, str(item['folder']/'frozen_velocities.safetensors'))
    item['report']['artifact_complete'] = True
    _save(item['folder'], item['tensors'], item['report'], atomic_report=True)
    print(json.dumps({'variant': item['report']['variant'], 'folded_diagnostic': True,
        'prior_graph_exact': True, 'cleanup_passed': True, 'boundaries': item['report']['reference_boundaries'],
        'frozen_velocities': item['report']['frozen_velocity_comparisons']}), flush=True)


def _rec(tensors, metadata, name, value):
    metadata[name] = tensor_descriptor(value)
    tensors[name] = value.detach().cpu().contiguous().clone()
    metadata[name]['artifact_stride'] = list(tensors[name].stride())


def tensor_descriptor(value):
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'live_stride': list(value.stride())}


def record_native_cache(tensors, metadata, name, value, perm, kv_heads):
    import torch
    metadata[name + '.live_cache'] = tensor_descriptor(value)
    canonical = canonical_native_kv(_array(value), perm, kv_heads)
    artifact = torch.from_numpy(canonical.copy()).to(value.dtype)
    tensors[name] = artifact
    metadata[name] = {'dtype': str(artifact.dtype), 'shape': list(artifact.shape),
                      'artifact_stride': list(artifact.stride()),
                      'representation': 'canonical_cpu_original_token_order_compressed_kv_heads'}


@contextmanager
def observe_step_metadata(model, metadata):
    with ExitStack() as stack:
        step = 0
        def observer(original):
            def call(state, pad, cache, latent, time):
                nonlocal step
                index = step
                step += 1
                metadata[f'step{index}.latent_before'] = tensor_descriptor(latent)
                metadata[f'step{index}.time'] = tensor_descriptor(time)
                result = original(state, pad, cache, latent, time)
                metadata[f'step{index}.velocity'] = tensor_descriptor(result[0])
                return result
            return call
        wrap_method(stack, model, 'denoise_step', observer)
        yield


def _invariance(baseline, tensors):
    import torch
    names = ('prepared.images', 'prefix.embeddings', 'prefix.pad_mask', 'prefix.modality_mask', 'prefix.k', 'prefix.v')
    if baseline is None:
        return {name: tensors[name].clone() for name in names}, {name: True for name in names}
    return baseline, {name: bool(torch.equal(baseline[name], tensors[name])) for name in names}


def run_official(args, torch):
    from safetensors.torch import load_model
    from hy_vla import HyVLA, HyVLAConfig
    if not torch.__version__.startswith('2.7.1'):
        raise RuntimeError('Use the recorded official Torch2.7.1 environment, not the native overlay')
    validate_load, observe = observer_functions(args.observer_source)
    capture_attention = getattr(args,'capture_eager_attention',False)
    if capture_attention:
        capture = _sibling('eager_attention_capture')
        controls = _sibling('native_controls')
    policy = HyVLA(HyVLAConfig.from_pretrained(str(args.checkpoint), local_files_only=True))
    load_report = validate_load(*load_model(policy, str(args.checkpoint / 'model.safetensors'), strict=False, device='cpu'))
    policy.eval().to(device='cuda', dtype=torch.bfloat16)
    baseline = None
    for variant in getattr(args, 'variants', None) or VARIANTS:
        folder = args.output / variant
        folder.mkdir()
        with np.load(args.fixtures / f'{variant}.npz', allow_pickle=False) as archive:
            data = {name: archive[name].copy() for name in archive.files}
        batch = {'task': [str(data['prompt'].item())],
                 'observation.state': torch.from_numpy(data['state'])[None].cuda().bfloat16()}
        for camera in ('top_head', 'hand_left', 'hand_right'):
            batch['observation.images.' + camera] = torch.from_numpy(data[camera]).permute(0, 3, 1, 2).contiguous()[None].float().div(255).cuda().bfloat16()
        images, masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        tokens, token_mask, _ = policy.prepare_language(batch)
        tensors, meta = {}, {}
        for name, value in (('prepared.images', torch.cat(images, 0)), ('prepared.state', state),
                            ('prepared.tokens', tokens), ('prepared.token_mask', token_mask)):
            _rec(tensors, meta, name, value)
        with observe(policy, tensors), observe_step_metadata(policy.model, meta), ExitStack() as stack:
            if capture_attention:
                capture_state = stack.enter_context(capture.observe_eager_attention(policy.model.dual_tower,tensors,meta,
                    num_layers=int(policy.model.dual_tower.vlm.config.num_hidden_layers),num_steps=int(policy.model.config.num_steps)))
            def tower_observer(original):
                def call(*a, **kw):
                    result = original(*a, **kw)
                    if kw.get('fill_kv_cache'):
                        for key, field in (('k', 'key_states'), ('v', 'value_states')):
                            values = [result[1][i][field] for i in range(32)]
                            for i, value in enumerate(values):
                                meta[f'prefix.{key}.layer{i}'] = tensor_descriptor(value)
                            tensors[f'prefix.{key}'] = torch.stack([value.detach().cpu() for value in values])
                            artifact = tensors[f'prefix.{key}']
                            meta[f'prefix.{key}'] = {'dtype': str(artifact.dtype), 'shape': list(artifact.shape),
                                                    'artifact_stride': list(artifact.stride()),
                                                    'representation': 'stacked_cpu_original_order_layer_caches'}
                    return result
                return call
            wrap_method(stack, policy.model.dual_tower, 'forward', tower_observer)
            output = policy.model.sample_actions(images=images, img_masks=masks, state=state,
                lang_tokens=tokens, lang_masks=token_mask, noise=torch.from_numpy(data['noise'].copy())[None].cuda())
        tensors['suffix.state_embedding'] = tensors['suffix.embeddings'][:, :1].clone()
        for name in meta.keys() & tensors.keys():
            meta[name]['artifact_stride'] = list(tensors[name].stride())
        _rec(tensors, meta, 'actions', output)
        same = difference(_array(output)[0], data['expected_actions32'])
        baseline, invariant = _invariance(baseline, tensors)
        report = {'engine': 'official', 'variant': variant, 'oracle': same, 'prefix_invariance': invariant,
                  'load_report': load_report, 'torch': torch.__version__, 'hy_vla_source': sys.modules['hy_vla'].__file__,
                  'job_id': os.environ['SLURM_JOB_ID'], 'tensor_metadata': meta, 'performance_measurement': False}
        if capture_attention:
            from safetensors import safe_open
            historical_report = json.loads((args.reference/variant/'result.json').read_text())
            controls.admit_historical_reference(historical_report,expected_job=args.reference_job,variant=variant)
            names = (*capture.IDENTITY_KEYS,'prefix.attention_mask','suffix.attention_mask',
                     *(f'query_norm.layer0.{index}' for index in range(1+policy.model.config.num_steps)),
                     *(f'step{index}.{field}' for index in range(policy.model.config.num_steps)
                       for field in ('latent_before','time','velocity')))
            with safe_open(str(args.reference/variant/'trace.safetensors'),framework='pt',device='cpu') as saved:
                historical = {name:saved.get_tensor(name) for name in names}
            admission = capture.admit_recorded_attention(tensors,historical,num_steps=int(policy.model.config.num_steps))
            source = Path(inspect.getfile(type(policy.model.dual_tower))).resolve()
            stat = source.stat()
            report.update(artifact_complete=True,status='recorded_oracle_reproduced' if same['exact'] and all(invariant.values()) else 'reference_reproduction_failed',
                source_revision=args.source_revision,eager_attention_capture=capture_state,historical_attention_admission=admission,
                historical_official_reference={'job_id':args.reference_job,'path':str((args.reference/variant).resolve())},
                official_attention_source={'path':str(source),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},
                checkpoint=_sibling('proj1_camera_ablation')._checkpoint_stat(args.checkpoint),
                capture_scope='Observed official eager layer0 inputs/output/probabilities; no native policy qualification')
            torch.cuda.synchronize()
        _save(folder, tensors, report,atomic_report=capture_attention)
        print(json.dumps({'variant': variant, 'oracle_exact': same['exact'], 'invariant': invariant}), flush=True)
        if not same['exact'] or not all(invariant.values()):
            raise RuntimeError('Official trace prerequisites failed; do not interpret a mismatched reference as the recorded oracle')


def run_native(args, torch):
    from safetensors.torch import load_file
    from flash_rt.vla import BackendConfig, load_backend
    baseline = None
    folded = getattr(args, 'folded_proj1', False)
    probe_order=getattr(args,'probe_suffix_key_order',False)
    variants = getattr(args, 'variants', None) or VARIANTS
    config = BackendConfig(model='hyvla', checkpoint=str(args.checkpoint), hardware='cuda_sm80', precision='bf16',
        sampler='euler', num_steps=10, options={'use_cuda_graph': False, 'numerical_profile': 'official_local_fp32'})
    pending, admitted = [], {}
    if folded:
        import platform
        ablation, controls = _sibling('proj1_camera_ablation'), _sibling('native_controls')
        proof = json.loads(args.preflight_report.read_text())
        ablation.validate_precondition(proof, job_id=os.environ['SLURM_JOB_ID'],
            source_revision=args.source_revision, checkpoint=args.checkpoint, fold_spatial=True)
        if probe_order:
            order=_sibling('suffix_key_order')
            order_proof=json.loads(args.order_preflight_report.read_text())
            order.validate_order_proof(order_proof,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
                native_source=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py')
        validator_spec = importlib.util.spec_from_file_location('hy_trace_validator', Path(__file__).parents[1]/'validate_hyvla.py')
        validator = importlib.util.module_from_spec(validator_spec)
        validator_spec.loader.exec_module(validator)
        _, checkpoint_record = validator.checkpoint_record(args.checkpoint)
        runtime = {'python': platform.python_version(), 'executable': sys.executable,
                   'torch': torch.__version__, 'cuda': torch.version.cuda, 'numpy': np.__version__}
        for variant in variants:
            prior_folder = args.prior/variant
            prior_report = json.loads((prior_folder/'result.json').read_text())
            prior_exit = json.loads((prior_folder/'child_exit.json').read_text())['returncode']
            ablation.classify_exit(prior_exit, prior_report)  # state oracle mismatch stays a historical failure
            with np.load(prior_folder/'actions.npz', allow_pickle=False) as archive:
                prior_arrays = {name: archive[name].copy() for name in archive.files}
            with np.load(args.fixtures/f'{variant}.npz', allow_pickle=False) as archive:
                prior_input = {name: archive[name].copy() for name in archive.files}
            admitted[variant] = controls.admit_candidate_prior(prior_report, prior_arrays, prior_input,
                variant=variant, expected_job=args.prior_job, checkpoint=checkpoint_record, runtime=runtime,
                fixture=validator.fixture_record(args.fixtures/f'{variant}.npz'))
            if probe_order:
                baseline_report=json.loads((args.order_baseline/variant/'result.json').read_text())
                order.admit_sorted_baseline_report(baseline_report,job_id=args.order_baseline_job,
                    reference_job=args.reference_job,checkpoint=checkpoint_record,runtime=runtime)
        from flash_rt.frontends.torch.hyvla_probeflow import HyVLAFlowFrontend
        scope_state = {}
        backend_context = ablation.scoped_backend(load_backend, config, arm='camera_proj1', preflight=proof,
            state=scope_state, synchronize=torch.cuda.synchronize, expected_frontend_type=HyVLAFlowFrontend,
            fold_spatial=True, execution='diagnostic_eager')
    else:
        backend_context = load_backend(config)
    with backend_context as backend:
        frontend, pipe = backend.frontend, backend.frontend.pipe
        for variant in variants:
            folder = args.output / variant
            folder.mkdir()
            reference = load_file(str(args.reference / variant / 'trace.safetensors'))
            official_report = json.loads((args.reference / variant / 'result.json').read_text())
            if not official_report['oracle']['exact']:
                raise ValueError('Cannot use an unreproduced official oracle as a frozen trajectory')
            if folded:
                controls.admit_historical_reference(official_report, expected_job=args.reference_job, variant=variant)
                controls.verify_reference_integration(reference)
            with np.load(args.fixtures / f'{variant}.npz', allow_pickle=False) as archive:
                data = {name: archive[name].copy() for name in archive.files}
            keys = ('top_head', 'hand_left', 'hand_right')
            observation = {'images': {'observation.images.' + key: data[key] for key in keys},
                           'state': data['state'], 'prompt': str(data['prompt'].item())}
            tensors, meta, hold = {}, {}, {}
            with ExitStack() as stack:
                def preprocess(original):
                    def call(*a, **kw):
                        result = original(*a, **kw)
                        _rec(tensors, meta, 'prepared.images', torch.cat(result, 0))
                        return result
                    return call
                def assemble(original):
                    def call(*a, **kw):
                        _rec(tensors, meta, 'vision.merged', a[0])
                        result = original(*a, **kw)
                        for name, value in zip(('embeddings', 'pad_mask', 'block_mask', 'modality_mask'), result[:4]):
                            _rec(tensors, meta, 'prefix.' + name, value)
                        mm = result[3][0]
                        hold['perm'] = torch.cat([mm.nonzero().flatten(), (~mm).nonzero().flatten()]).cpu().numpy()
                        return result
                    return call
                def prefill(original):
                    def call(*a, **kw):
                        for name, value in (('prefix.mask_sorted', a[2]), ('prefix.cos_sorted', a[3]), ('prefix.sin_sorted', a[4])):
                            _rec(tensors, meta, name, value)
                        result = original(*a, **kw)
                        hold['S_p'] = a[0].shape[1]
                        for name, value in (('k', a[5]), ('v', a[6])):
                            record_native_cache(tensors, meta, 'prefix.' + name, value, hold['perm'], frontend.n_kv)
                        return result
                    return call
                def velocity(original):
                    def call(*a, **kw):
                        index = hold.get('step', 0)
                        _rec(tensors, meta, f'step{index}.latent_before', a[0])
                        _rec(tensors, meta, f'step{index}.time', torch.tensor([frontend._embedding_times[index]]))
                        meta[f'step{index}.time']['source'] = 'frontend_schedule_reconstruction_not_observed_scalar'
                        _rec(tensors, meta, f'step{index}.time_embedding', a[1])
                        if folded and index == 0:
                            hold['snapshot'] = controls.snapshot_velocity_inputs(a)
                            hold['snapshot']['perm'] = hold['perm'].copy()
                        if index == 0:
                            for name, value in (('suffix.mask_sorted', a[3]), ('suffix.cos', a[4]), ('suffix.sin', a[5])):
                                _rec(tensors, meta, name, value)
                        result = original(*a, **kw)
                        _rec(tensors, meta, f'step{index}.velocity', result)
                        hold['args'], hold['step'] = a, index + 1
                        return result
                    return call
                def merger(original):
                    def call(value):
                        _rec(tensors, meta, 'vision.merger_input', value)
                        return original(value)
                    return call
                def state_embedding(original):
                    def call(state):
                        _rec(tensors, meta, 'prepared.state', state)
                        result = original(state)
                        _rec(tensors, meta, 'suffix.state_embedding', result)
                        return result
                    return call
                wrap_method(stack, frontend, '_preprocess_images', preprocess)
                wrap_method(stack, frontend, '_assemble_prefix', assemble)
                wrap_method(stack, pipe, 'prefill', prefill)
                wrap_method(stack, pipe, 'evaluate_velocity', velocity)
                wrap_method(stack, pipe, 'embed_state', state_embedding)
                wrap_method(stack, pipe, 'merger_forward', merger)
                result = backend.infer(observation, noise=data['noise'].copy())
            _rec(tensors, meta, 'prepared.tokens', torch.as_tensor(frontend._lang_tokens))
            _rec(tensors, meta, 'prepared.token_mask', torch.as_tensor(frontend._lang_masks))
            tensors['actions'] = torch.from_numpy(result['actions'][None].copy())
            if hold['step'] != 10:
                raise RuntimeError('Trace did not observe exactly ten raw non-integrating velocity evaluations')
            prior_folder = args.prior/variant if folded else args.prior/f'euler10-{variant}'
            with np.load(prior_folder / 'actions.npz', allow_pickle=False) as prior:
                prior_comparison = difference(result['actions'], prior['actions32'])
            baseline, invariant = _invariance(baseline, tensors)
            boundaries = {name: difference(_array(tensors[name]), _array(reference[name])) for name in
                ('prepared.images', 'prepared.state', 'prepared.tokens', 'prepared.token_mask',
                 'prefix.embeddings', 'prefix.pad_mask', 'prefix.modality_mask', 'prefix.k', 'prefix.v', 'suffix.state_embedding')}
            report = {'engine': 'native', 'variant': variant, 'prior_graph_output': prior_comparison,
                      'prefix_invariance': invariant, 'reference_boundaries': boundaries,
                      'job_id': os.environ['SLURM_JOB_ID'], 'torch': torch.__version__,
                      'tensor_metadata': meta, 'performance_measurement': False}
            if not folded:
                _save(folder, tensors, report)
            else:
                report.update(experimental_projection={'fold_spatial': True, 'effective_projection': 'per_camera_hwd',
                    'production_option': False, 'original_gate_changed': False, 'source_revision': args.source_revision,
                    'precondition': proof, 'override_source': validator._file_record(Path(ablation.__file__))},
                    historical_candidate_prior={**admitted[variant], 'path': str(prior_folder)},
                    historical_official_reference={'job_id': args.reference_job, 'path': str(args.reference/variant)},
                    config={'use_cuda_graph': False, 'sampler': 'euler', 'num_steps': 10, 'numerical_profile': 'official_local_fp32'},
                    runtime=runtime, checkpoint=checkpoint_record, model_sources=validator.model_sources(backend),
                    policy_qualification='unchanged_historical_failed_6_of_7_not_a_new_gate_result')
            if not prior_comparison['exact'] or not all(invariant.values()):
                raise RuntimeError('Native eager/prior-graph or prefix-invariance prerequisite failed; inspect trace before counterfactuals')
            def frozen_controls():
                if folded:
                    common=SimpleNamespace(expand_official_kv=expand_official_kv,_array=_array,difference=difference)
                    frozen, rows, evidence = controls.evaluate_controls(frontend, pipe, reference, tensors,
                        hold['snapshot'],common=common)
                    if probe_order:
                        from safetensors import safe_open
                        with safe_open(str(args.order_baseline/variant/'frozen_velocities.safetensors'),framework='pt',device='cpu') as saved:
                            baseline_tensors={name:saved.get_tensor(name) for name in saved.keys()}
                        baseline_check=order.admit_sorted_baseline(frozen,baseline_tensors)
                        mask_check=order.admit_original_mask(hold['snapshot']['smask'],reference['suffix.attention_mask'],hold['snapshot']['perm'])
                        order_state={}
                        context=lambda:order.suffix_original_order(frontend,hold['snapshot']['perm'],state=order_state)
                        changed,changed_rows,changed_evidence=controls.evaluate_controls(frontend,pipe,reference,tensors,
                            hold['snapshot'],common=common,frozen_context=context)
                        contrasts=[]
                        for step in range(10):
                            if not torch.equal(frozen[f'native_replay.step{step}.velocity'],changed[f'native_replay.step{step}.velocity']):
                                raise ValueError('Native replay changed before suffix-order intervention')
                            for condition in ('native_kv','official_kv_sorted'):
                                key=f'{condition}.step{step}.velocity'
                                contrasts.append({'step':step,'condition':condition,
                                    **difference(_array(changed[key]),_array(frozen[key]))})
                        for row in rows:row['suffix_key_order']='sorted_prefix'
                        for row in changed_rows:row['suffix_key_order']='original_prefix'
                        rows.extend(changed_rows)
                        frozen.update({'original_order.'+name:value for name,value in changed.items()})
                        report['suffix_order_control']={'proof':order_proof,'intervention':order_state,
                            'original_mask_admission':mask_check,
                            'sorted_baseline':{'path':str((args.order_baseline/variant).resolve()),'job_id':args.order_baseline_job,**baseline_check},
                            'original_order_evidence':changed_evidence,'order_contrasts':contrasts,
                            'scope':'Frozen official trajectory only; native replay outside intervention; unchanged prior policy gate'}
                    report['frozen_control_evidence'] = evidence
                    return frozen, rows
                args_live = list(hold['args'])
                conditions = {'native_kv': (args_live[6], args_live[7])}
                external = []
                for name, dtype in (('k', args_live[6].dtype), ('v', args_live[7].dtype)):
                    expanded = expand_official_kv(_array(reference['prefix.' + name]), hold['perm'], frontend.n_heads, 1 + frontend.chunk)
                    external.append(torch.from_numpy(expanded).to(device=frontend.device, dtype=dtype))
                conditions['official_kv_sorted'] = tuple(external)
                frozen, rows = {}, []
                for step in range(10):
                    latent = reference[f'step{step}.latent_before'].to(frontend.device)
                    time = float(reference[f'step{step}.time'].reshape(-1)[0])
                    time_emb = frontend._sinusoidal_time(time).to(device=frontend.device, dtype=torch.bfloat16)
                    for condition, (kbuf, vbuf) in conditions.items():
                        velocity = pipe.evaluate_velocity(latent, time_emb, args_live[2], args_live[3], args_live[4],
                            args_live[5], kbuf, vbuf, hold['S_p'])
                        _rec(frozen, meta, f'{condition}.step{step}.velocity', velocity)
                        rows.append({'step': step, 'time': time, 'condition': condition,
                                     **difference(_array(velocity), _array(reference[f'step{step}.velocity']))})
                return frozen, rows
            try:
                frozen, rows = admitted_frozen_evaluation(reference, tensors, data, official_report, variant, frozen_controls)
            except ReferenceInputMismatch as error:
                report['frozen_admission'] = {'status': 'rejected_reference_inputs', 'reason': str(error)}
                if not folded:
                    (folder / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                raise
            from safetensors.torch import save_file
            if not folded:
                save_file(frozen, str(folder / 'frozen_velocities.safetensors'))
            report.update(frozen_admission={'status': 'matched_exact_inputs'}, frozen_velocity_comparisons=rows, tensor_metadata=meta,
                          caveat='Official KV is permuted to native sorted reduction order; this does not eliminate attention-order differences')
            if folded:
                pending.append({'folder': folder, 'tensors': tensors, 'frozen': frozen, 'report': report})
            else:
                (folder / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
                print(json.dumps({'variant': variant, 'prior_graph_exact': True, 'boundaries': boundaries, 'frozen_velocities': rows}), flush=True)
    if folded:
        controls.publish_after_cleanup(pending, scope_state, publish_folded_trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('official', 'native'), required=True)
    for name in ('checkpoint', 'fixtures', 'output', 'official-source', 'observer-source'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--folded-proj1', action='store_true', help='Explicit HWD candidate eager diagnosis, never a policy/default change')
    parser.add_argument('--capture-eager-attention',action='store_true',help='Observe actual official eager layer0 calls, with historical exact admission')
    parser.add_argument('--probe-suffix-key-order',action='store_true',help='Folded eager frozen-trajectory prefix-order intervention only')
    parser.add_argument('--order-preflight-report',type=Path)
    parser.add_argument('--order-baseline',type=Path)
    parser.add_argument('--order-baseline-job')
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--source-revision')
    parser.add_argument('--prior-job')
    parser.add_argument('--reference-job')
    parser.add_argument('--variants', nargs='+', choices=VARIANTS)
    args = parser.parse_args(argv)
    require_step()
    args.variants = validate_trace_options(args)
    if args.engine == 'native' and (args.reference is None or args.prior is None):
        parser.error('native requires fresh --reference and --prior recorded graph outputs')
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.official_source))
    import torch
    if tuple(torch.cuda.get_device_capability()) != (8, 0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('This diagnostic is restricted to the recorded A800 profile')
    torch.set_num_threads(2)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    with torch.inference_mode():
        (run_official if args.engine == 'official' else run_native)(args, torch)


if __name__ == '__main__':
    main()
