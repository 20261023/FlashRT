"""Process-local camera-PROJ1 ablation; not a production Hy option or new gate.

The logical architecture, nominal cast profile, attention and solver stay fixed.
Only the first merger Linear's call batching changes. Never patch global
torch.nn.functional.linear or construct another policy's weights.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType


VARIANTS = ('baseline', 'episode_start', 'current', 'history', 'prompt', 'state', 'noise')
_PREFLIGHT_KIND = 'hy_camera_proj1_exact_frozen_precondition'
_MISSING = object()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sibling(name):
    return _module(Path(__file__).with_name(name + '.py'), 'hy_proj1_' + name)


def _layout(value):
    return {'shape': list(value.shape), 'live_stride': list(value.stride()),
            'storage_offset': value.storage_offset(), 'dtype': str(value.dtype)}


def per_camera_proj1(values, weight, bias, *, observer=None, fold_spatial=False):
    import torch
    import torch.nn.functional as F
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError('Camera PROJ1 requires nonempty BHWD input')
    if type(fold_spatial) is not bool:
        raise TypeError('fold_spatial must be bool')
    outputs = []
    for index in range(values.shape[0]):
        # HWD removes only singleton metadata: preserve strides/storage/offset.
        # Non-contiguous nD linear permits mm plus a separate bias addition;
        # do not flatten to a 2D+bias addmm or make a contiguous copy.
        view = values[index] if fold_spatial else values[index:index + 1]
        if observer is not None:
            observer(view)
        output = F.linear(view, weight, bias)
        outputs.append(output.unsqueeze(0) if fold_spatial else output)
    return torch.cat(outputs, dim=0)


def projection_controls(view, weight, bias, *, fold_spatial=False):
    """Frozen diagnostic arms; preserve the rejected B1/B3 controls and input."""
    from collections import Counter
    import torch
    import torch.nn.functional as F
    before = view.detach().cpu().clone()
    observed, b1_observed, lowering = [], [], {}
    def run(name, fn):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], acc_events=True) as profile:
            output = fn()
        lowering[name] = dict(Counter(event.name for event in profile.events()
            if event.name in ('aten::linear', 'aten::mm', 'aten::bmm', 'aten::add_', 'aten::addmm')))
        return output
    outputs = {'candidate': run('candidate', lambda: per_camera_proj1(view, weight, bias,
        fold_spatial=fold_spatial, observer=lambda item: observed.append(_layout(item))))}
    outputs['native_b1'] = run('native_b1', lambda: per_camera_proj1(view, weight, bias,
        observer=lambda item: b1_observed.append(_layout(item))))
    outputs['native_batch'] = run('native_batch', lambda: F.linear(view, weight, bias))
    if not torch.equal(view.detach().cpu(), before):
        raise RuntimeError('Camera projection mutated its frozen input')
    return outputs, {'candidate_slice_views': observed, 'native_b1_slice_views': b1_observed,
                     'input_unchanged': True, 'lowering': lowering}


def camera_merger(self, x, grid=14):
    """Source-body control: pipeline_thor.merger_forward, first Linear only."""
    import torch
    import torch.nn.functional as F
    W = self.W
    B = x.shape[0]
    h = w = grid
    x = x.reshape(B, h, w, -1)
    observed = []
    x = per_camera_proj1(x, W._mg_proj1_w, W._mg_proj1_b,
                        fold_spatial=getattr(self, '_camera_proj1_fold_spatial', False),
                        observer=lambda view: observed.append(_layout(view)))
    self._camera_proj1_encoding_views = observed
    # The production post-PROJ1 math remains batched, without casts/contiguous.
    C = x.shape[-1]
    new_x = (x.reshape(B, h // 2, 2, w // 2, 2, C)
             .permute(0, 1, 3, 2, 4, 5).reshape(B, h // 2, w // 2, 4, C))
    pooled = new_x.mean(-2, keepdim=True).expand(-1, -1, -1, 4, -1)
    fused = torch.cat([new_x, pooled], dim=-1)
    score = F.linear(fused, W._mg_pred0_w, W._mg_pred0_b)
    score = F.gelu(score)
    score = F.linear(score, W._mg_pred2_w, W._mg_pred2_b)
    x = (new_x * score.softmax(dim=-2)).sum(dim=-2)
    x = F.gelu(x)
    x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)
    return x.reshape(B, -1, C)


def _owned_graphs(frontend):
    seen = set()
    for name in ('_vit_graph_cache', '_graph_cache'):
        for entry in getattr(frontend, name, {}).values():
            graphs = [entry[key] for key in ('graph', 'prefix_graph') if key in entry]
            graphs.extend(entry.get('velocity_graphs', {}).values())
            for graph in graphs:
                if id(graph) not in seen:
                    seen.add(id(graph))
                    yield graph


@contextmanager
def scoped_backend(real_factory, config, *, arm, preflight, state, synchronize,
                   expected_frontend_type, fold_spatial=False, execution='graph'):
    if arm not in ('baseline', 'camera_proj1'):
        raise ValueError('Unknown ablation arm')
    if preflight.get('operator_exact') is not True:
        raise ValueError('Exact frozen projection precondition must pass before model construction')
    if type(fold_spatial) is not bool or preflight.get('fold_spatial', False) is not fold_spatial:
        raise ValueError('Projection precondition fold layout mode differs from requested mode')
    if execution not in ('graph', 'diagnostic_eager'):
        raise ValueError('Unknown ablation execution mode')
    if (config.model != 'hyvla' or config.sampler != 'euler' or config.num_steps != 10
            or config.options.get('numerical_profile') != 'official_local_fp32'
            or config.options.get('use_cuda_graph') is not (execution == 'graph')):
        raise ValueError('This ablation requires unchanged Hy official-local Euler10 and the admitted graph/execution mode')
    state['execution'] = execution
    state['cleanup'] = {'passed': False, 'status': 'loading'}
    backend = real_factory(config)
    frontend = pipe = None
    own = _MISSING
    own_fold = _MISSING
    primary = None
    state['cleanup'] = {'passed': False, 'status': 'active'}
    try:
        frontend = backend.frontend
        if not isinstance(frontend, expected_frontend_type) or frontend.numerical_profile != 'official_local_fp32':
            raise ValueError('Unexpected frontend/profile at ablation boundary')
        pipe = frontend.pipe
        own = pipe.__dict__.get('merger_forward', _MISSING)
        own_fold = pipe.__dict__.get('_camera_proj1_fold_spatial', _MISSING)
        if frontend._vit_graph_cache or frontend._graph_cache:
            raise ValueError('Ablation binding must precede every prepare/capture; graph caches must be empty')
        if arm == 'camera_proj1':
            pipe._camera_proj1_fold_spatial = fold_spatial
            pipe.merger_forward = MethodType(camera_merger, pipe)
        state['bound_before_prepare'] = True
        yield backend
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        reset_count = 0
        synchronized = False
        try:
            synchronize()
            synchronized = True
        except BaseException as error:
            errors.append(f'synchronize: {type(error).__name__}: {error}')
        if synchronized:
            try:
                graphs = list(_owned_graphs(frontend))
            except BaseException as error:
                errors.append(f'graph inventory: {type(error).__name__}: {error}')
                graphs = []
            for graph in graphs:
                try:
                    graph.reset()
                    reset_count += 1
                except BaseException as error:
                    errors.append(f'graph reset: {type(error).__name__}: {error}')
            if not errors:
                for name in ('_vit_graph_cache', '_graph_cache'):
                    try:
                        getattr(frontend, name, {}).clear()
                    except BaseException as error:
                        errors.append(f'graph cache clear: {type(error).__name__}: {error}')
        state['observed_projection_views'] = getattr(pipe, '_camera_proj1_encoding_views', [])
        try:
            if pipe is None:
                pass
            elif own is _MISSING:
                pipe.__dict__.pop('merger_forward', None)
            else:
                pipe.merger_forward = own
            if pipe is not None:
                if own_fold is _MISSING:
                    pipe.__dict__.pop('_camera_proj1_fold_spatial', None)
                else:
                    pipe._camera_proj1_fold_spatial = own_fold
        except BaseException as error:
            errors.append(f'method restore: {type(error).__name__}: {error}')
        try:
            backend.close()
        except BaseException as error:
            errors.append(f'backend close: {type(error).__name__}: {error}')
        state['cleanup'] = {'passed': not errors, 'status': 'passed' if not errors else 'failed',
                            'owned_graphs_reset': reset_count, 'errors': errors}
        if errors:
            # backend.close may drop its frontend; preserve the real owner and
            # its unreset graph/weight references until this isolated process exits.
            state.setdefault('failed_owners', []).append((backend, frontend, pipe))
            if primary is not None:
                primary.add_note('Ablation cleanup also failed: ' + '; '.join(errors))
            else:
                raise RuntimeError('Ablation cleanup failed: ' + '; '.join(errors))


def _runtime_record(record):
    if (record.get('runtime_checks_passed') is not True or record.get('finite') is not True
            or record.get('shape') != [40, 32] or record.get('public_shape') != [40, 20]
            or record.get('dtype') != 'float32'
            or record.get('action_domain') != 'normalized-padded-model-actions'
            or record.get('profile') != 'flashrt_official_local_fp32_hooks_not_exact'
            or record.get('fixed_noise_replay', {}).get('exact') is not True
            or record.get('sampler_metadata_replay_equal') is not True
            or record.get('graph_eager_check', {}).get('passed') is False):
        raise ValueError('Ablation child lacks completed mechanical/replay/finite evidence')
    cfg = record.get('config', {})
    if (cfg.get('sampler') != 'euler' or cfg.get('num_steps') != 10 or cfg.get('use_graph') is not True
            or cfg.get('numerical_profile') != 'official_local_fp32'):
        raise ValueError('Ablation child changed its requested profile/schedule')
    metadata = record.get('sampler_metadata', [])
    times = [1. + index * (-1. / 10) for index in range(10)]
    if len(metadata) < 2 or any(row.get('sampler') != 'euler' or row.get('scheduled_steps') != 10
        or row.get('nfe') != 10 or row.get('timesteps') != times for row in metadata):
        raise ValueError('Ablation child lacks actual Euler10 execution evidence')
    oracle = record.get('official_oracle', {})
    if oracle.get('eligible_same_schedule') is not True or oracle.get('gate') != {
            'atol': .01, 'rtol': .01, 'minimum_cosine': .999, 'legacy_gate_domain': 'normalized_padded_40x32'}:
        raise ValueError('Original oracle gate changed or is ineligible')
    for name, width in (('actions32', 32), ('actions20', 20)):
        metric = oracle.get(name, {})
        outside = metric.get('outside_tolerance')
        if (metric.get('shape') != [40, width] or metric.get('elements') != 40*width
                or type(outside) is not int or not 0 <= outside <= 40*width
                or metric.get('allclose') != (outside == 0)
                or metric.get('atol') != .01 or metric.get('rtol') != .01):
            raise ValueError('Original oracle metrics are incomplete or contradictory')
    passed = oracle['actions32']['allclose'] and oracle['actions32']['cosine'] >= .999
    if oracle.get('same_action_gate_passed') is not bool(passed):
        raise ValueError('Original oracle disposition contradicts its metrics')
    return bool(passed)


def classify_exit(returncode, record):
    passed = _runtime_record(record)
    if record.get('experimental_merger_proj1', {}).get('cleanup', {}).get('passed') is not True:
        raise ValueError('Ablation child did not complete owned cleanup')
    if returncode == 0 and passed and record.get('status') == 'runtime_and_recorded_official_gate_passed':
        return 'passed'
    if returncode == 3 and not passed and record.get('status') == 'runtime_passed_oracle_mismatch':
        return 'oracle_mismatch'
    raise ValueError('Unexpected child exit/status: not a verified numerical oracle mismatch')


def compatible_prior(current, arrays, prior, old_arrays):
    import numpy as np
    _runtime_record(current)
    _runtime_record(prior)
    for field in ('profile', 'official_reference_profile', 'config', 'checkpoint', 'fixture', 'runtime',
                  'shape', 'public_shape', 'dtype', 'action_domain', 'sampler_metadata'):
        if not current.get(field) or current.get(field) != prior.get(field):
            raise ValueError(f'Baseline prior {field} provenance differs')
    for field in ('noise', 'input_state', 'input_prompt', 'oracle_actions32', 'oracle_actions20'):
        if field not in arrays or field not in old_arrays or not np.array_equal(arrays[field], old_arrays[field]):
            raise ValueError(f'Baseline prior input/oracle differs: {field}')
    for field in ('actions32', 'actions20'):
        if not np.array_equal(arrays[field], old_arrays[field]):
            raise ValueError(f'Baseline prior actions differ: {field}')
    if not current.get('model_sources') or not prior.get('model_sources'):
        raise ValueError('Baseline/prior source provenance is missing')
    return {'passed': True, 'actions_exact': True, 'inputs_exact': True,
            'model_sources_same': current['model_sources'] == prior['model_sources'],
            'source_note': 'Snapshot path/stat differences are expected and reported, not forged equal; no content digest attestation',
            'current_model_sources': current['model_sources'], 'prior_model_sources': prior['model_sources']}


def aggregate(arms):
    expected = set(VARIANTS)
    if set(arms) != {'baseline', 'camera_proj1'} or any(set(rows) != expected for rows in arms.values()):
        raise ValueError('Ablation requires complete baseline/candidate fixture sets; cases are missing')
    counts = {}
    public = {}
    anchor = None
    for arm, rows in arms.items():
        counts[arm] = public[arm] = 0
        seen_paths = set()
        for variant, (returncode, record) in rows.items():
            outcome = classify_exit(returncode, record)
            extra = record['experimental_merger_proj1']
            fixture = record.get('fixture', {})
            path = fixture.get('path')
            if (not isinstance(path, str) or Path(path).stem != variant or path in seen_paths
                    or fixture.get('manifest', {}).get('variant', variant) != variant):
                raise ValueError('Ablation fixture identity is duplicated or differs from its variant key')
            seen_paths.add(path)
            proof = extra.get('precondition', {})
            job, source = record.get('job_id'), extra.get('source_revision')
            if (not job or not source or proof.get('kind') != _PREFLIGHT_KIND
                    or proof.get('operator_exact') is not True or proof.get('job_id') != job
                    or proof.get('source_revision') != source
                    or type(extra.get('fold_spatial', False)) is not bool
                    or extra.get('fold_spatial', False) is not proof.get('fold_spatial', False)):
                raise ValueError('Ablation execution proof/job/source identity differs')
            identity = (job, source, proof)
            if anchor is None:
                anchor = identity
            elif identity != anchor:
                raise ValueError('Ablation records do not share one admitted experiment')
            if extra.get('arm') != arm:
                raise ValueError('Ablation arm label differs from its actual run')
            if arm == 'baseline' and extra.get('baseline_prior', {}).get('passed') is not True:
                raise ValueError('Baseline has no compatible prior control')
            if arm == 'camera_proj1' and record.get('graph_eager_check', {}).get('passed') is not True:
                raise ValueError('Candidate graph/eager check is incomplete')
            if arm == 'camera_proj1':
                baseline = arms['baseline'][variant][1]
                for field in ('fixture', 'checkpoint', 'profile', 'config', 'runtime', 'model_sources',
                              'official_reference_profile', 'shape', 'public_shape', 'dtype', 'action_domain'):
                    if not record.get(field) or record[field] != baseline.get(field):
                        raise ValueError(f'Ablation candidate/baseline {field} identity differs')
            counts[arm] += outcome == 'passed'
            public[arm] += record['official_oracle']['public20_gate_passed'] is True
    return {'kind': 'isolated_merger_proj1_batching_not_robot_quality', 'complete': True,
            'baseline_gate_passes': counts['baseline'], 'candidate_gate_passes': counts['camera_proj1'],
            'baseline_public20_passes': public['baseline'], 'candidate_public20_passes': public['camera_proj1'],
            'exit_code': 0 if counts['camera_proj1'] == len(VARIANTS) else 1,
            'original_gate_unchanged': True, 'production_default_changed': False}


def _read_arrays(folder):
    import numpy as np
    with np.load(Path(folder)/'actions.npz', allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in ('actions32', 'actions20', 'noise', 'input_state',
                'input_prompt', 'oracle_actions32', 'oracle_actions20')}


def run_wrapped_validator(helper, vla_module, forwarded, *, arm, preflight, source_revision,
                          synchronize, expected_frontend_type, prior=None,
                          checkpoint_path=None, output_path=None, fold_spatial=False):
    state = {'cleanup': {'passed': False, 'status': 'not_constructed'}, 'failed_owners': []}
    real_factory, real_writer = vla_module.load_backend, helper.write_json
    def factory(config):
        if checkpoint_path is not None and Path(config.checkpoint).expanduser().resolve() != Path(checkpoint_path).expanduser().resolve():
            raise ValueError('Actual backend checkpoint differs from admitted projection precondition')
        return scoped_backend(real_factory, config, arm=arm, preflight=preflight, state=state,
                              synchronize=synchronize, expected_frontend_type=expected_frontend_type,
                              fold_spatial=fold_spatial)
    def publish(path, record):
        if output_path is not None and Path(path).resolve().parent != Path(output_path).expanduser().resolve():
            raise ValueError('Actual validator output differs from the admitted output directory')
        extra = {'arm': arm, 'only_changed_operator': 'merger.PROJ1 call batching' if arm == 'camera_proj1' else None,
                 'fold_spatial': fold_spatial,
                 'effective_projection': ('per_camera_hwd' if fold_spatial else 'per_camera_b1') if arm == 'camera_proj1' else 'native_b3',
                 'nominal_profile': 'official_local_fp32', 'production_option': False,
                 'logical_architecture_changed': False, 'attention_changed': False, 'original_gate_changed': False,
                 'source_revision': source_revision, 'source_file': str(Path(__file__).resolve()),
                 'source_stat': {'bytes': Path(__file__).stat().st_size, 'mtime_ns': Path(__file__).stat().st_mtime_ns},
                 'precondition': preflight, 'cleanup': state['cleanup'],
                 'observed_projection_views_at_python_encoding': state.get('observed_projection_views', []),
                 'view_note': 'Observed warmup/encoding inputs, not steady CPU-call or graph GPU-work counts'}
        record['experimental_merger_proj1'] = extra  # same dict printed by validator after first write
        if Path(path).name == 'result.json':
            if state['cleanup'].get('passed') is not True:
                raise RuntimeError('Ablation cannot publish success before completed cleanup')
            if arm == 'baseline' and prior is not None:
                old_record, _ = helper.read_saved_run(prior)
                extra['baseline_prior'] = compatible_prior(record, _read_arrays(Path(path).parent),
                                                            old_record, _read_arrays(prior))
            if arm == 'camera_proj1' and record.get('graph_eager_check', {}).get('passed') is not True:
                raise RuntimeError('Candidate graph/eager check did not complete')
        return real_writer(path, record)
    vla_module.load_backend, helper.write_json = factory, publish
    try:
        return helper.main(forwarded)
    finally:
        vla_module.load_backend, helper.write_json = real_factory, real_writer


def _checkpoint_file(checkpoint):
    checkpoint = Path(checkpoint).expanduser().resolve()
    return checkpoint/'model.safetensors' if checkpoint.is_dir() else checkpoint


def _checkpoint_stat(checkpoint):
    path = _checkpoint_file(checkpoint)
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def validate_precondition(proof, *, job_id, source_revision, checkpoint, fold_spatial=False):
    if (proof.get('kind') != _PREFLIGHT_KIND or proof.get('operator_exact') is not True
            or proof.get('job_id') != job_id or proof.get('source_revision') != source_revision
            or proof.get('checkpoint') != _checkpoint_stat(checkpoint)
            or type(fold_spatial) is not bool or proof.get('fold_spatial', False) is not fold_spatial):
        raise ValueError('Projection precondition job/source/checkpoint/fold mode admission failed')


def validate_forwarded_point(point, *, checkpoint, output):
    if Path(point.checkpoint).expanduser().resolve() != Path(checkpoint).expanduser().resolve():
        raise ValueError('Forwarded validator checkpoint differs from admitted checkpoint')
    if Path(point.output).expanduser().resolve() != Path(output).expanduser().resolve():
        raise ValueError('Forwarded validator output differs from admitted output')


def validate_profile_confirmation(summary):
    names = summary.get('cuda_kernel_names_in_region', {})
    if summary.get('cuda_activity_observed') is not True or not names:
        raise ValueError('Actual CUDA kernel activity is unverified; filtered-profile confirmation incomplete')
    if 'frozen_auto_sdpa_no_timing_acceptance' in names:
        raise ValueError('GPU region annotation was still misclassified as a kernel')


def preflight_operator(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    common, ops = _sibling('diagnose'), _sibling('frozen_ops')
    if tuple(torch.cuda.get_device_capability()) != (8, 0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('This ablation precondition requires the recorded A800 target')
    accepted = json.loads((args.official_trace/'result.json').read_text())
    native = json.loads(args.native_report.read_text())
    if accepted.get('engine') != 'official' or accepted.get('oracle', {}).get('exact') is not True:
        raise ValueError('Projection precondition requires accepted official frozen values')
    if native.get('engine') != 'native' or native.get('frozen_admission', {}).get('status') != 'matched_exact_inputs':
        raise ValueError('Projection precondition requires admitted native layout metadata')
    args.output.mkdir(parents=True, exist_ok=False)
    with torch.inference_mode(), safe_open(str(args.official_trace/'trace.safetensors'), framework='pt', device='cpu') as frozen:
        values = [frozen.get_tensor(f'vision.proj1_input.{i}').cuda() for i in range(3)]
        strides = [frozen.get_tensor(f'vision.proj1_input_strides.{i}').tolist() for i in range(3)]
        _, view, _ = ops.projection_inputs(values, strides, native['tensor_metadata']['vision.merger_input'])
        weight, bias = frozen.get_tensor('vision.proj1_weight').cuda(), frozen.get_tensor('vision.proj1_bias').cuda()
        expected = torch.cat([frozen.get_tensor(f'vision.proj1.{i}') for i in range(3)]).cuda()
        with safe_open(str(_checkpoint_file(args.checkpoint)), framework='pt', device='cpu') as checkpoint:
            prefix = 'model.dual_tower.vlm.model.visual.merger.proj1.'
            if (not torch.equal(weight.cpu(), checkpoint.get_tensor(prefix+'weight').to(torch.bfloat16))
                    or not torch.equal(bias.cpu(), checkpoint.get_tensor(prefix+'bias').to(torch.bfloat16))):
                raise ValueError('Current checkpoint PROJ1 tensors differ from accepted frozen reference')
        outputs, evidence = projection_controls(view, weight, bias, fold_spatial=args.fold_spatial)
        exact = torch.equal(outputs['candidate'], expected)
        metrics = {}
        for name, value in outputs.items():
            metrics[name + '_vs_reference'] = common.difference(common._array(value), common._array(expected))
            metrics[name + '_vs_reference']['unequal_elements'] = int((value != expected).sum().item())
        save_file({name: value.cpu().contiguous() for name, value in {**outputs, 'reference': expected}.items()},
                  str(args.output/'operator_outputs.safetensors'))
        # A separate current-GPU check confirms annotation exclusion. The QKV
        # source remains explicitly historical, never disguised as this job.
        attention = _sibling('frozen_attention')
        qkv_path = args.profile_qkv/'operator_outputs.safetensors'
        with safe_open(str(qkv_path), framework='pt', device='cpu') as reference:
            qkv = reference.get_tensor('qkv_per_camera').cuda()
        dispatch = attention._profile_auto_sdpa(qkv, heads=16, group_size=6)
        validate_profile_confirmation(dispatch)
    record = {'kind': _PREFLIGHT_KIND, 'operator_exact': exact, 'job_id': os.environ['SLURM_JOB_ID'],
              'fold_spatial': args.fold_spatial,
              'source_revision': args.source_revision, 'checkpoint': _checkpoint_stat(args.checkpoint),
              'official_trace': str(args.official_trace.resolve()), 'native_report': str(args.native_report.resolve()),
              **evidence, 'metrics': metrics, 'profile_dispatch': dispatch,
              'profile_qkv_historical_source': str(args.profile_qkv.resolve()),
              'model_constructed': False, 'original_gate_changed': False}
    (args.output/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record), flush=True)
    return 0 if exact else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--arm', choices=('preflight', 'baseline', 'camera_proj1'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--runner', type=Path, default=Path(__file__).resolve().parents[1]/'validate_hyvla.py')
    parser.add_argument('--official-trace', type=Path)
    parser.add_argument('--native-report', type=Path)
    parser.add_argument('--profile-qkv', type=Path)
    parser.add_argument('--preflight-report', type=Path)
    parser.add_argument('--prior', type=Path)
    parser.add_argument('--fold-spatial', action='store_true',
                        help='Prototype no-copy HWD views instead of native B1 slices; same exact precondition required')
    args, forwarded = parser.parse_known_args(argv)
    _sibling('diagnose').require_step()  # before Torch, model IO or output creation
    if args.arm == 'preflight':
        if forwarded or any(value is None for value in (args.official_trace, args.native_report, args.profile_qkv)):
            raise ValueError('Preflight requires frozen trace/native report/historical QKV and no model arguments')
        return preflight_operator(args)
    if args.preflight_report is None or (args.arm == 'baseline' and args.prior is None):
        raise ValueError('Full model arms require current exact precondition; baseline also requires prior control')
    proof = json.loads(args.preflight_report.read_text())
    validate_precondition(proof, job_id=os.environ['SLURM_JOB_ID'], source_revision=args.source_revision,
                          checkpoint=args.checkpoint, fold_spatial=args.fold_spatial)
    helper = _module(args.runner, 'hy_camera_validation_runner')
    actual = ['--checkpoint', str(args.checkpoint), '--output', str(args.output), *forwarded]
    point = helper.parse_args(actual)
    validate_forwarded_point(point, checkpoint=args.checkpoint, output=args.output)
    if (point.compare or point.sampler != 'euler' or point.num_steps != 10 or not point.use_graph
            or point.numerical_profile != 'official_local_fp32' or not point.require_oracle
            or (args.arm == 'camera_proj1' and not point.check_graph_eager)):
        raise ValueError('Ablation must keep original Euler10/profile/graph/oracle gate; candidate needs graph/eager check')
    import torch
    import flash_rt.vla as vla
    from flash_rt.frontends.torch.hyvla_probeflow import HyVLAFlowFrontend
    return run_wrapped_validator(helper, vla, actual, arm=args.arm, preflight=proof,
        source_revision=args.source_revision, synchronize=torch.cuda.synchronize,
        expected_frontend_type=HyVLAFlowFrontend, prior=args.prior,
        checkpoint_path=args.checkpoint, output_path=args.output, fold_spatial=args.fold_spatial)


if __name__ == '__main__':
    raise SystemExit(main())
