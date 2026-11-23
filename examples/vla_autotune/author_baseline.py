"""Produce bound original-author outputs for the ordinary VLA experiment loop.

See author_baseline.md. This is a harness companion, never a project backend.
Model imports/computation require an allocated GPU srun step. It reuses the
loaders and source APIs of the adjacent official_*_latency entrypoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
# The producer is executed by absolute filename from a new empty cwd.
sys.path.insert(0, str(HERE.parents[1]))


def _module(filename, name):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _require_step():
    host = socket.gethostname().split('.')[0]
    if (not os.environ.get('SLURM_JOB_ID', '').isdigit()
            or not os.environ.get('SLURM_STEP_ID', '').isdigit()
            or host.startswith(('ln', 'login'))
            or host != os.environ.get('SLURMD_NODENAME', '').split('.')[0]
            or os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1')):
        raise RuntimeError('Author model execution requires an actual allocated GPU srun compute step')


def _prompt(request):
    text = request.prompt if request.prompt is not None else request.observation.get('prompt')
    if hasattr(text, 'item'):
        text = text.item()
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Author request requires an explicit nonempty prompt')
    return text


def _cameras(request, names):
    images = request.observation.get('images')
    if isinstance(images, dict):
        images = [images[name] for name in names]
    elif images is None:
        images = [request.observation[name] for name in names]
    if len(images) != len(names):
        raise ValueError('Author request must supply exactly the configured cameras in order')
    return images


def _openpi(spec, torch):
    """Same OpenPI Config/Policy and checkpoint loading as official_native_latency."""
    import numpy as np
    sys.path.insert(0, str(Path(spec['source']) / 'src'))
    from openpi.models import model as author_model
    from openpi.models.pi0_config import Pi0Config
    from openpi.policies.policy import Policy
    from openpi.training.config import ModelTransformFactory
    options = spec['options']
    checkpoint = Path(spec['checkpoint'])
    config_path = checkpoint / 'config.json'
    checkpoint_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if not isinstance(checkpoint_config, dict):
        raise ValueError('OpenPI checkpoint config.json must be an object')
    declared = checkpoint_config.get('pi05')
    if declared is None and checkpoint_config.get('type') in ('pi0', 'pi05'):
        declared = checkpoint_config['type'] == 'pi05'
    if declared is not None and declared != (spec['model'] == 'pi05'):
        raise ValueError('OpenPI checkpoint variant differs from the requested author model')
    kwargs = {'pi05': spec['model'] == 'pi05',
              'dtype': 'bfloat16',
              'action_horizon': checkpoint_config.get('action_horizon', 50),
              'discrete_state_input': spec['model'] == 'pi05'}
    for key in ('paligemma_variant', 'action_expert_variant', 'action_dim',
                'max_token_len', 'discrete_state_input'):
        if checkpoint_config.get(key) is not None:
            kwargs[key] = checkpoint_config[key]
    for key in ('max_token_len', 'action_horizon'):
        if key in options:
            kwargs[key] = options[key]
    config = Pi0Config(**kwargs)
    if spec['model'] == 'pi0':
        import jax
        import jax.numpy as jnp
        params_path = checkpoint / 'params'
        if not params_path.is_dir():
            raise FileNotFoundError(f'Author OpenPI JAX requires its actual Orbax params directory: {params_path}')
        params = author_model.restore_params(params_path, restore_type=jax.Array, dtype=jnp.bfloat16)
        # The author loader checks all keys/shapes and installs the restored
        # pure parameter tree. Refuse unexpected leaves instead of intersecting
        # a different checkpoint/config with a newly initialized model.
        model = config.load(params, remove_extra_params=False)
        if jax.devices()[0].platform != 'gpu':
            raise RuntimeError('Author OpenPI JAX must execute on the allocated GPU')
        implementation = 'author OpenPI JAX module_jit'
        leaves = jax.tree.leaves(params)
        loading = {'loader': 'author restore_params -> BaseModelConfig.load',
                   'params_path': str(params_path), 'restore_dtype': 'bfloat16',
                   'remove_extra_params': False, 'parameter_leaves': len(leaves),
                   'parameter_elements': sum(int(np.prod(leaf.shape)) for leaf in leaves)}
        del params, leaves
    else:
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from safetensors import safe_open
        from safetensors.torch import load_model
        model = PI0Pytorch(config)
        weights = checkpoint / 'model.safetensors'
        with safe_open(str(weights), framework='pt', device='cpu') as saved:
            keys = list(saved.keys())
        if not keys:
            raise ValueError('OpenPI safetensors checkpoint has no actual parameters')
        prefixed = [key.startswith('model.') for key in keys]
        if all(prefixed):
            target = torch.nn.Module()
            target.add_module('model', model)
            namespace = 'model.'
        elif any(prefixed):
            raise ValueError('OpenPI checkpoint mixes prefixed and unprefixed parameter namespaces')
        else:
            target, namespace = model, ''
        # load_model handles tied storage while strict=True forbids random
        # leftovers. Namespace selection comes from actual saved tensor keys.
        load_model(target, str(weights), strict=True, device='cpu')
        model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
        model.eval()
        implementation = 'author OpenPI PyTorch Policy default compilation'
        loading = {'loader': 'safetensors.torch.load_model', 'weights': str(weights),
                   'strict': True, 'saved_namespace': namespace, 'saved_tensor_keys': len(keys)}
    policy = Policy(model, transforms=ModelTransformFactory()(config).inputs,
                    is_pytorch=spec['model'] == 'pi05', pytorch_device='cuda')
    names = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')

    def infer(request):
        images = _cameras(request, names)
        inputs = {'image': dict(zip(names, images)),
                  'image_mask': {name: np.True_ for name in names},
                  'state': request.observation['state'], 'prompt': _prompt(request)}
        policy._sample_kwargs = {'num_steps': spec['num_steps']}
        return np.asarray(policy.infer(inputs, noise=request.noise.copy())['actions'], dtype=np.float32)

    return model, infer, {'implementation': implementation, 'noise_contract': 'explicit Policy.infer(noise=...)',
                          'expected_shape': [config.action_horizon, config.action_dim],
                          'parameter_loading': loading,
                          'config': {'action_horizon': config.action_horizon,
                                     'action_dim': config.action_dim,
                                     'paligemma_variant': config.paligemma_variant,
                                     'action_expert_variant': config.action_expert_variant,
                                     'max_token_len': config.max_token_len,
                                     'discrete_state_input': config.discrete_state_input}}


def _fastwam(spec, torch):
    """Call the original infer_action; do not install a project sampler or kernels."""
    sys.path.insert(0, str(Path(spec['source']) / 'src'))
    helper = _module('validate_fastwam.py', 'author_fastwam_loaders')
    options = spec['options']
    backbone = Path(options['backbone_dir'])
    config = helper.resolved_model_config(spec['source'], backbone, options.get('model_config'))
    model = helper.load_upstream_reference(
        SimpleNamespace(checkpoint=Path(spec['checkpoint']), backbone_dir=backbone), config, torch)

    def infer(request):
        observation = request.observation
        if set(observation) != {'input_image', 'proprio', 'context', 'context_mask'}:
            raise ValueError('Author FastWAM requires the same four prepared host fields; T5 is outside this workload')
        output = model.infer_action(
            prompt=None, action_horizon=options.get('action_horizon', 32),
            num_inference_steps=spec['num_steps'], sigma_shift=options.get('sigma_shift', 5.0),
            seed=options['noise_seeds'][request.request_id], rand_device='cpu', tiled=False,
            compile_action_infer=options.get('compile_action_infer', True),
            **{key: torch.from_numpy(value) for key, value in observation.items()})
        return output['action'].float().cpu().numpy()[None]

    from flash_rt.vla.autotune.author import content_identity
    return model, infer, {
        'implementation': 'author FastWAM.infer_action',
        'expected_shape': [1, options.get('action_horizon', 32), config['action_dit_config']['action_dim']],
        'noise_contract': 'original seeded CPU RNG, verified against typed workload noise by an untimed observation',
        'resolved_model_config': config,
        'auxiliary_assets': {'vae': content_identity(backbone / 'Wan2.2_VAE.pth')},
    }


def _hyvla(spec, torch):
    """Original prepare_images/state/language and model.sample_actions, as in official_hyvla_latency."""
    import inspect
    import numpy as np
    from safetensors.torch import load_model
    if not torch.__version__.startswith('2.7.1'):
        raise ValueError('This original Hy profile requires its recorded author Torch 2.7.1 environment')
    sys.path.insert(0, spec['source'])
    import hy_vla
    from hy_vla import HyVLA, HyVLAConfig
    helper = _module('official_hyvla_latency.py', 'author_hyvla_api')
    validator = _module('validate_hyvla.py', 'author_hyvla_checkpoint')
    checkpoint_config, _ = validator.checkpoint_record(spec['checkpoint'])
    paths = {'package': hy_vla.__file__, 'policy': inspect.getfile(HyVLA), 'config': inspect.getfile(HyVLAConfig)}
    helper.admit_author(spec['source'], paths, missing=[], unexpected=[])
    policy = HyVLA(HyVLAConfig.from_pretrained(spec['checkpoint'], local_files_only=True))
    missing, unexpected = load_model(policy, str(Path(spec['checkpoint']) / 'model.safetensors'),
                                    strict=False, device='cpu')
    helper.admit_author(spec['source'], paths, missing=missing, unexpected=unexpected)
    policy.enable_video_encoder_if_needed()
    policy.eval().to(device='cuda', dtype=torch.bfloat16)
    keys, names = validator.camera_order(checkpoint_config['image_features'])

    def infer(request):
        images = _cameras(request, keys)
        noise = np.asarray(request.noise)
        if noise.shape == (1, 40, 32):
            noise = noise[0]
        if noise.shape != (40, 32):
            raise ValueError('Author Hy requires the fixed [40,32] workload noise')
        arrays = {name: image for name, image in zip(names, images)}
        arrays.update(state=request.observation['state'], noise=noise, prompt=np.asarray(_prompt(request)))
        return helper.author_request(policy, arrays)

    return policy, infer, {
        'implementation': 'author HyVLA prepare_images/state/language + model.sample_actions',
        'noise_contract': 'explicit model.sample_actions(noise=...)',
        'expected_shape': [40, 32], 'nfe_control_model': policy.model,
    }


def _dm05(spec, torch):
    """Use the author's complete _prepare_model_input -> _predict public policy."""
    import numpy as np
    from PIL import Image
    sys.path.insert(0, spec['source'])
    from opendm.exp.dm05_exp import DM05Exp
    exp = DM05Exp(task='inference', logger_level='WARNING')
    exp.model_config.model_name_or_path = spec['checkpoint']
    exp.model_config.vision_attn_implementation = 'sdpa'
    exp.model_config.liger_kernel = False
    exp.model_config.chunk_size = 50
    exp.inference_config.backend = 'default'
    exp.inference_config.diffusion_steps = 10
    exp.inference_config.enable_vlm_chat = False
    exp.data_config.is_history = False
    exp._initialize_inference_runtime()
    policy = exp.inference_config

    def infer(request):
        os.environ['DM05_INFERENCE_SEED'] = str(spec['options']['noise_seeds'][request.request_id])
        images = _cameras(request, ('image', 'wrist_image', 'wrist_image_right'))
        converted = []
        for image in images:
            if isinstance(image, np.ndarray):
                if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                    raise ValueError('Author DM requires uint8 HWC RGB images, without implicit normalization')
                image = Image.fromarray(image)
            converted.append(image)
        state = np.asarray(request.observation['state'], dtype=np.float32)
        if state.shape != (14,):
            raise ValueError('Author DM requires the same absolute state[14]')
        data = policy._prepare_model_input(
            text=_prompt(request), images=converted, states=state.tolist(), robot_type='DOS W1')
        return np.asarray(policy._predict(data), dtype=np.float32)

    return policy.model, infer, {
        'implementation': 'author OpenDM DM05Exp default backend _prepare_model_input + _predict',
        'noise_contract': 'original DM05_INFERENCE_SEED CUDA RNG, verified against typed workload noise by an untimed observation',
        'expected_shape': [50, 14],
        'output_semantics': 'full denormalized reconstructed absolute DOS W1 CPU actions',
    }


def _check_seeded_noise(infer, request, torch):
    """Observe an actual author RNG draw outside samples; never replace its result.

    The authors' seeded APIs cannot accept arbitrary fixed noise. A seed/name
    match alone is insufficient: verify the author's actual typed draw. Sources
    that use another RNG API fail explicitly and need their own observer.
    """
    import numpy as np
    shape = tuple(request.noise.shape)
    shapes = {shape, (1, *shape)} if len(shape) == 2 else {shape}
    original, observed = torch.randn, []

    def recorded(*args, **kwargs):
        result = original(*args, **kwargs)
        if tuple(result.shape) in shapes:
            observed.append(result.detach().clone())
        return result

    try:
        torch.randn = recorded
        infer(request)
        torch.cuda.synchronize()
    finally:
        torch.randn = original
    if len(observed) != 1:
        raise ValueError(f'Author RNG control must observe exactly one initial draw; got {len(observed)}')
    draw = observed[0]
    expected = torch.as_tensor(request.noise).reshape(draw.shape).to(dtype=draw.dtype)
    if not np.array_equal(draw.float().cpu().numpy(), expected.float().cpu().numpy()):
        raise ValueError(f'Author seed produces different typed initial noise for request {request.request_id}')
    return {'request_id': request.request_id, 'dtype': str(draw.dtype), 'shape': list(draw.shape),
            'exact_typed_noise_match': True, 'timed': False, 'rng_output_replaced': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--spec-json', required=True)
    parser.add_argument('--workload', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--comparison-id', required=True)
    parser.add_argument('--campaign-instance-id', required=True)
    parser.add_argument('--bind-dm05-saved', type=Path,
                        help='CPU-only: bind original four-case official_dm05_latency outputs; never execute a model')
    args = parser.parse_args(argv)
    from flash_rt.vla.autotune.author import (
        BOUNDARY, _validate_spec, _write_json, content_identity, file_sha256,
        prove_author_model, source_identity, workload_identity)
    from flash_rt.vla.autotune.quality import output_domain
    from flash_rt.vla.autotune.workloads import NPZWorkload
    spec = _validate_spec(json.loads(args.spec_json))
    workload = NPZWorkload.from_file(args.workload)
    if workload.model != spec['model'] or args.warmup < 0 or args.repeats < len(workload):
        raise ValueError('Author producer requires matching model and a repeat budget covering the complete workload')
    if spec['model'] in ('fastwam', 'dm05') and set(spec['options']['noise_seeds']) != {
            request['request_id'] for request in workload.specs}:
        raise ValueError('Author seed map must cover exactly the workload requests')
    if args.bind_dm05_saved is not None:
        if spec['model'] != 'dm05':
            raise ValueError('--bind-dm05-saved requires model=dm05')
        from flash_rt.vla.autotune.author import bind_dm05_saved
        record = bind_dm05_saved(spec, workload, args.bind_dm05_saved, args.output)
        print(json.dumps({'origin': 'author', 'kind': record['kind'],
                          'repeats': record['repeats'], 'new_model_requests': 0}), flush=True)
        return 0
    _require_step()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        # Force local assets; never implicitly fetch a model during a campaign.
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        if spec['model'] == 'pi0':
            # The FlashRT workload environment uses JAX only to read weights
            # and pins it to CPU. This process executes the actual JAX policy.
            os.environ['JAX_PLATFORMS'] = 'cuda'
        import numpy as np
        import torch
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Author baseline requires one allocated BF16-capable CUDA GPU')
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        identity = {'model': spec['model'], 'workload': workload_identity(workload),
                    'checkpoint': content_identity(spec['checkpoint']),
                    'author_source': source_identity(spec), 'output_domain': output_domain(spec['model'])}
        started = time.perf_counter_ns()
        model, infer, metadata = {
            'pi0': _openpi, 'pi05': _openpi, 'hyvla': _hyvla,
            'fastwam': _fastwam, 'dm05': _dm05,
        }[spec['model']](spec, torch)
        torch.cuda.synchronize()
        load_ms = (time.perf_counter_ns() - started) / 1e6
        identity['author_model'] = prove_author_model(model, spec)
        if metadata.get('auxiliary_assets') is not None:
            identity['auxiliary_assets'] = metadata['auxiliary_assets']
        nfe_model = metadata.pop('nfe_control_model', None)
        controls = []
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            # Warm every request before formal samples, including when warmup=0,
            # just as worker.prepare initializes each workload request outside timing.
            for index in range(len(workload)):
                infer(workload.request(index))
            for index in range(args.warmup):
                infer(workload.request(index))
            torch.cuda.synchronize()
            if spec['model'] in ('fastwam', 'dm05'):
                controls = [_check_seeded_noise(infer, workload.request(index), torch)
                            for index in range(len(workload))]
            if nfe_model is not None:
                helper = _module('official_hyvla_latency.py', 'author_hyvla_nfe')
                control = helper.check_nfe_once(
                    nfe_model, infer, workload.request(0), expected=spec['num_steps'],
                    synchronize=torch.cuda.synchronize)
                controls.append({key: value for key, value in control.items() if key != 'actions'})
            actions, ids, latency = [], [], []
            expected_shape = tuple(metadata['expected_shape'])
            for index in range(args.repeats):
                request = workload.request(index)
                torch.cuda.synchronize()
                started = time.perf_counter_ns()
                action = infer(request)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if action.shape != expected_shape or action.dtype != np.float32 or not np.isfinite(action).all():
                    raise ValueError(f'Author action shape/domain differs: {action.shape}, expected {expected_shape}')
                latency.append(elapsed)
                actions.append(action.copy())
                ids.append(request.request_id)
        # Check the actual imported implementation again after warmup/compile.
        prove_author_model(model, spec)
        outputs = args.output.resolve() / 'outputs.npz'
        np.savez_compressed(outputs, actions=np.stack(actions), request_ids=np.asarray(ids),
                            latency_ms=np.asarray(latency, dtype=np.float64),
                            output_domain=np.asarray(identity['output_domain']))
        properties = torch.cuda.get_device_properties(0)
        record = {
            'kind': 'bound_author_baseline', 'status': 'complete', 'origin': 'author',
            'model': spec['model'], 'identity': identity, 'producer_spec': spec,
            'producer_files': {name: file_sha256(HERE / name) for name in (
                'author_baseline.py', 'official_native_latency.py', 'official_dm05_latency.py',
                'official_hyvla_latency.py', 'validate_fastwam.py', 'validate_hyvla.py')},
            'producer_command': (list(sys.orig_argv) if argv is None else
                                 [sys.executable, str(Path(__file__).resolve()), *argv]),
            'comparison_id': args.comparison_id, 'campaign_instance_id': args.campaign_instance_id,
            'source_url': identity['author_source']['source_url'],
            'author_source_revision': spec['source_revision'],
            'implementation': metadata['implementation'], 'checkpoint': spec['checkpoint'],
            'gpu': {'name': properties.name, 'uuid': str(getattr(properties, 'uuid', '')),
                    'capability': list(torch.cuda.get_device_capability())},
            'job_id': os.environ['SLURM_JOB_ID'], 'step_id': os.environ['SLURM_STEP_ID'],
            'runtime': {'python': sys.version, 'executable': sys.executable,
                        'torch': torch.__version__, 'cuda': torch.version.cuda},
            'warmup': args.warmup, 'untimed_prepare_requests': len(workload),
            'repeats': args.repeats, 'load_ms': load_ms, 'sampler': 'euler', 'nfe': spec['num_steps'],
            'NFE_evidence': 'Original author fixed Euler loop/config; Hy additionally has an untimed call count',
            'precision': 'bf16', 'boundary': BOUNDARY, 'request_ids': ids, 'latency_ms': latency,
            'timing_contract': {
                'noise_source': ('seeded-author-rng' if spec['model'] in ('fastwam', 'dm05')
                                 else 'provided-host-noise'),
                'cpu_actions': [{'field': 'actions', 'domain': identity['output_domain'],
                                 'shape': metadata['expected_shape'], 'dtype': np.dtype(np.float32).str}]},
            'statistics': {'p50_ms': float(np.median(latency)), 'p95_ms': float(np.percentile(latency, 95)),
                           'mean_ms': float(np.mean(latency))},
            'outputs_path': str(outputs), 'outputs_sha256': file_sha256(outputs),
            'controls': controls, 'metadata': metadata,
            'quality_interpretation': 'Author fixed-input numerical reference; no closed-loop task-success claim',
        }
        _write_json(args.output / 'result.json', record)
        print(json.dumps({'origin': 'author', 'model': spec['model'], 'statistics': record['statistics']}), flush=True)
    except Exception as error:
        import traceback
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        _write_json(args.output / 'failure.json', {'origin': 'author', 'status': 'failed',
                                                  'exception': type(error).__name__, 'message': str(error)})
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
