"""Original openpi policy loader and request contract."""

from pathlib import Path
import sys
import json
from .common import _cameras, _prompt


def load(spec, torch):
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


def prepare_environment(spec):
    if spec["model"] == "pi0":
        import os
        os.environ["JAX_PLATFORMS"] = "cuda"
