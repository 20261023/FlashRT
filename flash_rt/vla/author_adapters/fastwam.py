"""Original fastwam policy loader and request contract."""

from pathlib import Path
import sys
from types import SimpleNamespace
from .common import example


def load(spec, torch):
    """Call the original infer_action; do not install a project sampler or kernels."""
    sys.path.insert(0, str(Path(spec['source']) / 'src'))
    helper = example('validate_fastwam.py')
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


def verify_auxiliary(backend, result):
    from flash_rt.vla.autotune.author import content_identity, AuthorReferenceError
    expected_options = result["producer_spec"]["options"]
    for key, default in (("sigma_shift", 5.0), ("action_horizon", 32)):
        if backend.options.get(key, default) != expected_options.get(key, default):
            raise AuthorReferenceError(f"FastWAM {key} differs from author contract")
    vae = Path(backend.options["backbone_dir"]) / "Wan2.2_VAE.pth"
    if content_identity(vae)["sha256"] != result["identity"].get("auxiliary_assets", {}).get("vae", {}).get("sha256"):
        raise AuthorReferenceError("FastWAM VAE weights differ from author checkpoint")


def validate_options(options):
    if 'backbone_dir' not in options:
        raise ValueError('Author FastWAM requires a local backbone_dir')
