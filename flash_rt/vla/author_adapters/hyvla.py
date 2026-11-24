"""Original hyvla policy loader and request contract."""

from pathlib import Path
import sys
from .common import _cameras, _prompt, example


def load(spec, torch):
    """Original prepare_images/state/language and model.sample_actions, as in official_hyvla_latency."""
    import inspect
    import numpy as np
    from safetensors.torch import load_model
    if not torch.__version__.startswith('2.7.1'):
        raise ValueError('This original Hy profile requires its recorded author Torch 2.7.1 environment')
    sys.path.insert(0, spec['source'])
    import hy_vla
    from hy_vla import HyVLA, HyVLAConfig
    validator = example('validate_hyvla.py')
    checkpoint_config, _ = validator.checkpoint_record(spec['checkpoint'])
    paths = {'package': hy_vla.__file__, 'policy': inspect.getfile(HyVLA), 'config': inspect.getfile(HyVLAConfig)}
    admit_author(spec['source'], paths, missing=[], unexpected=[])
    policy = HyVLA(HyVLAConfig.from_pretrained(spec['checkpoint'], local_files_only=True))
    missing, unexpected = load_model(policy, str(Path(spec['checkpoint']) / 'model.safetensors'),
                                    strict=False, device='cpu')
    admit_author(spec['source'], paths, missing=missing, unexpected=unexpected)
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
        return author_request(policy, arrays)

    return policy, infer, {
        'implementation': 'author HyVLA prepare_images/state/language + model.sample_actions',
        'noise_contract': 'explicit model.sample_actions(noise=...)',
        'expected_shape': [40, 32], 'nfe_control_model': policy.model,
    }

AUTHOR_REVISION = 'af57e7507ec5964b52fdf6296741e553cbcd3288'
CAMERAS = ('top_head', 'hand_left', 'hand_right')


def file_record(path):
    p=Path(path).resolve();s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def admit_author(root,paths,*,missing,unexpected):
    root=Path(root).resolve()
    if (root/'AUTHOR_GIT_COMMIT').read_text().strip()!=AUTHOR_REVISION:raise ValueError('Official Native author source revision differs')
    expected={'package':root/'hy_vla/__init__.py','policy':root/'hy_vla/modeling_hy_vla.py','config':root/'hy_vla/configuration_hy_vla.py'}
    if set(paths)!=set(expected) or any(Path(paths[k]).resolve()!=p or not p.is_file() for k,p in expected.items()):
        raise ValueError('Official Native source/import path differs from exported author files')
    allowed={'model.dual_tower.expert.lm_head.weight'}
    if set(missing)-allowed or unexpected:raise ValueError('Official Native checkpoint has missing/unexpected inference weight keys')
    return {'author_revision':AUTHOR_REVISION,'source_records':{k:file_record(p) for k,p in expected.items()},
        'missing_inference_keys':[],'ignored_unused_keys':sorted(set(missing)&allowed)}


def author_request(policy,arrays,*,device='cuda'):
    import numpy as np
    import torch
    batch={'task':[str(arrays['prompt'].item())],
        'observation.state':torch.from_numpy(arrays['state'])[None].to(device=device,dtype=torch.bfloat16)}
    for camera in CAMERAS:
        batch['observation.images.'+camera]=torch.from_numpy(arrays[camera]).permute(0,3,1,2).contiguous()[None].float().div(255.).to(device=device,dtype=torch.bfloat16)
    images,masks=policy.prepare_images(batch);state=policy.prepare_state(batch);tokens,token_mask,_=policy.prepare_language(batch)
    noise=torch.from_numpy(np.array(arrays['noise'],copy=True))[None].to(device=device,dtype=torch.float32)
    output=policy.model.sample_actions(images=images,img_masks=masks,state=state,lang_tokens=tokens,lang_masks=token_mask,noise=noise)
    return output[0].float().cpu().numpy()


def check_nfe_once(model,infer,request,*,expected,synchronize):
    sentinel=object();owned=model.__dict__.get('denoise_step',sentinel);original=model.denoise_step;calls=[]
    def observed(*a,**kw):calls.append(True);return original(*a,**kw)
    try:
        model.denoise_step=observed;actions=infer(request);synchronize()
        if len(calls)!=expected:raise ValueError('Official Native untimed NFE control differs from fixed Euler budget')
        return {'observed_calls':len(calls),'timed':False,'actions':actions}
    finally:
        if owned is sentinel:model.__dict__.pop('denoise_step',None)
        else:model.denoise_step=owned
