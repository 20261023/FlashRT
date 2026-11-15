"""Direct author OpenPI/FastWAM A800 latency, without FlashRT model execution."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('pi0','pi05','fastwam'),required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--source-revision',required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--workload',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backbone-dir',type=Path)
    p.add_argument('--model-config',type=Path)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--repeats',type=int,required=True)
    args=p.parse_args(argv)
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_STEP_ID','').isdigit():
        raise RuntimeError('Run the actual benchmark inside an allocated GPU srun step')
    args.output.mkdir(parents=True,exist_ok=False)
    import numpy as np
    import torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available() or 'A800' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This comparison requires the allocated A800')
    manifest=json.loads(args.workload.read_text())
    requests=[]
    for spec in manifest['requests']:
        path=Path(spec['npz_path'])
        if not path.is_absolute():path=args.workload.parent/path
        with np.load(path,allow_pickle=False) as data:
            obs={key:([data[n].copy() for n in source] if isinstance(source,list) else data[source].copy()) for key,source in spec['fields'].items()}
            requests.append((spec,obs,data[spec['noise_key']].copy()))
    load_start=time.perf_counter_ns()
    if args.model in ('pi0','pi05'):
        sys.path.insert(0,str(args.source/'src'))
        from openpi.models import model as author_model
        from openpi.models.pi0_config import Pi0Config
        from openpi.policies.policy import Policy
        from openpi.training.config import ModelTransformFactory
        import jax
        config=Pi0Config(pi05=args.model=='pi05',action_horizon=50,
            discrete_state_input=args.model=='pi05')
        if args.model=='pi0':
            model=config.load(author_model.restore_params(args.checkpoint/'params',dtype='bfloat16'))
            if jax.devices()[0].platform!='gpu':raise RuntimeError('Author JAX model must execute on GPU, not CPU')
            weight_adapter='Original author Orbax checkpoint loader; BF16'
        else:
            from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
            from safetensors.torch import load_model
            model=PI0Pytorch(config)
            namespace=torch.nn.Module()
            namespace.add_module('model',model)
            load_model(namespace,str(args.checkpoint/'model.safetensors'),strict=True,device='cpu')
            model.paligemma_with_expert.to_bfloat16_for_selected_params('bfloat16')
            weight_adapter='Resident LeRobot H50 checkpoint; outer model. loading namespace only, safetensors shared-weight-aware strict load; original author model and mixed BF16/FP32 policy'
        policy=Policy(model,transforms=ModelTransformFactory()(config).inputs,
            is_pytorch=args.model=='pi05',pytorch_device='cuda')
        def call(index,steps):
            spec,obs,noise=requests[index]
            rgb=obs['images']
            names=('base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb')
            inputs={'image':{name:rgb[i] for i,name in enumerate(names)},
                'image_mask':{name:np.True_ for name in names},'state':obs['state'],
                'prompt':spec['prompt']}
            policy._sample_kwargs={'num_steps':steps}
            return np.asarray(policy.infer(inputs,noise=noise.copy())['actions'],dtype=np.float32)
        modes=[('author-default-steps',10),('matched-selected-steps',2 if args.model=='pi0' else 8)]
        implementation='author OpenPI '+('JAX module_jit' if args.model=='pi0' else 'PyTorch default max-autotune compile')
        author_module=inspect.getfile(type(model))
        extra={'jax':jax.__version__,'config':{'action_horizon':50,'max_token_len':config.max_token_len,
            'discrete_state_input':config.discrete_state_input,'dtype':config.dtype},'weight_adapter':weight_adapter,
            'input_scope':'Three supplied camera slots, frontend-ready normalized state, author tokenizer/resize/padding; raw padded CPU output, no robot decoding'}
    else:
        sys.path.insert(0,str(args.source/'src'))
        helper_path=Path(__file__).with_name('validate_fastwam.py')
        spec=importlib.util.spec_from_file_location('native_fastwam_loading',helper_path)
        helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
        cfg=helper.resolved_model_config(args.source,args.backbone_dir,args.model_config)
        model=helper.load_upstream_reference(SimpleNamespace(checkpoint=args.checkpoint,backbone_dir=args.backbone_dir),cfg,torch)
        def call(index,steps):
            spec,obs,noise=requests[index]
            seed=int(spec['request_id'].split('-',1)[1]) if spec['request_id'].startswith('noise-') else 20261109
            out=model.infer_action(prompt=None,action_horizon=32,num_inference_steps=steps,
                sigma_shift=5.,seed=seed,rand_device='cpu',tiled=False,compile_action_infer=True,
                **{key:torch.from_numpy(value) for key,value in obs.items()})
            return out['action'].float().cpu().numpy()[None]
        modes=[('author-default-steps',10),('matched-selected-steps',20)]
        implementation='author FastWAM.infer_action, author compile_action_infer=True'
        author_module=inspect.getfile(type(model))
        extra={'sigma_shift':5.,'action_horizon':32,'text_scope':'cached-T5, same supplied contexts/masks; text encoder not timed',
            'noise_scope':'Original author seeded Torch CPU generator; same documented fixture seeds, no custom sampler'}
    torch.cuda.synchronize()
    load_ms=(time.perf_counter_ns()-load_start)/1e6
    gpu={'name':torch.cuda.get_device_name(0),'uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid',''))}
    status=0
    for mode,steps in modes:
        target=args.output/mode;target.mkdir()
        samples=[];actions=[];ids=[]
        try:
            with torch.inference_mode():
                cold_start=time.perf_counter_ns();call(0,steps);torch.cuda.synchronize()
                cold_ms=(time.perf_counter_ns()-cold_start)/1e6
                for i in range(args.warmup):call(i%len(requests),steps)
                torch.cuda.synchronize()
                for i in range(args.repeats):
                    index=i%len(requests);torch.cuda.synchronize();started=time.perf_counter_ns()
                    action=call(index,steps);torch.cuda.synchronize()
                    samples.append((time.perf_counter_ns()-started)/1e6);actions.append(action.copy());ids.append(requests[index][0]['request_id'])
            np.savez_compressed(target/'outputs.npz',actions=np.stack(actions),latency_ms=np.asarray(samples),request_ids=np.asarray(ids))
            result={'status':'complete','origin':'author','model':args.model,'mode':mode,'implementation':implementation,
                'author_source_revision':args.source_revision,'author_model_file':author_module,
                'checkpoint':str(args.checkpoint),'job_id':os.environ['SLURM_JOB_ID'],'gpu':gpu,
                'runtime':{'torch':torch.__version__,'cuda':torch.version.cuda,'python':sys.version,'executable':sys.executable},
                'load_ms':load_ms,'cold_compile_call_ms':cold_ms,'warmup':args.warmup,'repeats':args.repeats,
                'nfe':steps,'NFE_evidence':'Fixed original author Euler loop/config; no additional NFE probes requested',
                'precision':'bf16','latency_ms':samples,'request_ids':ids,'action_shape':list(actions[0].shape),
                'statistics':{'p50_ms':float(np.median(samples)),'p95_ms':float(np.percentile(samples,95)),'mean_ms':float(np.mean(samples))},
                'boundary':'host_observation_noise_to_CPU_raw_chunk',**extra}
            (target/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
            print(json.dumps({'model':args.model,'mode':mode,'statistics':result['statistics'],'steps':steps}),flush=True)
        except Exception as error:
            import traceback
            (target/'failure.txt').write_text(traceback.format_exc())
            (target/'failure.json').write_text(json.dumps({'status':'failed','error':str(error)},indent=2)+'\n')
            print(f'{args.model} {mode} failed: {type(error).__name__}: {error}',flush=True);status=1
    return status


if __name__=='__main__':raise SystemExit(main())
