"""Time unmodified author Hy APIs, not a FlashRT neural implementation."""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time

AUTHOR_REVISION='af57e7507ec5964b52fdf6296741e553cbcd3288'
CASES=('baseline','episode_start','current','history','prompt','state','noise')
CAMERAS=('top_head','hand_left','hand_right')


def _module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


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


def require_step():
    host=socket.gethostname().split('.')[0]
    step = os.environ.get('SLURM_STEP_ID', 'batch')
    node = os.environ.get('SLURMD_NODENAME') or os.environ.get('SLURM_JOB_NODELIST', '')
    if (not os.environ.get('SLURM_JOB_ID','').isdigit() or not (step == 'batch' or step.isdigit())
            or host.startswith(('ln','login')) or host != node.split('.')[0]
            or os.environ.get('CUDA_VISIBLE_DEVICES') in (None,'','-1')):
        raise RuntimeError('Official Native measurement requires an allocated GPU srun or sbatch compute node')


def save_complete(output,result,samples,*,control_actions):
    import numpy as np
    if samples.get('complete') is not True or len(samples['actions'])!=len(samples['latency_ms']):raise ValueError('Official Native measurement is incomplete')
    np.savez_compressed(Path(output)/'outputs.npz',actions=np.stack(samples['actions']),latency_ms=np.asarray(samples['latency_ms'],dtype=np.float64),
        request_indices=np.asarray(samples['request_indices']),control_actions=control_actions)
    temp=Path(output)/'result.json.tmp';temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temp.replace(Path(output)/'result.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--official-source',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--fixtures',type=Path,required=True);p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=8);p.add_argument('--repeats',type=int,default=70)
    args=p.parse_args(argv);require_step()
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact benchmark source revision required')
    marker=Path(__file__).resolve().parents[3]/'GIT_COMMIT'
    if not marker.is_file() or marker.read_text().strip()!=args.source_revision:raise ValueError('Benchmark frozen source marker differs')
    args.output.mkdir(parents=True,exist_ok=False);samples=None
    try:
        import numpy as np
        import torch
        from safetensors.torch import load_model
        if not torch.__version__.startswith('2.7.1'):raise ValueError('Official Hy requires recorded author Torch2.7.1 environment')
        if torch.cuda.device_count()!=1 or tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():
            raise ValueError('Official Hy latency requires one actual A800')
        torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        before=list(sys.path)
        try:
            sys.path.insert(0,str(args.official_source.resolve()))
            import hy_vla
            from hy_vla import HyVLA,HyVLAConfig
        finally:sys.path[:]=before
        source_paths={'package':hy_vla.__file__,'policy':inspect.getfile(HyVLA),'config':inspect.getfile(HyVLAConfig)}
        admit_author(args.official_source,source_paths,missing=[],unexpected=[])
        validator=_module(Path(__file__).with_name('validate_hyvla.py'),'official_latency_fixture_contract')
        cfg,checkpoint=validator.checkpoint_record(args.checkpoint)
        requests=[validator.load_fixture(args.fixtures/(case+'.npz')) for case in CASES]
        helpers=_module(Path(__file__).with_name('latency_comparison.py'),'official_latency_helpers')
        load_started=time.perf_counter_ns()
        policy=HyVLA(HyVLAConfig.from_pretrained(str(args.checkpoint),local_files_only=True))
        proof=admit_author(args.official_source,source_paths,missing=[],unexpected=[])
        missing,unexpected=load_model(policy,str(args.checkpoint/'model.safetensors'),strict=False,device='cpu')
        proof=admit_author(args.official_source,source_paths,missing=missing,unexpected=unexpected)
        policy.enable_video_encoder_if_needed();policy.eval().to(device='cuda',dtype=torch.bfloat16);torch.cuda.synchronize()
        if any(name.startswith(('flash_rt.models','flash_rt.frontends')) or name=='flash_rt.flash_rt_kernels' for name in sys.modules):
            raise ValueError('Official Native process must not use a FlashRT neural implementation')
        load_ms=(time.perf_counter_ns()-load_started)/1e6
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            infer=lambda request:author_request(policy,request)
            control=check_nfe_once(policy.model,infer,requests[0],expected=10,synchronize=torch.cuda.synchronize)
            samples=helpers.measure_calls(infer,requests,warmup=args.warmup,repeats=args.repeats,synchronize=torch.cuda.synchronize,expected_shape=(40,32))
        import flash_attn
        props=torch.cuda.get_device_properties(0)
        result={'kind':'author_HyVLA_official_Native_latency','status':'complete','origin':'author','job_id':os.environ['SLURM_JOB_ID'],
            'source_revision':args.source_revision,'author_proof':proof,'checkpoint':checkpoint,
            'runtime':{'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'torch_git':torch.version.git_version,
                'cuda':torch.version.cuda,'flash_attn':flash_attn.__version__},
            'gpu':{'name':props.name,'uuid':str(getattr(props,'uuid','')),'capability':[props.major,props.minor]},
            'model_class':type(policy).__module__+'.'+type(policy).__qualname__,'model_source':file_record(inspect.getfile(type(policy))),
            'load_ms':load_ms,'warmup':args.warmup,'repeats':args.repeats,'statistics':samples['statistics'],'cases':list(CASES),
            'latency_ms':samples['latency_ms'],'request_indices':samples['request_indices'],'action_shape':[40,32],
            'NFE_control':{k:v for k,v in control.items() if k!='actions'},'formal_NFE_interpretation':'Fixed author source/config10 verified by untimed10-call control; no timing hooks installed',
            'sampler':'euler','nfe':10,'precision':'bf16','boundary':'host_observation_noise_to_CPU_raw_chunk',
            'input_signature':{'batch':1,'cameras':3,'history':6,'image_hw':[480,640],'chunk':40,'action_dim':32},
            'source_url':'https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/tree/'+AUTHOR_REVISION,
            'fixture_records':{case:file_record(args.fixtures/(case+'.npz')) for case in CASES},
            'scope':'Original author prepare_images/state/language + sample_actions with fixed noise; not FlashRT, not action-queue hits, not a paper-reported timing or robot quality test.'}
        save_complete(args.output,result,samples,control_actions=control['actions'])
        print(json.dumps({'origin':'author','method':'official Native HyVLA','statistics':samples['statistics'],'nfe_control':control['observed_calls']}),flush=True)
    except BaseException as error:
        partial=getattr(error,'latency_partial',samples)
        if partial:
            import numpy as np
            arrays={f'action_{i}':v for i,v in enumerate(partial['actions']) if isinstance(v,np.ndarray)}
            arrays['latency_ms']=np.asarray(partial['latency_ms'],dtype=np.float64)
            try:np.savez_compressed(args.output/'unqualified_partial.npz',**arrays)
            except BaseException as secondary:error.add_note('Native evidence write failed: '+str(secondary))
        try:(args.output/'failure.json').write_text(json.dumps({'status':'failed','origin':'author','error_type':type(error).__name__,'error':str(error)},indent=2)+'\n')
        except BaseException as secondary:error.add_note('Native failure record write failed: '+str(secondary))
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
