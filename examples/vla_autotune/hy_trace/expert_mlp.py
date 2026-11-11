"""Recorded Expert0 MLP observation and frozen GU controls; never a policy patch."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


def _file_record(path):
    path=Path(path).resolve();stat=path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


@contextmanager
def observe_mlp(mlp,tensors,metadata,*,num_steps=10):
    import torch
    modules=[mlp.gate_proj,mlp.up_proj,mlp.down_proj]
    if type(num_steps) is not int or num_steps<1 or any(m.bias is not None for m in modules):
        raise ValueError('MLP capture requires unbiased projections and positive step count')
    hidden,middle=modules[0].weight.shape[1],modules[0].weight.shape[0]
    if modules[1].weight.shape!=(middle,hidden) or modules[2].weight.shape!=(hidden,middle):
        raise ValueError('MLP projection geometry differs from gate/up/down contract')
    state={'complete':False,'steps':0,'events':0}
    handles=[]
    fields=('input','gate','up','activation','output')
    def record(field,value):
        index=state['events']%5;step=state['steps']
        width=hidden if field in ('input','output') else middle
        if (field!=fields[index] or step>=num_steps or value.ndim!=2 or value.shape[1]!=width
                or value.shape[0]<1 or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all()):
            raise ValueError('MLP event/input must follow recorded 2D BF16 suffix geometry and order')
        if field!='input' and value.shape[0]!=tensors[f'expert0.mlp.step{step}.input'].shape[0]:
            raise ValueError('MLP input rows changed within a step')
        name=f'expert0.mlp.step{step}.{field}'
        metadata[name]={'shape':list(value.shape),'live_stride':list(value.stride()),
                        'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}
        tensors[name]=value.detach().cpu().contiguous().clone()
        state['events']+=1
        if field=='output':state['steps']+=1
    def projection(field):
        def hook(module,args,output):
            current=tensors[f"expert0.mlp.step{state['steps']}.input"]
            if args[0].dtype!=current.dtype or not torch.equal(args[0].detach().cpu(),current):
                raise ValueError('MLP gate/up projection input differs from observed shared input')
            record(field,output)
        return hook
    try:
        handles.append(mlp.register_forward_pre_hook(lambda m,a:record('input',a[0])))
        handles.append(mlp.gate_proj.register_forward_hook(projection('gate')))
        handles.append(mlp.up_proj.register_forward_hook(projection('up')))
        handles.append(mlp.down_proj.register_forward_pre_hook(lambda m,a:record('activation',a[0])))
        handles.append(mlp.register_forward_hook(lambda m,a,o:record('output',o)))
        yield state
        if state['steps']!=num_steps or state['events']!=5*num_steps:
            raise ValueError('Incomplete MLP suffix step count')
        state['complete']=True
    finally:
        for handle in reversed(handles):handle.remove()


def mlp_controls(x,gate,up,down):
    import torch
    import torch.nn.functional as F
    if (x.ndim!=2 or gate.ndim!=2 or gate.shape!=up.shape or down.shape!=(gate.shape[1],gate.shape[0])
            or x.shape[1]!=gate.shape[1] or any(v.dtype!=torch.bfloat16 or not torch.isfinite(v).all() for v in (x,gate,up,down))):
        raise ValueError('MLP controls require finite 2D BF16 inputs and admitted gate/up/down geometry')
    result={}
    for name in ('separate','fused'):
        if name=='separate':g,u=F.linear(x,gate),F.linear(x,up)
        else:g,u=(x@torch.cat((gate,up),dim=0).t()).chunk(2,dim=-1)
        activation=F.silu(g)*u
        output=F.linear(activation,down) if name=='separate' else activation@down.t()
        result.update({name+'.gate':g,name+'.up':u,name+'.activation':activation,name+'.output':output})
    return result


def validate_mlp_capture(report,*,job_id,source_revision,variant,checkpoint,mlp_source):
    state=report.get('expert_mlp_capture',{})
    if (report.get('engine')!='official' or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision
            or report.get('variant')!=variant or report.get('checkpoint')!=checkpoint or state.get('complete') is not True
            or state.get('steps')!=10 or state.get('events')!=50 or state.get('module')!='expert.model.layers.0.mlp_v'
            or state.get('source')!=_file_record(mlp_source)):
        raise ValueError('MLP capture job/source/checkpoint/complete-step admission failed')


def publish(folder,tensors,report):
    from safetensors.torch import save_file
    save_file({name:v.detach().cpu().contiguous().clone() for name,v in tensors.items()},str(folder/'operator_outputs.safetensors'))
    temporary=folder/'result.json.tmp'
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(folder/'result.json')


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_mlp_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True)
    for name in ('reference','checkpoint','official-mlp-source','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    args=parser.parse_args(argv)
    common=_sibling('diagnose');common.require_step()
    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():
        raise RuntimeError('Frozen MLP controls require the recorded A800 target')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):
        raise RuntimeError('Use the recorded engine-specific Torch runtime')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    checkpoint=_sibling('proj1_camera_ablation')._checkpoint_stat(args.checkpoint)
    frozen=_sibling('frozen_eager_attention')
    attention_source=args.official_mlp_source.resolve().parents[1]/'modeling_dual_tower.py'
    keys=['model.dual_tower.expert.model.layers.0.mlp_v.'+name+'.weight' for name in ('gate_proj','up_proj','down_proj')]
    args.output.mkdir(parents=True,exist_ok=False)
    all_exact=True
    for variant in ('baseline','state'):
        report=json.loads((args.reference/variant/'result.json').read_text())
        frozen.admit_capture_report(report,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
            variant=variant,attention_source=attention_source)
        validate_mlp_capture(report,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,
            variant=variant,checkpoint=checkpoint,mlp_source=args.official_mlp_source)
        original=frozen.load_method(args.official_mlp_source,'HunYuanVLMoTMLP','forward')
        with safe_open(checkpoint['path'],framework='pt',device='cpu') as saved:
            weights=[saved.get_tensor(key).to('cuda') for key in keys]
        gate,up,down=weights
        owner=SimpleNamespace(gate_proj=lambda x:F.linear(x,gate),up_proj=lambda x:F.linear(x,up),
            down_proj=lambda x:F.linear(x,down),act_fn=F.silu)
        tensors,metrics,metadata={},{},{}
        case_exact=True
        with torch.inference_mode(),safe_open(str(args.reference/variant/'trace.safetensors'),framework='pt',device='cpu') as saved:
            for step in range(10):
                base=f'expert0.mlp.step{step}.'
                x=frozen.restore_layout(saved.get_tensor(base+'input'),report['tensor_metadata'][base+'input'],device='cuda')
                if x.shape!=(41,1024):raise ValueError('MLP capture must be Expert0 41row suffix input')
                before=x.clone();outputs=mlp_controls(x,*weights)
                outputs['official_api.output']=original(owner,x)
                if not torch.equal(x,before):raise ValueError('MLP control mutated the frozen input')
                for name,value in outputs.items():
                    field=name.rsplit('.',1)[1];expected=saved.get_tensor(base+field).to('cuda')
                    key=f'step{step}.{name}'
                    tensors[key]=value
                    metric=common.difference(common._array(value),common._array(expected))
                    metric['unequal_elements']=int((value!=expected).sum().item())
                    metrics[key+'_vs_recorded']=metric
                    metadata[key]={'shape':list(value.shape),'live_stride':list(value.stride()),'dtype':str(value.dtype)}
                    if name.startswith(('separate.','official_api.')):case_exact=case_exact and metric['exact']
                for field in ('gate','up','activation','output'):
                    a,b=outputs['fused.'+field],outputs['separate.'+field]
                    metrics[f'step{step}.fused_vs_separate.{field}']=common.difference(common._array(a),common._array(b))
                    metrics[f'step{step}.fused_vs_separate.{field}']['unequal_elements']=int((a!=b).sum().item())
        required=case_exact or args.engine!='official';all_exact=all_exact and required
        folder=args.output/variant;folder.mkdir()
        result={'kind':'recorded_expert0_mlp_same_input_gu_controls_not_policy_gate',
            'status':'diagnostic_complete' if required else 'recorded_operator_reproduction_failed',
            'engine':args.engine,'variant':variant,'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
            'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'original_and_separate_exact_to_recorded':bool(case_exact),'checkpoint':checkpoint,
            'selected_checkpoint_keys':keys,'model_constructed':False,'weights_loaded':3,'performance_measurement':False,
            'official_mlp_source':_file_record(args.official_mlp_source),'prototype_source':_file_record(__file__),
            'metrics':metrics,'tensor_metadata':metadata,'reference':str((args.reference/variant).resolve()),
            'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            'scope':'Expert0 on10observed normalized suffix inputs; no full-Expert/whole-policy causal fraction or new6of7 disposition'}
        publish(folder,tensors,result)
        print(json.dumps({'engine':args.engine,'variant':variant,'status':result['status'],
            'original_and_separate_exact_to_recorded':bool(case_exact)}),flush=True)
    return 0 if all_exact else 1


if __name__=='__main__':
    raise SystemExit(main())
