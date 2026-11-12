"""Frozen complete Merger stages; actual APIs anchor diagnostic reconstruction."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_merger_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


ablation=_sibling('proj1_camera_ablation');frozen=_sibling('frozen_eager_attention')
STAGES=('input','proj1','new_x','pooled','fused','predictor0','predictor_activation',
        'score','probabilities','products','pooled_sum','activation','proj2','merged')
WEIGHTS={f'{name}.{field}':f'_mg_{attr}_{suffix}' for name,attr in
    (('proj1','proj1'),('proj2','proj2'),('pooler.predictor.0','pred0'),('pooler.predictor.2','pred2'))
    for field,suffix in (('weight','w'),('bias','b'))}
CHECKPOINT_PREFIX='model.dual_tower.vlm.model.visual.merger.'


def file_record(path):
    path=Path(path).resolve();stat=path.stat()
    return {'path':str(path),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def _exact(a,b):
    import torch
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)


def _weights(weights):
    import torch
    if set(weights)!=set(WEIGHTS):raise ValueError('Merger weight role set differs')
    first=weights['proj1.weight']
    if first.ndim!=2:raise ValueError('Merger weight geometry differs')
    C,D=first.shape;shapes={'proj1.weight':(C,D),'proj1.bias':(C,),'proj2.weight':(C,C),'proj2.bias':(C,),
        'pooler.predictor.0.weight':(C,2*C),'pooler.predictor.0.bias':(C,),
        'pooler.predictor.2.weight':(C,C),'pooler.predictor.2.bias':(C,)}
    if min(C,D)<1 or any(v.dtype!=torch.bfloat16 or v.shape!=shapes[k] or v.device!=first.device or not torch.isfinite(v).all() for k,v in weights.items()):
        raise ValueError('Merger BF16 weight shape/device/finite contract differs')
    return D,C


def _input(x,weights):
    import torch
    D,C=_weights(weights)
    if (not isinstance(x,torch.Tensor) or x.ndim!=4 or min(x.shape)<1 or x.shape[-1]!=D
            or x.shape[1]!=x.shape[2] or x.shape[1]%2 or x.dtype!=torch.bfloat16
            or x.device!=weights['proj1.weight'].device or not torch.isfinite(x).all()):
        raise ValueError('Merger requires finite BF16 Tensor square even BHWD geometry')
    return C


def native_owner(weights):
    _weights(weights)
    return SimpleNamespace(W=SimpleNamespace(**{attr:weights[k] for k,attr in WEIGHTS.items()}),_camera_proj1_fold_spatial=True)


def official_owner(path,weights):
    import torch
    _weights(weights)
    forward=frozen.load_method(path,'_HYViT2MLPProjector','forward')
    pool=frozen.load_method(path,'_NormalizedDwPooler','forward')
    def linear(name):
        w,b=weights[name+'.weight'],weights[name+'.bias']
        layer=torch.nn.Linear(w.shape[1],w.shape[0],device='meta',dtype=torch.bfloat16)
        layer.weight=torch.nn.Parameter(w,requires_grad=False);layer.bias=torch.nn.Parameter(b,requires_grad=False)
        return layer
    pooler=type('FrozenOfficialPooler',(),{'__call__':pool})()
    pooler.predictor=torch.nn.Sequential(linear('pooler.predictor.0'),torch.nn.GELU(),linear('pooler.predictor.2'))
    owner=SimpleNamespace(proj1=linear('proj1'),proj2=linear('proj2'),act=torch.nn.GELU(),pooler=pooler)
    owner.forward=MethodType(forward,owner)
    return owner


def staged_merger(x,weights,*,hwd=False):
    import torch
    import torch.nn.functional as F
    C=_input(x,weights);B,H,W,D=x.shape;s={'input':x}
    s['proj1']=(ablation.per_camera_proj1(x,weights['proj1.weight'],weights['proj1.bias'],fold_spatial=True) if hwd
        else F.linear(x,weights['proj1.weight'],weights['proj1.bias']))
    s['new_x']=s['proj1'].reshape(B,H//2,2,W//2,2,C).permute(0,1,3,2,4,5).reshape(B,H//2,W//2,4,C)
    s['pooled']=s['new_x'].mean(-2,keepdim=True).expand(-1,-1,-1,4,-1)
    s['fused']=torch.cat([s['new_x'],s['pooled']],dim=-1)
    s['predictor0']=F.linear(s['fused'],weights['pooler.predictor.0.weight'],weights['pooler.predictor.0.bias'])
    s['predictor_activation']=F.gelu(s['predictor0'])
    s['score']=F.linear(s['predictor_activation'],weights['pooler.predictor.2.weight'],weights['pooler.predictor.2.bias'])
    s['probabilities']=F.softmax(s['score'],dim=-2);s['products']=s['new_x']*s['probabilities']
    s['pooled_sum']=s['products'].sum(dim=-2);s['activation']=F.gelu(s['pooled_sum'])
    s['proj2']=F.linear(s['activation'],weights['proj2.weight'],weights['proj2.bias'])
    s['merged']=s['proj2'].reshape(B,-1,C)
    return s,{k:ablation._layout(v) for k,v in s.items()}


def observe_merger(inputs,weights,*,form,grid=14,official=None):
    import torch
    if type(grid) is not int or grid<2 or grid%2:raise ValueError('Merger grid geometry unsupported')
    if form=='official':
        if not isinstance(inputs,tuple) or len(inputs)!=3 or official is None:raise ValueError('Merger official form requires three Tensor camera views and actual owner')
        stages=[];per=[];ends=[]
        for x in inputs:
            _input(x,weights)
            if x.shape[:3]!=(1,grid,grid):raise ValueError('Merger camera geometry differs')
            before=x.clone();endpoint=official.forward(x,size=(grid,grid));s,meta=staged_merger(x,weights)
            if not _exact(endpoint,s['merged']) or not _exact(x,before):raise ValueError('Merger reconstructed endpoint differs from actual official API or input changed')
            stages.append(s);per.append(meta);ends.append(endpoint)
        raw={k:torch.cat([s[k] for s in stages],0) for k in STAGES};raw['actual_endpoint']=torch.cat(ends,0)
        return raw,{'per_camera':per,'concatenation_is_artifact_only':True,'actual_endpoint_per_camera':[ablation._layout(v) for v in ends]}
    if form!='native_hwd':raise ValueError('Merger form must be official or native_hwd')
    if not isinstance(inputs,torch.Tensor) or inputs.ndim!=3 or inputs.shape[0]!=3 or inputs.shape[1]!=grid*grid:
        raise ValueError('Merger native form requires three-camera Tensor geometry')
    x=inputs.reshape(3,grid,grid,-1);_input(x,weights);owner=native_owner(weights);before=inputs.clone()
    endpoint=ablation.camera_merger(owner,inputs,grid=grid);s,meta=staged_merger(x,weights,hwd=True)
    if not _exact(endpoint,s['merged']) or not _exact(inputs,before):raise ValueError('Merger reconstructed endpoint differs from actual native API or input changed')
    return {**s,'actual_endpoint':endpoint},{'stages':meta,'actual_input_views':owner._camera_proj1_encoding_views,'actual_endpoint':ablation._layout(endpoint)}


def validate_raw(raw,*,batch=3,grid=14,in_channels=1152,out_channels=2048):
    import torch
    B,H,D,C=batch,grid,in_channels,out_channels;h=H//2
    shapes={'input':(B,H,H,D),'proj1':(B,H,H,C),'new_x':(B,h,h,4,C),'pooled':(B,h,h,4,C),'fused':(B,h,h,4,2*C),
        **{k:(B,h,h,4,C) for k in ('predictor0','predictor_activation','score','probabilities','products')},
        **{k:(B,h,h,C) for k in ('pooled_sum','activation','proj2')},'merged':(B,h*h,C),'actual_endpoint':(B,h*h,C)}
    expected={f'{o}.{f}.{k}':shape for o in ('gold','native') for f in ('official','native_hwd') for k,shape in shapes.items()}
    if set(raw)!=set(expected) or any(v.dtype!=torch.bfloat16 or v.shape!=expected[k] or not torch.isfinite(v).all() for k,v in raw.items()):raise ValueError('Merger raw role/shape/dtype/finite differs')
    for origin in ('gold','native'):
        for form in ('official','native_hwd'):
            base=origin+'.'+form+'.'
            if not _exact(raw[base+'actual_endpoint'],raw[base+'merged']):raise ValueError('Merger actual endpoint raw parity differs')
    return {'tensors':len(raw),'actual_endpoints_exact':4}


def _difference(a,b):
    common=_sibling('diagnose');result=common.difference(common._array(a),common._array(b))
    result['unequal_elements']=int((a!=b).sum());return result


def compare_stages(a,b):
    comparisons={k:_difference(a[k],b[k]) for k in STAGES}
    return {'first_different_stage':next((k for k in STAGES if not comparisons[k]['exact']),None),'stages':comparisons}


def rebuild_inputs(values,official_strides,native_metadata,*,device):
    import torch
    def restore(value,strides):
        shape=value.shape
        if len(strides)!=len(shape) or any(type(s) is not int or s<0 for s in strides):raise ValueError('Merger historical layout strides malformed')
        span=1
        for stride,size in sorted((s,n) for s,n in zip(strides,shape) if n>1):
            if stride<span:raise ValueError('Merger historical strides overlap')
            span+=stride*(size-1)
        if span>4*value.numel():raise ValueError('Merger historical layout span exceeds bound')
        out=torch.empty_strided(shape,tuple(strides),device=device,dtype=value.dtype);out.copy_(value.to(device));return out
    if len(values)!=3 or len(official_strides)!=3 or any(v.ndim!=4 or v.shape[0]!=1 or v.dtype!=torch.bfloat16 for v in values):raise ValueError('Merger historical camera layout geometry differs')
    combined=torch.cat(values,0);B,H,W,D=combined.shape
    if native_metadata['shape']!=[B,H*W,D] or native_metadata.get('dtype')!='torch.bfloat16':raise ValueError('Merger historical native layout geometry/dtype differs')
    native=restore(combined.reshape(B,H*W,D),native_metadata['live_stride'])
    official=tuple(restore(v,s) for v,s in zip(values,official_strides))
    return official,native,{'storage_offset_assumption':0,'historical_storage_offset_observed':False}


def expected_paths(task):
    task=Path(task);paths={}
    for origin,job,engine in (('gold','hy-trace-308690','official'),('native','hy-folded-state-trace-313766','native'),('current','hy-native-policy-tree-316975','native/control')):
        for case in ('baseline','state'):
            folder=task/'results'/job/engine/case
            if origin=='current':folder=folder/'replay0'
            paths.update({f'{origin}.{case}.report':folder/'result.json',f'{origin}.{case}.raw':folder/'trace.safetensors'})
    paths.update({'r33':task/'results/hy-native-policy-tree-316975/native/result.json',
        **{'runtime.'+e:task/'results/hy-rms-stages-315256'/e/'result.json' for e in ('official','native')}})
    return paths


def admit_file_records(manifest,*,factory):
    if manifest.get('kind')!='hy_frozen_complete_merger_inputs':raise ValueError('Merger input manifest kind differs')
    task=Path(manifest['task_root']).resolve();expected=expected_paths(task);records=manifest.get('records',{})
    if set(records)!=set(expected) or any(Path(records[k]['path']).resolve()!=p for k,p in expected.items()):raise ValueError('Merger actual input record role/path differs')
    source=manifest['official_source'];sp=Path(source['path']).resolve()
    if not sp.is_relative_to(task/'snapshots') or sp.name!='modeling_hunyuan_vl_mot.py':raise ValueError('Merger official source path differs')
    for record in (*records.values(),source,manifest['checkpoint']):
        if file_record(record['path'])!=record:raise ValueError('Merger input/source/checkpoint record stat changed')
    return factory()


def admit_endpoints(raw,*,engine,gold,native,current_bridge):
    if not _exact(native,current_bridge):raise ValueError('Historical native Merger endpoint differs from R33 image output bridge')
    key='gold.official.actual_endpoint' if engine=='official' else 'native.native_hwd.actual_endpoint'
    target=gold if engine=='official' else native
    if engine not in ('official','native') or not _exact(raw[key],target):raise ValueError('Merger historical runtime endpoint anchor differs')
    return {'historical_anchor_exact':True,'native_R33_output_bridge_exact':True,'anchor':key}


def admit_history_headers(gold,native,current,r33,*,checkpoint,variant):
    if (gold.get('engine')!='official' or gold.get('variant')!=variant or gold.get('job_id')!='308690'
            or gold.get('oracle',{}).get('exact') is not True
            or gold.get('load_report',{}).get('missing_inference_keys')!=[]
            or gold.get('load_report',{}).get('load_unexpected_keys')!=[]
            or set(gold.get('load_report',{}).get('load_missing_keys',[]))!={'model.dual_tower.expert.lm_head.weight'}):
        raise ValueError('Merger accepted Gold header/load/oracle differs')
    _sibling('expert_layers').admit_baseline_header(native,job_id='313766',source_revision='2959bd8710038c7dcdebdd17ac645b7feed4333f',checkpoint=checkpoint)
    if (native.get('variant')!=variant or r33.get('kind')!='hy_native_policy_expert_phase_source_tree_diagnostic'
            or r33.get('status')!='diagnostic_complete' or r33.get('job_id')!='316975'
            or r33.get('source_revision')!='54a661d52a7b3a9d4619eb4f3639ff5ae7f29e91'
            or r33.get('backend_cleanup',{}).get('passed') is not True
            or r33.get('precondition',{}).get('checkpoint')!=checkpoint or r33.get('checkpoint')!=native.get('checkpoint')
            or r33.get('runtime')!=native.get('runtime')
            or r33.get('original_padded_gate_passes')!={'control':6,'candidate':6}
            or current.get('status')!='request_captured_pending_backend_cleanup' or current.get('request_id')!=f'control.{variant}.r0'
            or current.get('scope',{}).get('enabled') is not False or current.get('scope',{}).get('complete') is not True
            or current.get('scope',{}).get('cleanup',{}).get('passed') is not True
            or current.get('scope',{}).get('nfe')!=10 or current.get('scope',{}).get('prefill_calls')!=1):
        raise ValueError('Merger native/history/current checkpoint/runtime/header binding differs')
    return {'historical_input_job':'313766','accepted_gold_job':'308690','current_output_bridge_job':'316975'}


def run_combinations(origins,official_strides,native_metadata,weights,*,official,device,grid=14):
    import torch
    raw={};details={};stages={}
    for origin in ('gold','native'):
        per,native,layout_proof=rebuild_inputs(origins[origin],official_strides,native_metadata,device=device)
        for form,value in (('official',per),('native_hwd',native)):
            captured,layouts=observe_merger(value,weights,form=form,grid=grid,official=official)
            key=origin+'.'+form;stages[key]=captured
            raw.update({key+'.'+n:v.detach().cpu().clone() for n,v in captured.items()})
            details[key]={'layouts':layouts,'reconstruction':layout_proof,'own_endpoint_exact':True}
    comparisons={}
    for origin in ('gold','native'):comparisons['forms.'+origin]=compare_stages(stages[origin+'.native_hwd'],stages[origin+'.official'])
    for form in ('official','native_hwd'):comparisons['origins.'+form]=compare_stages(stages['native.'+form],stages['gold.'+form])
    return raw,details,comparisons


def load_history(manifest,*,engine,runtime,torch_git):
    import torch
    from safetensors import safe_open
    admit_file_records(manifest,factory=lambda:None)
    paths={k:Path(v['path']) for k,v in manifest['records'].items()}
    docs={k:json.loads(p.read_text()) for k,p in paths.items() if not k.endswith('.raw')}
    ref=docs['runtime.'+engine];checkpoint=manifest['checkpoint'];task=Path(manifest['task_root']).resolve()
    if (ref.get('runtime')!=runtime or ref.get('torch_git_version')!=torch_git or ref.get('checkpoint')!=checkpoint
            or ref.get('job_id')!='315256' or ref.get('source_revision')!='b522e8fd04616c22b680a32a3c183a566c4088a4'
            or ref.get('status')!='diagnostic_complete' or ref.get('engine')!=engine):raise ValueError('Merger actual runtime/Git/checkpoint reference differs')
    layers=_sibling('expert_rms_stages');layers._record_files_unchanged(ref['source_records'],task/'snapshots')
    if Path(ref['source_records']['native']['path']).read_bytes()!=(Path(__file__).parents[3]/'flash_rt/models/hyvla/pipeline_thor.py').read_bytes():raise ValueError('Merger native source changed from admitted runtime')
    camera=task/'snapshots/round033/code/examples/vla_autotune/hy_trace/proj1_camera_ablation.py'
    if camera.read_bytes()!=Path(ablation.__file__).read_bytes():raise ValueError('Merger HWD camera source changed from R33')
    if (task/'results/hy-native-policy-tree-316975/SOURCE_GIT_COMMIT').read_text().strip()!='54a661d52a7b3a9d4619eb4f3639ff5ae7f29e91':raise ValueError('Merger R33 source marker differs')
    samples={};header_proofs={}
    for case in ('baseline','state'):
        gold_doc,native_doc,current_doc=(docs[o+'.'+case+'.report'] for o in ('gold','native','current'))
        header_proofs[case]=admit_history_headers(gold_doc,native_doc,current_doc,docs['r33'],checkpoint=checkpoint,variant=case)
        expected_source=Path(gold_doc['hy_vla_source']).parent/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
        if expected_source.resolve()!=Path(manifest['official_source']['path']).resolve():raise ValueError('Merger official source does not belong to accepted Gold snapshot')
        with safe_open(str(paths['gold.'+case+'.raw']),framework='pt',device='cpu') as f:
            gold={k:f.get_tensor(k) for k in [*(f'vision.proj1_input.{i}' for i in range(3)),*(f'vision.proj1_input_strides.{i}' for i in range(3)),
                'vision.proj1_weight','vision.proj1_bias','prefix.embeddings','prefix.modality_mask','prefix.pad_mask',
                'prepared.images','prepared.state','prepared.tokens','prepared.token_mask']}
        with safe_open(str(paths['native.'+case+'.raw']),framework='pt',device='cpu') as f:
            native={k:f.get_tensor(k) for k in ('vision.merger_input','vision.merged','prefix.embeddings','prefix.modality_mask','prefix.pad_mask',
                'prepared.images','prepared.state','prepared.tokens','prepared.token_mask')}
        with safe_open(str(paths['current.'+case+'.raw']),framework='pt',device='cpu') as f:
            current={k:f.get_tensor(k) for k in ('prefix.embeddings','prefix.modality_mask','prefix.pad_mask','prepared.state','prepared.tokens','prepared.token_mask')}
        if any(not torch.isfinite(v).all() for d in (gold,native,current) for v in d.values()):raise ValueError('Merger historical arrays must be finite')
        for k in ('prepared.images','prepared.state','prepared.tokens','prepared.token_mask','prefix.modality_mask','prefix.pad_mask'):
            if not _exact(gold[k],native[k]):raise ValueError('Merger historical prepared input pairing differs')
            if k in current and not _exact(native[k],current[k]):raise ValueError('Merger R33 current prepared/prefix masks differ from historical native')
        mask=gold['prefix.modality_mask']
        if mask.dtype!=torch.bool or mask.ndim!=2 or mask.shape[0]!=1 or int(mask.sum())!=147:raise ValueError('Merger expected three49image-token groups')
        g=gold['prefix.embeddings'][:,mask[0]].reshape(3,49,2048)
        bridge=current['prefix.embeddings'][:,mask[0]].reshape(3,49,2048)
        if native['vision.merger_input'].shape!=(3,196,1152) or native['vision.merged'].shape!=(3,49,2048) or not _exact(bridge,native['vision.merged']):raise ValueError('Merger native geometry/current R33 output bridge differs')
        values=[gold[f'vision.proj1_input.{i}'] for i in range(3)]
        if any(v.shape!=(1,14,14,1152) or v.dtype!=torch.bfloat16 for v in values) or g.dtype!=torch.bfloat16 or native['vision.merger_input'].dtype!=torch.bfloat16:
            raise ValueError('Merger historical BF16 input geometry differs')
        samples[case]={'gold_values':values,'gold_merged':g,'native_input':native['vision.merger_input'],'native_merged':native['vision.merged'],'bridge':bridge,
            'official_strides':[gold[f'vision.proj1_input_strides.{i}'].tolist() for i in range(3)],
            'native_metadata':native_doc['tensor_metadata']['vision.merger_input'],
            'proj1_weight':gold['vision.proj1_weight'],'proj1_bias':gold['vision.proj1_bias']}
    a,b=samples['baseline'],samples['state']
    for name in ('gold_merged','native_input','native_merged','bridge','proj1_weight','proj1_bias'):
        if not _exact(a[name],b[name]):raise ValueError('Merger baseline/state shared vision context differs')
    if a['official_strides']!=b['official_strides'] or a['native_metadata']!=b['native_metadata'] or any(not _exact(x,y) for x,y in zip(a['gold_values'],b['gold_values'])):
        raise ValueError('Merger baseline/state vision input/layout context differs')
    return a,{'header_proofs':header_proofs,'shared_baseline_state_vision_exact':True,'runtime_reference':ref,
        'source_stats':{'official':manifest['official_source'],'historical_camera':file_record(camera),'historical_native':ref['source_records']['native']}}


def publish(output,raw,report):
    from safetensors.torch import save_file
    save_file({k:v.detach().cpu().contiguous() for k,v in raw.items()},str(output/'operator_outputs.safetensors'))
    tmp=output/'result.json.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output/'result.json')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--engine',choices=('official','native'),required=True);p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);_sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors import safe_open
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact Merger producer source revision required')
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Frozen Merger requires recorded A800')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    manifest=json.loads(args.inputs.read_text());history,proof=load_history(manifest,engine=args.engine,runtime=runtime,torch_git=torch.version.git_version)
    if torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=proof['runtime_reference']['bf16_reduced_precision_reduction']:
        raise ValueError('Merger actual runtime reduction flag changed')
    weights={};selected=[]
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(manifest['checkpoint']['path'],framework='pt',device='cpu') as sf:
        for name in WEIGHTS:
            key=CHECKPOINT_PREFIX+name;v=sf.get_tensor(key)
            if v.dtype!=torch.bfloat16 or not torch.isfinite(v).all():raise ValueError('Merger checkpoint tensors must remain finite BF16')
            weights[name]=v.to('cuda');selected.append(key)
        if _weights(weights)!=(1152,2048):raise ValueError('Actual Merger checkpoint geometry differs')
        if not _exact(weights['proj1.weight'].cpu(),history['proj1_weight']) or not _exact(weights['proj1.bias'].cpu(),history['proj1_bias']):raise ValueError('Merger selected checkpoint PROJ1 differs from Gold captured weights')
        native_values=[v for v in history['native_input'].reshape(3,14,14,1152).split(1)]
        owner=official_owner(manifest['official_source']['path'],weights)
        raw,details,comparisons=run_combinations({'gold':history['gold_values'],'native':native_values},history['official_strides'],history['native_metadata'],weights,official=owner,device='cuda')
    raw_proof=validate_raw(raw)
    anchors=admit_endpoints(raw,engine=args.engine,gold=history['gold_merged'],native=history['native_merged'],current_bridge=history['bridge'])
    ref=proof.pop('runtime_reference');report={'kind':'hy_frozen_complete_merger_stages','status':'diagnostic_complete','engine':args.engine,
        'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch.version.git_version,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'input_manifest':manifest,'input_manifest_stat':file_record(args.inputs),'history_proof':proof,'anchors':anchors,'raw_proof':raw_proof,
        'stages':list(STAGES),'observations':details,'comparisons':comparisons,'selected_checkpoint_keys':selected,'weight_read_events':len(selected),
        'source_records':{n:file_record(f) for n,f in (('driver',__file__),('official',manifest['official_source']['path']),('camera',ablation.__file__),('AST_loader',frozen.__file__))},
        'model_constructed':False,'performance_measurement':False,'production_default_changed':False,
        'scope':'Frozenhistorical313766input andverifiedR33outputbridge only, no R33visioninputcapture/compiledbranch/errorfraction/policy/robot qualification. Zerooffset reconstruction notobservedoldoffset. Historical6/7FAILED unchanged.'}
    args.output.mkdir(parents=True,exist_ok=False);publish(args.output,raw,report)
    print(json.dumps({'engine':args.engine,'raw_proof':raw_proof,'anchors':anchors,'first_differences':{k:v['first_different_stage'] for k,v in comparisons.items()}}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
