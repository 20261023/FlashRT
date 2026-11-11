"""Observe native RMS calls, then characterize fixed-input arithmetic stages."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_rms_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


layers=_sibling('expert_layers')
PROBE_SPECS=(('baseline',0,1),('baseline',2,0),('state',0,1),('state',2,0))
PROBES=tuple(f'{case}.s{step}.l{layer}' for case,step,layer in PROBE_SPECS)
STAGES=('float_input','square','variance','variance_plus_epsilon','inverse_rsqrt','scaled','cast_back','weighted_output')


def rms_stages(x,weight,epsilon=1e-5):
    import torch
    if (epsilon!=1e-5 or x.ndim<2 or x.dtype not in (torch.bfloat16,torch.float32)
            or weight.dtype!=torch.bfloat16 or weight.shape!=(x.shape[-1],)
            or not torch.isfinite(x).all() or not torch.isfinite(weight).all()):
        raise ValueError('RMS stages require finite BF16/FP32 input, BF16 weight and fixed epsilon')
    value=x.float();square=value.pow(2);variance=square.mean(-1,keepdim=True)
    shifted=variance+epsilon;inverse=torch.rsqrt(shifted);scaled=value*inverse;cast=scaled.to(x.dtype)
    result=dict(zip(STAGES,(value,square,variance,shifted,inverse,scaled,cast,weight*cast)))
    if any(not torch.isfinite(v).all() for v in result.values()):raise ValueError('RMS intermediate is not finite')
    return result


def _descriptor(value):
    return {'shape':list(value.shape),'live_stride':list(value.stride()),'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}


@contextmanager
def capture_norms(owner,tensors,metadata):
    import torch
    original=owner._rms_norm;state={'calls':0,'complete':False}
    def call(value,weight):
        index=state['calls']
        if index>=4:raise ValueError('Extra RMS norm call outside recorded block contract')
        if value.dtype not in (torch.bfloat16,torch.float32) or weight.dtype!=torch.bfloat16:
            raise ValueError('Unexpected RMS input/weight dtype')
        name=f'n{index}';metadata[name]={'input':_descriptor(value),'weight':_descriptor(weight)}
        tensors[name+'.input']=value.detach().cpu().contiguous().clone()
        tensors[name+'.weight']=weight.detach().cpu().contiguous().clone()
        output=original(value,weight)
        metadata[name]['output']=_descriptor(output)
        tensors[name+'.observed_output']=output.detach().cpu().contiguous().clone()
        state['calls']+=1
        return output
    try:
        owner._rms_norm=call
        yield state
        if state['calls']!=4:raise ValueError('Incomplete RMS norm calls for actual block')
        state['complete']=True
    finally:owner._rms_norm=original


def admit_reference(r,*,job_id,source_revision,checkpoint,sources):
    if (r.get('kind')!='hy_fixed_input_rms_stages' or r.get('status')!='diagnostic_complete'
            or r.get('engine')!='official' or r.get('job_id')!=job_id or r.get('source_revision')!=source_revision
            or r.get('checkpoint')!=checkpoint or r.get('source_records')!=sources or r.get('probe_ids')!=list(PROBES)
            or r.get('block_oracles_exact') is not True or r.get('norm_calls')!=16
            or r.get('own_stage_replay_exact') is not True or not r.get('torch','').startswith('2.7.1')):
        raise ValueError('Current old-runtime RMS reference job/source/oracle admission failed')


def admitted_reference_owner(report,raw,*,job_id,source_revision,checkpoint,sources,factory):
    admit_reference(report,job_id=job_id,source_revision=source_revision,checkpoint=checkpoint,sources=sources)
    validate_reference_raw(raw,report.get('norm_metadata',{}))
    return factory()


def validate_reference_raw(raw,metadata):
    import torch
    roles={};shapes=((1,41,1024),(1,16,41,128),(1,4,41,128),(1,41,1024))
    dtypes=(torch.bfloat16,torch.float32,torch.float32,torch.bfloat16)
    for probe in PROBES:
        for field in ('residual','output'):roles[f'own.{probe}.block.{field}']=((1,41,1024),torch.bfloat16)
        for n,(shape,dtype) in enumerate(zip(shapes,dtypes)):
            base=f'own.{probe}.n{n}'
            roles.update({base+'.input':(shape,dtype),base+'.weight':((shape[-1],),torch.bfloat16),base+'.observed_output':(shape,dtype)})
            for stage in STAGES:
                reduced=stage in ('variance','variance_plus_epsilon','inverse_rsqrt')
                roles[base+'.stage.'+stage]=(shape[:-1]+(1,) if reduced else shape,dtype if stage in ('cast_back','weighted_output') else torch.float32)
    if set(raw)!=set(roles) or set(metadata)!=set(PROBES):raise ValueError('RMS reference role/key set differs')
    for name,(shape,dtype) in roles.items():
        value=raw[name]
        if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=dtype or not torch.isfinite(value).all():
            raise ValueError('RMS reference role shape/dtype/finite contract differs: '+name)
    for probe in PROBES:
        if set(metadata[probe])!={'n0','n1','n2','n3'}:raise ValueError('RMS reference metadata roles differ')
        for n in range(4):
            base=f'own.{probe}.n{n}';meta=metadata[probe][f'n{n}']
            for field,key in (('input','input'),('weight','weight'),('output','observed_output')):
                value=raw[base+'.'+key];desc=meta.get(field,{})
                strides=desc.get('live_stride');offset=desc.get('storage_offset')
                if (desc.get('shape')!=list(value.shape) or desc.get('dtype')!=str(value.dtype)
                        or not isinstance(strides,list) or len(strides)!=value.ndim
                        or any(type(s) is not int or s<0 for s in strides) or type(offset) is not int or offset<0):
                    raise ValueError('RMS reference live layout metadata differs')
                span=1
                for stride,size in sorted((s,n) for s,n in zip(strides,value.shape) if n>1):
                    if stride<span:raise ValueError('RMS reference layout overlaps')
                    span+=stride*(size-1)
                if offset+span>4*value.numel():raise ValueError('RMS reference layout span/offset exceeds bounded role')
            if not torch.equal(raw[base+'.stage.weighted_output'],raw[base+'.observed_output']):
                raise ValueError('RMS reference staged endpoint differs from actual observed norm')
    return {'complete':True,'tensors':len(raw),'observed_endpoints_exact':16}


def stage_comparison_summary(comparisons):
    if not comparisons or any(type(v.get('exact')) is not bool for v in comparisons.values()):
        raise ValueError('Runtime stage summary requires actual comparison results')
    count=sum(not v['exact'] for v in comparisons.values())
    return {'comparison_count':len(comparisons),'differing_stage_count':count,'runtime_stage_difference_observed':count>0}


def publish(folder,tensors,report):
    layers.publish(folder,tensors,report)


def _source_records():
    root=Path(__file__).resolve().parent
    return {name:layers._file_record(path) for name,path in (
        ('driver',__file__),('layers',layers.__file__),('native',root.parents[2]/'flash_rt/models/hyvla/pipeline_thor.py'),
        ('adapter',root/'suffix_key_order.py'),('layout',root/'frozen_eager_attention.py'))}


def _record_files_unchanged(records,allowed_root):
    for record in records.values():
        path=Path(record['path']).resolve()
        if not path.is_relative_to(Path(allowed_root).resolve()) or layers._file_record(path)!=record:
            raise ValueError('Historical record changed or escaped declared root')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--engine',choices=('official','native'),required=True)
    p.add_argument('--history-inputs',type=Path,required=True)
    p.add_argument('--history-job',default='315024')
    p.add_argument('--history-source-revision',default='5248e14b544aa8edc17b58f1b61186b5c08ca492')
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-rms',type=Path)
    args=p.parse_args(argv);common=_sibling('diagnose');common.require_step()
    if (args.engine=='native')!=(args.reference_rms is not None):raise ValueError('Native RMS probe requires current old reference only')
    import platform,sys
    import torch
    import numpy as np
    from safetensors import safe_open
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('RMS probe requires recorded A800')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):raise RuntimeError('RMS runtime differs')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    order=_sibling('expert_order_replay');layout=_sibling('frozen_eager_attention')
    m=json.loads(args.history_inputs.read_text());history_root=Path(m['current_root'])
    order.validate_inputs(m,job_id=args.history_job,source_revision=args.history_source_revision,engine=args.engine,arm='original',output=history_root/(args.engine+'-original'))
    checkpoint=_sibling('proj1_camera_ablation')._checkpoint_stat(m['checkpoint']);sources=_source_records();job=os.environ['SLURM_JOB_ID']
    cases={};history_records={}
    for variant in ('baseline','state'):
        original_root=history_root/(args.engine+'-original')/variant;sorted_root=history_root/(args.engine+'-sorted')/variant
        r=json.loads((original_root/'result.json').read_text());s=json.loads((sorted_root/'result.json').read_text())
        if (r.get('kind')!='hy_split_projection_order_replay' or r.get('status')!='original_order_observed'
                or r.get('job_id')!=args.history_job or r.get('source_revision')!=args.history_source_revision
                or r.get('engine')!=args.engine or r.get('variant')!=variant or r.get('checkpoint')!=checkpoint
                or r.get('runtime')!=runtime or r.get('projection')!='separate' or r.get('attention_order')!='original'
                or r.get('source_records')!=s.get('source_records') or r.get('input_records')!=s.get('input_records')
                or r.get('current_split_sorted_control',{}).get('raw1300exact') is not True):
            raise ValueError('Historical original-order block source/runtime/case admission failed')
        _record_files_unchanged(r['source_records'],m['snapshots_root'])
        _record_files_unchanged(r['input_records'],history_root.parent)
        if r['input_records']['manifest']!=layers._file_record(args.history_inputs):raise ValueError('Historical input manifest differs')
        order.admit_current_sorted(s,job_id=args.history_job,source_revision=args.history_source_revision,engine=args.engine,variant=variant,
            checkpoint=checkpoint,runtime=runtime,sources=r['source_records'],input_records=r['input_records'],history_job=m['qkv']['job_id'],history_source=m['qkv']['source_revision'])
        gp=Path(m['gold']['root'])/'official'/variant/'trace.safetensors';tp=Path(m['native']['root'])/variant/'trace.safetensors'
        with safe_open(str(gp),framework='pt',device='cpu') as saved:
            gold={key:saved.get_tensor(key) for key in ('prefix.k','prefix.v','prefix.modality_mask','suffix.attention_mask')}
            for case,step,layer in PROBE_SPECS:
                if case==variant:gold[f'expert.step{step}.layer{layer}.input']=saved.get_tensor(f'expert.step{step}.layer{layer}.input')
        with safe_open(str(tp),framework='pt',device='cpu') as saved:
            trace={key:saved.get_tensor(key) for key in ('suffix.mask_sorted','suffix.cos','suffix.sin')}
        mm=gold['prefix.modality_mask'][0]
        if mm.dtype!=torch.bool or mm.numel()!=240:raise ValueError('Historical prefix geometry differs')
        perm=torch.cat((mm.nonzero().flatten(),(~mm).nonzero().flatten())).numpy()
        _sibling('suffix_key_order').admit_original_mask(trace['suffix.mask_sorted'],gold['suffix.attention_mask'],perm)
        cases[variant]={'gold':gold,'trace':trace,'perm':perm,'oracle_file':original_root/'operator_outputs.safetensors'}
        history_records[variant]={'report':layers._file_record(original_root/'result.json'),'raw':layers._file_record(cases[variant]['oracle_file'])}
    reference=None;reference_report=None
    if args.engine=='native':
        reference_report=json.loads((args.reference_rms/'result.json').read_text())
        reference=load_file(str(args.reference_rms/'operator_outputs.safetensors'))
    factory=lambda:layers.load_native_owner(sources['native']['path'],projection='separate')
    owner=admitted_reference_owner(reference_report,reference,job_id=job,source_revision=args.source_revision,checkpoint=checkpoint,sources=sources,factory=factory) if reference is not None else factory()
    outputs={};metadata={};metrics={};selected=[];block_proofs={}
    args.output.mkdir(parents=True,exist_ok=False)
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(checkpoint['path'],framework='pt',device='cpu') as weights_file:
        for variant,step,layer in PROBE_SPECS:
            probe=f'{variant}.s{step}.l{layer}';case=cases[variant];gold=case['gold'];trace=case['trace']
            raw={}
            for name,key in layers.layer_keys(layer).items():
                value=weights_file.get_tensor(key)
                if value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('Probe weights must remain finite BF16')
                raw[name]=value.to('cuda');selected.append(key)
            weights,qk=layers.fuse_layer_weights(raw);cache=[]
            for field,dtype in (('k',torch.float32),('v',torch.bfloat16)):
                expanded=common.expand_official_kv(common._array(gold['prefix.'+field][layer:layer+1]),case['perm'],16,41)[0]
                cache.append(torch.from_numpy(expanded).to(device='cuda',dtype=dtype))
            observed={};meta={};scope={}
            with capture_norms(owner,observed,meta) as captured, layers.attention_order_scope(owner,case['perm'],suffix_rows=41,order='original',device='cuda',state=scope):
                residual,hidden=layers.replay_block(owner,gold[f'expert.step{step}.layer{layer}.input'][None].to('cuda'),weights,qk,*cache,
                    trace['suffix.mask_sorted'].to('cuda'),trace['suffix.cos'].to('cuda'),trace['suffix.sin'].to('cuda'),prefix_rows=240)
            with safe_open(str(case['oracle_file']),framework='pt',device='cpu') as oracle:
                for field,value in (('residual',residual),('output',hidden)):
                    expected=oracle.get_tensor(f'teacher.step{step}.layer{layer}.{field}')
                    cpu=value.detach().cpu().clone()
                    if cpu.dtype!=expected.dtype or not torch.equal(cpu,expected):raise ValueError('Actual block does not reproduce selected historical oracle')
                    outputs[f'own.{probe}.block.{field}']=cpu
            block_proofs[probe]={'exact':True,'norm_calls':captured['calls'],'attention_calls':scope['attention_calls'],'cleanup':scope['cleanup']}
            for name,value in observed.items():outputs[f'own.{probe}.{name}']=value
            metadata[probe]=meta
            for n in range(4):
                base=f'n{n}';x=layout.restore_layout(observed[base+'.input'],meta[base]['input'],device='cuda')
                w=layout.restore_layout(observed[base+'.weight'],meta[base]['weight'],device='cuda')
                stages=rms_stages(x,w)
                if not torch.equal(stages['weighted_output'].cpu(),observed[base+'.observed_output']):raise ValueError('Own staged RMS does not reproduce actual norm')
                for name,value in stages.items():outputs[f'own.{probe}.{base}.stage.{name}']=value.detach().cpu().clone()
            del raw,weights,qk,cache
        fixed_layouts={}
        if reference is not None:
            for probe in PROBES:
                for n in range(4):
                    base=f'own.{probe}.n{n}';meta=reference_report['norm_metadata'][probe][f'n{n}']
                    x=layout.restore_layout(reference[base+'.input'],meta['input'],device='cuda')
                    w=layout.restore_layout(reference[base+'.weight'],meta['weight'],device='cuda')
                    actual=owner._rms_norm(x,w);stages=rms_stages(x,w)
                    if not torch.equal(actual,stages['weighted_output']):raise ValueError('Fixed-input staged RMS differs from actual native norm')
                    outbase=f'fixed_old.{probe}.n{n}'
                    outputs[outbase+'.observed_output']=actual.detach().cpu().clone();first=None
                    for name,value in stages.items():
                        cpu=value.detach().cpu().clone();outputs[outbase+'.stage.'+name]=cpu
                        comparison=common.difference(common._array(cpu),common._array(reference[base+'.stage.'+name]))
                        comparison['unequal_elements']=int((cpu!=reference[base+'.stage.'+name]).sum())
                        metrics[outbase+'.stage.'+name]=comparison
                        if first is None and not comparison['exact']:first=name
                    fixed_layouts[outbase]={'input':_descriptor(x),'weight':_descriptor(w),'first_different_stage':first,
                        'own_input_exact_to_old':torch.equal(outputs[base+'.input'],reference[base+'.input'])}
    own_roles=validate_reference_raw({k:v for k,v in outputs.items() if k.startswith('own.')},metadata)
    report={'kind':'hy_fixed_input_rms_stages','status':'diagnostic_complete','engine':args.engine,'job_id':job,'source_revision':args.source_revision,
        'checkpoint':checkpoint,'source_records':sources,'runtime':runtime,'torch':torch.__version__,'torch_git_version':torch.version.git_version,
        'probe_ids':list(PROBES),'norm_calls':16,'block_oracles_exact':True,'block_proofs':block_proofs,'own_stage_replay_exact':True,
        'epsilon':1e-5,'stages':list(STAGES),'norm_metadata':metadata,'own_raw_role_validation':own_roles,
        'fixed_old_layouts':fixed_layouts,'fixed_old_stage_comparisons':metrics,
        'selected_checkpoint_keys':sorted(set(selected)),'weight_read_events':len(selected),'model_constructed':False,'performance_measurement':False,
        'historical_block_reference':{'job_id':args.history_job,'source_revision':args.history_source_revision,'manifest':layers._file_record(args.history_inputs),'records':history_records},
        'current_old_reference':None if reference is None else {'job_id':job,'source_revision':args.source_revision,'path':str(args.reference_rms.resolve())},
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'scope':'Four fixed local probes, actual norm observation and standalone stages only; no primitive replacement or new6of7policygate.'}
    publish(args.output,outputs,report)
    print(json.dumps({'engine':args.engine,'probes':4,'norm_calls':16,'own_stage_replay_exact':True,'raw_tensors':len(outputs)}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
