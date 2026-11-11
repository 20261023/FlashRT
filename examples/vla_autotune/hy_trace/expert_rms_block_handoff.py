"""Guarded old-variance handoff in four standalone native Expert blocks."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_block_cut_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


handoff=_sibling('expert_rms_handoff');rms=handoff.rms


class NormInputMismatch(ValueError):
    """A diagnostic outcome: an old variance cannot be used on this input."""


@contextmanager
def variance_scope(owner,reference,metadata,*,tensors,details):
    import torch
    layout=_sibling('frozen_eager_attention');original=owner._rms_norm
    state={'calls':0,'attempted':0,'complete':False,'input_mismatch':None,'restored':False}
    def call(x,w):
        n=state['calls']
        if n>=4:raise ValueError('extra RMS calls outside four-norm block')
        name=f'n{n}';state['attempted']+=1
        if any(not torch.isfinite(v).all() for v in (x,w)):raise ValueError('Nonfinite actual RMS input/weight')
        current={'input':rms._descriptor(x),'weight':rms._descriptor(w)}
        tensors[name+'.input']=x.detach().cpu().clone();tensors[name+'.weight']=w.detach().cpu().clone()
        checks={'input_exact':handoff._exact(tensors[name+'.input'],reference[name+'.input']),
            'weight_exact':handoff._exact(tensors[name+'.weight'],reference[name+'.weight']),
            'layout_exact':all(current[f]==metadata[name][f] for f in ('input','weight'))}
        details[name]={'checks':checks,'current_metadata':current}
        if not all(checks.values()):
            state['input_mismatch']={'norm':name,**checks}
            raise NormInputMismatch('Actual RMS input/weight/layout differs before old variance handoff: '+name)
        native=original(x,w)
        native_variance=x.float().pow(2).mean(-1,keepdim=True)
        desc=rms._descriptor(native_variance)
        injected=layout.restore_layout(reference[name+'.stage.variance'],desc,device=x.device)
        suffix=handoff.resume_rms(x,w,'variance',injected)
        returned=suffix['weighted_output']
        if not handoff._exact(returned.detach().cpu(),reference[name+'.observed_output']):
            raise ValueError('Guarded variance suffix no longer reproduces old actual RMS endpoint')
        for field,value in (('native_output',native),('variance',injected),('returned_output',returned)):
            tensors[name+'.'+field]=value.detach().cpu().clone()
        details[name]['variance_layout']=desc
        state['calls']+=1
        return returned
    try:
        owner._rms_norm=call
        yield state
        if state['calls']!=4:raise ValueError('Incomplete RMS calls in guarded block')
        state['complete']=True
    finally:
        owner._rms_norm=original;state['restored']=True


def admit_baseline(observed,reference,*,probe):
    expected={k.removeprefix('own.'+probe+'.'):v for k,v in reference.items()
        if k.startswith('own.'+probe+'.') and '.stage.' not in k}
    if len(expected)!=14 or set(observed)!=set(expected) or any(not handoff._exact(observed[k],v) for k,v in expected.items()):
        raise ValueError('Actual native block baseline roles/dtype/values differ from315256')
    return {'exact':True,'tensors':len(observed)}


def compare_block(actual,reference):
    if set(actual)!=set(reference) or set(actual)!={'residual','output'} or any(actual[k].shape!=v.shape or actual[k].dtype!=v.dtype for k,v in reference.items()):
        raise ValueError('Block comparison role shape/dtype differs')
    return {k:handoff._difference(v,reference[k]) for k,v in actual.items()}


def validate_intervention(raw,state,reference):
    import torch
    calls=state['calls'];complete=state['complete'];mismatch=state['input_mismatch']
    if (state.get('restored') is not True or type(calls) is not int or not 0<=calls<=4
            or (complete and (calls!=4 or state['attempted']!=4 or mismatch is not None))
            or (not complete and (mismatch is None or calls>=4 or state['attempted']!=calls+1 or mismatch.get('norm')!=f'n{calls}'))):
        raise ValueError('Guarded RMS outcome/count contract failed')
    expected={f'n{n}.{f}' for n in range(calls) for f in ('input','weight','native_output','variance','returned_output')}
    if complete:expected.update(('block.residual','block.output'))
    else:expected.update((f'n{calls}.input',f'n{calls}.weight'))
    if set(raw)!=expected or any(not torch.isfinite(v).all() for v in raw.values()):raise ValueError('Guarded RMS raw role/finite contract failed')
    for n in range(calls):
        b=f'n{n}'
        for field,key in (('input','input'),('weight','weight'),('variance','stage.variance'),('returned_output','observed_output')):
            if not handoff._exact(raw[b+'.'+field],reference[b+'.'+key]):raise ValueError('Guarded RMS admitted raw role differs from old actual role')
        v=raw[b+'.native_output'];ref=reference[b+'.observed_output']
        if v.shape!=ref.shape or v.dtype!=ref.dtype:raise ValueError('Guarded RMS native observed output role differs')
    return {'status':'complete' if complete else 'input_mismatch','raw_tensors':len(raw),'norm_calls':calls}


def summarize(rows):
    if set(rows)!=set(rms.PROBES) or any(r.get('status') not in ('complete','input_mismatch') for r in rows.values()):raise ValueError('Four actual probe outcomes required')
    done=[r for r in rows.values() if r['status']=='complete']
    return {'probes':4,'complete':len(done),'input_mismatches':4-len(done),
        'residual_exact':sum(r['old_block_comparison']['residual']['exact'] for r in done),
        'output_exact':sum(r['old_block_comparison']['output']['exact'] for r in done)}


def admitted_input_owner(report,*,results_root,factory):
    hist=report['historical_block_reference'];manifest=hist['manifest']
    results_root=Path(results_root).resolve();expected_root=results_root/'hy-split-order-315024'
    if (hist.get('job_id')!='315024' or hist.get('source_revision')!='5248e14b544aa8edc17b58f1b61186b5c08ca492'
            or Path(manifest['path']).resolve()!=expected_root/'inputs.json'):
        raise ValueError('RMS historical manifest/job/source binding differs')
    rms._record_files_unchanged({'manifest':manifest},results_root)
    m=json.loads(Path(manifest['path']).read_text());paths={}
    if Path(m['current_root']).resolve()!=expected_root:raise ValueError('RMS historical report root differs')
    for variant in ('baseline','state'):
        expected_report=expected_root/'native-original'/variant/'result.json'
        record=hist.get('records',{}).get(variant,{}).get('report',{})
        if not record or Path(record['path']).resolve()!=expected_report:raise ValueError('RMS historical report path binding differs')
        rms._record_files_unchanged({'report':record},results_root)
        historical=json.loads(expected_report.read_text());records=historical.get('input_records',{})
        paths[variant]={'gold_trace':Path(m['gold']['root'])/'official'/variant/'trace.safetensors',
            'native_trace':Path(m['native']['root'])/variant/'trace.safetensors'}
        required={'manifest':expected_root/'inputs.json',**paths[variant]}
        if not set(required)<=set(records) or any(Path(records[k]['path']).resolve()!=v.resolve() for k,v in required.items()):
            raise ValueError('RMS historical input trace role/path binding differs')
        rms._record_files_unchanged(records,results_root)
    return m,paths,factory()


def run_guarded_block(owner,reference,metadata,*,run,attention_context,scope):
    raw={};details={}
    try:
        with variance_scope(owner,reference,metadata,tensors=raw,details=details) as state,attention_context:
            cr,ch=run()
        raw.update({'block.residual':cr.cpu().clone(),'block.output':ch.cpu().clone()})
    except NormInputMismatch as error:
        if scope.get('cleanup',{}).get('passed') is not True:
            raise RuntimeError('RMS input mismatch accompanied by unexpected attention cleanup failure') from error
    return raw,details,state


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--history-root',type=Path,required=True);p.add_argument('--history-job',default='315256')
    p.add_argument('--history-source-revision',default='b522e8fd04616c22b680a32a3c183a566c4088a4')
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);common=rms._sibling('diagnose');common.require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Guarded block requires recorded A800')
    if not torch.__version__.startswith('2.11.0'):raise RuntimeError('Guarded block requires recorded2.11')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact current source commit required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.history_root.resolve();task=root.parent.parent
    if root!=task/'results'/('hy-rms-stages-'+args.history_job) or (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.history_source_revision:raise ValueError('Historical RMS root/source differs')
    ro=json.loads((root/'official/result.json').read_text());rn=json.loads((root/'native/result.json').read_text())
    old=load_file(str(root/'official/operator_outputs.safetensors'));new=load_file(str(root/'native/operator_outputs.safetensors'))
    proof=handoff.admit_history(ro,old,rn,new,job_id=args.history_job,source_revision=args.history_source_revision)
    rms._record_files_unchanged(ro['source_records'],task/'snapshots')
    checkpoint=ro['checkpoint']
    if rms.layers._file_record(checkpoint['path'])!=checkpoint:raise ValueError('Checkpoint changed')
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    if (runtime!=rn['runtime'] or torch.version.git_version!=rn['torch_git_version']
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=rn['bf16_reduced_precision_reduction']
            or rn['current_old_reference']['path']!=str(root/'official')):raise ValueError('Historical native runtime/flags/reference path changed')
    manifest=rn['historical_block_reference']['manifest'];manifest_path=Path(manifest['path'])
    rms._record_files_unchanged({'manifest':manifest},task/'results')
    m=json.loads(manifest_path.read_text());order=rms._sibling('expert_order_replay')
    order.validate_inputs(m,job_id='315024',source_revision='5248e14b544aa8edc17b58f1b61186b5c08ca492',engine='native',arm='original',output=Path(m['current_root'])/'native-original')
    here=Path(__file__).resolve();native=here.parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    if native.read_bytes()!=Path(ro['source_records']['native']['path']).read_bytes():raise ValueError('Native implementation changed')
    m,input_paths,owner=admitted_input_owner(rn,results_root=task/'results',factory=lambda:rms.layers.load_native_owner(native,projection='separate'))
    cases={}
    for variant in ('baseline','state'):
        with safe_open(str(input_paths[variant]['gold_trace']),framework='pt',device='cpu') as f:
            gold={k:f.get_tensor(k) for k in ('prefix.k','prefix.v','prefix.modality_mask','suffix.attention_mask')}
        with safe_open(str(input_paths[variant]['native_trace']),framework='pt',device='cpu') as f:
            trace={k:f.get_tensor(k) for k in ('suffix.mask_sorted','suffix.cos','suffix.sin')}
        mask=gold['prefix.modality_mask'][0];perm=torch.cat((mask.nonzero().flatten(),(~mask).nonzero().flatten())).numpy()
        rms._sibling('suffix_key_order').admit_original_mask(trace['suffix.mask_sorted'],gold['suffix.attention_mask'],perm)
        cases[variant]=(gold,trace,perm)
    arrays={};rows={};selected=[]
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(checkpoint['path'],framework='pt',device='cpu') as sf:
        for variant,step,layer in rms.PROBE_SPECS:
            probe=f'{variant}.s{step}.l{layer}';gold,trace,perm=cases[variant];weights_raw={}
            for name,key in rms.layers.layer_keys(layer).items():
                value=sf.get_tensor(key)
                if value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('Weight must remain finite BF16')
                weights_raw[name]=value.to('cuda');selected.append(key)
            weights,qk=rms.layers.fuse_layer_weights(weights_raw);cache=[]
            for field,dtype in (('k',torch.float32),('v',torch.bfloat16)):
                expanded=common.expand_official_kv(common._array(gold['prefix.'+field][layer:layer+1]),perm,16,41)[0]
                cache.append(torch.from_numpy(expanded).to(device='cuda',dtype=dtype))
            ref={k.removeprefix('own.'+probe+'.'):v for k,v in old.items() if k.startswith('own.'+probe+'.')}
            meta=ro['norm_metadata'][probe];x=_sibling('frozen_eager_attention').restore_layout(ref['n0.input'],meta['n0']['input'],device='cuda')
            def run():return rms.layers.replay_block(owner,x,weights,qk,*cache,trace['suffix.mask_sorted'].to('cuda'),trace['suffix.cos'].to('cuda'),trace['suffix.sin'].to('cuda'),prefix_rows=240)
            baseline={};baseline_meta={};base_scope={}
            with rms.capture_norms(owner,baseline,baseline_meta),rms.layers.attention_order_scope(owner,perm,suffix_rows=41,order='original',device='cuda',state=base_scope):
                br,bh=run()
            baseline.update({'block.residual':br.cpu().clone(),'block.output':bh.cpu().clone()})
            baseproof=admit_baseline(baseline,new,probe=probe)
            if baseline_meta!=rn['norm_metadata'][probe]:raise ValueError('Native baseline live norm metadata changed')
            scope={};attention=rms.layers.attention_order_scope(owner,perm,suffix_rows=41,order='original',device='cuda',state=scope)
            raw,details,state=run_guarded_block(owner,ref,meta,run=run,attention_context=attention,scope=scope)
            outcome=validate_intervention(raw,state,ref)
            target={f:ref['block.'+f] for f in ('residual','output')}
            row={**outcome,'state':state,'norm_details':details,'attention_scope':scope,'native_baseline':baseproof,'native_attention_scope':base_scope,
                'native_old_block_comparison':compare_block({f:baseline['block.'+f] for f in target},target)}
            if state['complete']:row['old_block_comparison']=compare_block({f:raw['block.'+f] for f in target},target)
            rows[probe]=row
            for arm,values in (('baseline',baseline),('handoff',raw)):
                arrays.update({arm+'.'+probe+'.'+k:v for k,v in values.items()})
            del weights_raw,weights,qk,cache
    report={'kind':'hy_guarded_RMS_variance_block_handoff','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
        'historical_reference':{'job_id':args.history_job,'source_revision':args.history_source_revision,'root':str(root),'proof':proof,'manifest':manifest},
        'runtime':runtime,'torch_git_version':torch.version.git_version,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'source_records':{n:rms.layers._file_record(f) for n,f in (('driver',here),('stages',Path(rms.__file__)),('handoff',Path(handoff.__file__)),('native',native))},
        'checkpoint':checkpoint,'selected_checkpoint_keys':sorted(set(selected)),'weight_read_events':len(selected),'rows':rows,'summary':summarize(rows),
        'raw_tensors':len(arrays),'model_constructed':False,'production_norm_replaced':False,'epsilon':1e-5,
        'scope':'Only four teacher-input GoldKV blocks, guarded standalone-owner oldvariance; no propagated32layer/fullpolicy/nativevision/free-run/robotqualification. Historical6/7FAILED unchanged.'}
    args.output.mkdir(parents=True,exist_ok=False);rms.publish(args.output,arrays,report)
    print(json.dumps({'summary':report['summary'],'raw_tensors':len(arrays),'model_constructed':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
