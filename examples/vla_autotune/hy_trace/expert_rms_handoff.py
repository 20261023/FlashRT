"""Standalone RMS intermediate-value cuts; never installs a policy norm."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_handoff_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


rms=_sibling('expert_rms_stages')
CUTS=('variance','variance_plus_epsilon','inverse_rsqrt','scaled','cast_back')


def resume_rms(x,weight,cut,value,epsilon=1e-5):
    """Include the injected stage, execute only its original downstream suffix."""
    import torch
    if cut not in CUTS or epsilon!=1e-5:
        raise ValueError('RMS handoff requires an admitted cut and fixed epsilon')
    reduced=cut in CUTS[:3];shape=x.shape[:-1]+(1,) if reduced else x.shape
    dtype=x.dtype if cut=='cast_back' else torch.float32
    if (x.ndim<2 or x.dtype not in (torch.bfloat16,torch.float32)
            or weight.dtype!=torch.bfloat16 or weight.shape!=(x.shape[-1],)
            or value.shape!=shape or value.dtype!=dtype
            or x.device!=weight.device or x.device!=value.device
            or any(not torch.isfinite(v).all() for v in (x,weight,value))
            or (reduced and (value<0).any())):
        raise ValueError('RMS cut shape/dtype/device/finite contract failed')
    result={cut:value}
    index=rms.STAGES.index(cut)
    for stage in rms.STAGES[index+1:]:
        if stage=='variance_plus_epsilon':result[stage]=result['variance']+epsilon
        elif stage=='inverse_rsqrt':result[stage]=torch.rsqrt(result['variance_plus_epsilon'])
        elif stage=='scaled':result[stage]=x.float()*result['inverse_rsqrt']
        elif stage=='cast_back':result[stage]=result['scaled'].to(x.dtype)
        elif stage=='weighted_output':result[stage]=weight*result['cast_back']
    if any(not torch.isfinite(v).all() for v in result.values()):
        raise ValueError('RMS cut suffix is not finite')
    return result


def _exact(a,b):
    import torch
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)


def admit_history(old_report,old,new_report,new,*,job_id,source_revision):
    import torch
    rms.admit_reference(old_report,job_id=job_id,source_revision=source_revision,
        checkpoint=old_report.get('checkpoint'),sources=old_report.get('source_records'))
    rms.validate_reference_raw(old,old_report.get('norm_metadata',{}))
    if (new_report.get('kind')!='hy_fixed_input_rms_stages' or new_report.get('status')!='diagnostic_complete'
            or new_report.get('job_id')!=job_id or new_report.get('source_revision')!=source_revision
            or new_report.get('engine')!='native' or not new_report.get('torch','').startswith('2.11.0')
            or new_report.get('source_records')!=old_report.get('source_records')
            or new_report.get('checkpoint')!=old_report.get('checkpoint')
            or new_report.get('probe_ids')!=list(rms.PROBES)
            or new_report.get('block_oracles_exact') is not True or new_report.get('own_stage_replay_exact') is not True
            or new_report.get('norm_calls')!=16
            or any(r.get('epsilon')!=1e-5 or r.get('stages')!=list(rms.STAGES) for r in (old_report,new_report))
            or new_report.get('current_old_reference',{}).get('job_id')!=job_id
            or new_report.get('current_old_reference',{}).get('source_revision')!=source_revision):
        raise ValueError('RMS history native reference identity/oracle/epsilon contract failed')
    native_own={k:v for k,v in new.items() if k.startswith('own.')}
    rms.validate_reference_raw(native_own,new_report.get('norm_metadata',{}))
    fixed_keys=set();layout_keys=set()
    for probe in rms.PROBES:
        for n in range(4):
            own=f'own.{probe}.n{n}';fixed=f'fixed_old.{probe}.n{n}';layout_keys.add(fixed)
            for suffix in ('observed_output',*('stage.'+s for s in rms.STAGES)):
                key=fixed+'.'+suffix;fixed_keys.add(key);value=new.get(key);ref=old[own+'.'+suffix]
                if not isinstance(value,torch.Tensor) or value.shape!=ref.shape or value.dtype!=ref.dtype or not torch.isfinite(value).all():
                    raise ValueError('RMS fixed-old reference role shape/dtype/finite differs')
            if not _exact(new[fixed+'.observed_output'],new[fixed+'.stage.weighted_output']):
                raise ValueError('RMS fixed-old actual endpoint differs from staged endpoint')
            if not _exact(new[fixed+'.stage.float_input'],old[own+'.input'].float()) or not _exact(new[fixed+'.stage.square'],old[own+'.stage.square']):
                raise ValueError('RMS history fixed-old input/square reference pairing differs')
            meta=old_report['norm_metadata'][probe][f'n{n}'];live=new_report.get('fixed_old_layouts',{}).get(fixed,{})
            if any(live.get(f)!=meta[f] for f in ('input','weight')) or not _exact(new[own+'.weight'],old[own+'.weight']):
                raise ValueError('RMS history fixed-old layout/weight differs')
    if set(new)!=set(native_own)|fixed_keys or set(new_report.get('fixed_old_layouts',{}))!=layout_keys:
        raise ValueError('RMS history complete fixed-old reference role set differs')
    return {'old_tensors':len(old),'native_tensors':len(new),'actual_endpoints_checked':48}


def _difference(a,b):
    common=rms._sibling('diagnose')
    result=common.difference(common._array(a),common._array(b))
    result['unequal_elements']=int((a!=b).sum())
    return result


def run_handoffs(old,new,metadata,owner,*,device):
    """All baseline gates precede independent cuts; self-values must be no-ops."""
    layout=_sibling('frozen_eager_attention');arrays={};contexts=[]
    for probe in rms.PROBES:
        for n in range(4):
            own=f'own.{probe}.n{n}';fixed=f'fixed_old.{probe}.n{n}';meta=metadata[probe][f'n{n}']
            x=layout.restore_layout(old[own+'.input'],meta['input'],device=device)
            w=layout.restore_layout(old[own+'.weight'],meta['weight'],device=device)
            baseline=rms.rms_stages(x,w);actual=owner._rms_norm(x,w)
            if not _exact(actual,baseline['weighted_output']):raise ValueError('RMS actual native baseline endpoint differs')
            values={'observed_output':actual,**{'stage.'+s:v for s,v in baseline.items()}}
            for suffix,value in values.items():
                cpu=value.detach().cpu().clone()
                if not _exact(cpu,new[fixed+'.'+suffix]):raise ValueError('RMS native baseline does not reproduce historical fixed-old array')
                arrays[f'baseline.{probe}.n{n}.'+suffix]=cpu
            contexts.append((probe,n,x,w,baseline))
    rows={}
    for probe,n,x,w,baseline in contexts:
        own=f'own.{probe}.n{n}';name=f'{probe}.n{n}'
        for cut in CUTS:
            desc=rms._descriptor(baseline[cut])
            control_value=layout.restore_layout(baseline[cut].detach().cpu(),desc,device=device)
            control=resume_rms(x,w,cut,control_value)
            for stage,value in control.items():
                if not _exact(value,baseline[stage]):raise ValueError('RMS same-native-value handoff control differs')
                arrays[f'control.{name}.{cut}.{stage}']=value.detach().cpu().clone()
            injected=layout.restore_layout(old[own+'.stage.'+cut],desc,device=device)
            suffix=resume_rms(x,w,cut,injected);comparisons={};first=None
            for stage,value in suffix.items():
                cpu=value.detach().cpu().clone();arrays[f'old_cut.{name}.{cut}.{stage}']=cpu
                comparison=_difference(cpu,old[own+'.stage.'+stage]);comparisons[stage]=comparison
                if first is None and not comparison['exact']:first=stage
            rows[name+'.'+cut]={'native_control_exact':True,'native_cut_layout':desc,'injected_layout':rms._descriptor(injected),
                'first_different_stage':first,'weighted_output_exact_to_old':comparisons['weighted_output']['exact'],
                'comparisons':comparisons}
    return arrays,rows


def summarize_handoffs(rows):
    expected={f'{p}.n{n}.{c}' for p in rms.PROBES for n in range(4) for c in CUTS}
    if set(rows)!=expected or any(v.get('native_control_exact') is not True or type(v.get('weighted_output_exact_to_old')) is not bool for v in rows.values()):
        raise ValueError('RMS handoff summary requires all independent cut outcomes and native controls')
    exact={cut:sum(v['weighted_output_exact_to_old'] for k,v in rows.items() if k.endswith('.'+cut)) for cut in CUTS}
    stages={}
    for row in rows.values():
        first=row['first_different_stage']
        if first is not None:stages[first]=stages.get(first,0)+1
    return {'cut_count':len(rows),'exact_endpoints_by_cut':exact,'first_remaining_stage_counts':stages}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--history-root',type=Path,required=True);p.add_argument('--history-job',default='315256')
    p.add_argument('--history-source-revision',default='b522e8fd04616c22b680a32a3c183a566c4088a4')
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);rms._sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('RMS handoff requires recorded A800')
    if not torch.__version__.startswith('2.11.0'):raise RuntimeError('RMS handoff requires recorded native2.11')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Current source revision must be exact Git commit')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.history_root.resolve();task=root.parent.parent
    if root!=task/'results'/('hy-rms-stages-'+args.history_job):raise ValueError('Historical RMS result root/job differs')
    if (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.history_source_revision:raise ValueError('Historical RMS source marker differs')
    old_report=json.loads((root/'official/result.json').read_text());new_report=json.loads((root/'native/result.json').read_text())
    old=load_file(str(root/'official/operator_outputs.safetensors'));new=load_file(str(root/'native/operator_outputs.safetensors'))
    proof=admit_history(old_report,old,new_report,new,job_id=args.history_job,source_revision=args.history_source_revision)
    rms._record_files_unchanged(old_report['source_records'],task/'snapshots')
    if rms.layers._file_record(old_report['checkpoint']['path'])!=old_report['checkpoint']:raise ValueError('RMS historical checkpoint stat changed')
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    if (runtime!=new_report['runtime'] or torch.version.git_version!=new_report['torch_git_version']
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=new_report['bf16_reduced_precision_reduction']
            or new_report['current_old_reference']['path']!=str(root/'official')):
        raise ValueError('RMS historical native runtime/reduction flags/reference path changed')
    here=Path(__file__).resolve();source=here.parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    historical=Path(old_report['source_records']['native']['path'])
    if source.read_bytes()!=historical.read_bytes():raise ValueError('RMS native implementation changed')
    owner=rms.layers.load_native_owner(source,projection='separate')
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        arrays,rows=run_handoffs(old,new,old_report['norm_metadata'],owner,device='cuda')
    report={'kind':'hy_rms_intermediate_handoff','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],
        'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch.version.git_version,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'historical_reference':{'job_id':args.history_job,'source_revision':args.history_source_revision,'root':str(root),'proof':proof,
            'records':{str(f.relative_to(root)):rms.layers._file_record(f) for e in ('official','native') for f in (root/e/'result.json',root/e/'operator_outputs.safetensors')}},
        'source_records':{n:rms.layers._file_record(f) for n,f in (('driver',here),('stages',Path(rms.__file__)),('native',source),('layout',here.with_name('frozen_eager_attention.py')))},
        'baseline_raw_exact':144,'native_control_raw_exact':320,'raw_tensors':len(arrays),'epsilon':1e-5,'cuts':list(CUTS),
        'rows':rows,'summary':summarize_handoffs(rows),'checkpoint_tensors_read':0,'model_constructed':False,'policy_norm_replaced':False,
        'scope':'Old values in CURRENT native cut layouts; no old stage live-layout or full-policy equivalence claim. Historical6/7FAILED unchanged.'}
    args.output.mkdir(parents=True,exist_ok=False);rms.publish(args.output,arrays,report)
    print(json.dumps({'status':report['status'],'raw_tensors':len(arrays),'summary':report['summary'],'policy_norm_replaced':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
