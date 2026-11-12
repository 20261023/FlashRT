"""Input-derived FP32 addition-tree models, not a production mean replacement."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_tree_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


launch=_sibling('mean_launch');handoff=launch.handoff;rms=launch.rms
ARMS={'scalar_ascending':(False,False),'scalar_descending':(False,True),
      'vector_ascending':(True,False),'vector_descending':(True,True)}


def tree_mean(square,*,vectorized,descending):
    """32 lanes / four accumulators / explicit folds; source-model only."""
    import torch
    if (type(vectorized) is not bool or type(descending) is not bool or square.ndim<2
            or square.dtype!=torch.float32 or square.shape[-1] not in (128,1024)
            or min(square.shape)<1 or square.stride(-1)!=1 or square.data_ptr()%16
            or not torch.isfinite(square).all() or (square<0).any()):
        raise ValueError('FP32 tree requires admitted flags/width/stride/alignment and finite nonnegative values')
    width=square.shape[-1];rows=square.reshape(-1,width);count=rows.shape[0];groups=width//128
    chunks=rows.reshape(count,groups,32,4) if vectorized else rows.reshape(count,groups,4,32).permute(0,1,3,2)
    accum=torch.zeros((count,32,4),dtype=torch.float32,device=square.device)
    for group in range(groups):accum=accum+chunks[:,group]
    lanes=((accum[:,:,0]+accum[:,:,1])+accum[:,:,2])+accum[:,:,3]
    for offset in ((16,8,4,2,1) if descending else (1,2,4,8,16)):
        other=torch.cat((lanes[:,offset:],torch.zeros_like(lanes[:,:offset])),dim=1)
        lanes=lanes+other
    output=(lanes[:,:1]*(1.0/width)).reshape(square.shape[:-1]+(1,))
    if not torch.isfinite(output).all():raise ValueError('FP32 tree output is not finite')
    return output


def validate_raw(raw,capture):
    import torch
    expected={}
    for p in rms.PROBES:
        for n in range(4):
            name=f'{p}.n{n}';square=capture[name+'.square'];mean=capture[name+'.unprofiled_variance']
            expected[name+'.square']=square;expected[name+'.native_mean']=mean
            for arm in ARMS:expected[name+'.'+arm]=mean
    if set(raw)!=set(expected):raise ValueError('Mean tree raw role set differs')
    for key,ref in expected.items():
        value=raw[key]
        if value.dtype!=torch.float32 or value.shape!=ref.shape or not torch.isfinite(value).all():raise ValueError('Mean tree raw role shape/dtype/finite differs')
        if key.endswith(('.square','.native_mean')) and not handoff._exact(value,ref):raise ValueError('Mean tree native baseline differs')
    return {'tensors':len(raw),'native_baselines_exact':16,'squares_exact':16}


def summarize(rows):
    names={f'{p}.n{n}' for p in rms.PROBES for n in range(4)}
    if (set(rows)!=names or any(r.get('width') not in (128,1024) or set(r.get('arms',{}))!=set(ARMS)
            or any(type(v.get(ref,{}).get('exact')) is not bool for v in r['arms'].values() for ref in ('old','native')) for r in rows.values())):
        raise ValueError('Mean tree summary requires all four measured arms per input')
    by_arm={arm:{ref+'_exact':sum(r['arms'][arm][ref]['exact'] for r in rows.values()) for ref in ('old','native')} for arm in ARMS}
    old_count=sum(r['arms']['scalar_ascending' if r['width']==128 else 'vector_ascending']['old']['exact'] for r in rows.values())
    return {'inputs':16,'by_arm':by_arm,'source_old_exact':old_count,
        'source_native_exact':by_arm['vector_descending']['native_exact']}


def admit_capture(report):
    if (report.get('kind')!='hy_isolated_mean_launch_observation' or report.get('status')!='diagnostic_complete'
            or report.get('engine')!='native' or report.get('job_id')!='315622'
            or report.get('source_revision')!='071ea36442cefb0b394f676cd562eb5a52f8e8eb'
            or report.get('trace_count')!=16 or report.get('model_constructed') is not False
            or report.get('checkpoint_tensors_read')!=0
            or report.get('raw_proof')!={'tensors':64,'profiled_means_exact':16,'unprofiled_means_exact':16}):
        raise ValueError('R30 native mean capture identity/raw admission failed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--history-root',type=Path,required=True);p.add_argument('--capture-root',type=Path,required=True)
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);rms._sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Mean tree diagnostic requires recorded A800')
    if not torch.__version__.startswith('2.11.0'):raise RuntimeError('Mean tree diagnostic requires native2.11')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact current source commit required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.history_root.resolve();task=root.parent.parent;capture_root=args.capture_root.resolve()
    if root!=task/'results/hy-rms-stages-315256' or capture_root!=task/'results/hy-mean-launch-315622':raise ValueError('Historical mean roots differ')
    if (root/'SOURCE_GIT_COMMIT').read_text().strip()!='b522e8fd04616c22b680a32a3c183a566c4088a4' or (capture_root/'SOURCE_GIT_COMMIT').read_text().strip()!='071ea36442cefb0b394f676cd562eb5a52f8e8eb':raise ValueError('Historical mean source markers differ')
    ro=json.loads((root/'official/result.json').read_text());rn=json.loads((root/'native/result.json').read_text())
    old=load_file(str(root/'official/operator_outputs.safetensors'));native=load_file(str(root/'native/operator_outputs.safetensors'))
    proof=handoff.admit_history(ro,old,rn,native,job_id='315256',source_revision='b522e8fd04616c22b680a32a3c183a566c4088a4')
    rms._record_files_unchanged(ro['source_records'],task/'snapshots')
    if rms.layers._file_record(ro['checkpoint']['path'])!=ro['checkpoint']:raise ValueError('Checkpoint stat changed')
    cap=json.loads((capture_root/'native/result.json').read_text());admit_capture(cap)
    capture=load_file(str(capture_root/'native/operator_outputs.safetensors'))
    launch.validate_raw(capture,old,native,engine='native')
    expected_sources={'driver','stages','history','layout'}
    if set(cap.get('source_records',{}))!=expected_sources:raise ValueError('R30 capture source role set differs')
    rms._record_files_unchanged(cap['source_records'],task/'snapshots/round030/code')
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    if (runtime!=rn['runtime'] or runtime!=cap['runtime'] or torch.version.git_version!=rn['torch_git_version']
            or torch.version.git_version!=cap['torch_git_version']
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=rn['bf16_reduced_precision_reduction']
            or cap['bf16_reduced_precision_reduction']!=rn['bf16_reduced_precision_reduction']):raise ValueError('Mean tree runtime/Git/flags differ')
    raw={};rows={};layout=_sibling('frozen_eager_attention')
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for probe in rms.PROBES:
            for n in range(4):
                name=f'{probe}.n{n}';base='own.'+name;meta=ro['norm_metadata'][probe][f'n{n}']
                x=layout.restore_layout(old[base+'.input'],meta['input'],device='cuda');xf=x.float();square=xf.pow(2)
                current=launch.layout_info(square)
                if current!=cap['rows'][name]['input_layouts']['square']:raise ValueError('Direct mean square layout differs from actual R30 capture')
                if not handoff._exact(xf.cpu(),old[base+'.stage.float_input']):raise ValueError('Mean tree float input baseline differs')
                raw[name+'.square']=square.cpu().clone();baseline=square.mean(-1,keepdim=True);raw[name+'.native_mean']=baseline.cpu().clone()
                if not handoff._exact(raw[name+'.square'],capture[name+'.square']) or not handoff._exact(raw[name+'.native_mean'],capture[name+'.unprofiled_variance']):raise ValueError('Mean tree native baseline differs from R30')
                arms={}
                for arm,(vectorized,descending) in ARMS.items():
                    value=tree_mean(square,vectorized=vectorized,descending=descending).cpu().clone();raw[name+'.'+arm]=value
                    arms[arm]={'old':handoff._difference(value,old[base+'.stage.variance']),
                        'native':handoff._difference(value,capture[name+'.unprofiled_variance'])}
                rows[name]={'width':square.shape[-1],'square_layout':current,'arms':arms}
    raw_proof=validate_raw(raw,capture);here=Path(__file__).resolve()
    report={'kind':'hy_source_explicit_mean_tree_diagnostic','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],
        'source_revision':args.source_revision,'historical_reference':{'job_id':'315256','source_revision':'b522e8fd04616c22b680a32a3c183a566c4088a4','root':str(root),'proof':proof},
        'capture_reference':{'job_id':'315622','source_revision':'071ea36442cefb0b394f676cd562eb5a52f8e8eb','root':str(capture_root),
            'records':{n:rms.layers._file_record(capture_root/'native'/f) for n,f in (('report','result.json'),('raw','operator_outputs.safetensors'))}},
        'runtime':runtime,'torch_git_version':torch.version.git_version,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'source_records':{n:rms.layers._file_record(f) for n,f in (('driver',here),('launch',Path(launch.__file__)),('stages',Path(rms.__file__)),('history',Path(handoff.__file__)))},
        'arms':{k:list(v) for k,v in ARMS.items()},'rows':rows,'raw_proof':raw_proof,'summary':summarize(rows),
        'golden_variance_used_as_input':False,'model_constructed':False,'checkpoint_tensors_read':0,'performance_measurement':False,
        'runtime_reduce_config_observed':False,'scope':'16fixed-input source-tree hypotheses only, not compiledbranch/productionmean or fullpolicy proof. Historical6/7FAILED unchanged.'}
    args.output.mkdir(parents=True,exist_ok=False);rms.publish(args.output,raw,report)
    print(json.dumps({'summary':report['summary'],'raw_proof':raw_proof,'golden_variance_used_as_input':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
