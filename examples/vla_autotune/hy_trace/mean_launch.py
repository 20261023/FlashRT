"""Observe isolated FP32 mean launches without changing reduction arithmetic."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    s=importlib.util.spec_from_file_location('hy_mean_launch_'+name,Path(__file__).with_name(name+'.py'))
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


handoff=_sibling('expert_rms_handoff');rms=handoff.rms


def summarize_trace(trace,label):
    events=trace.get('traceEvents',[])
    if not isinstance(events,list) or sum(e.get('cat')=='user_annotation' and e.get('name')==label and e.get('ph')=='X' for e in events)!=1:
        raise ValueError('Isolated mean trace requires one exact CPU label')
    means=[e for e in events if e.get('cat')=='cpu_op' and e.get('name')=='aten::mean' and e.get('ph')=='X']
    kernels=[]
    for e in events:
        if e.get('cat')!='kernel':continue
        args=e.get('args',{});grid=args.get('grid');block=args.get('block')
        if (not isinstance(e.get('name'),str) or not e['name']
                or any(not isinstance(v,list) or len(v)!=3 or any(type(n) is not int or n<1 for n in v) for v in (grid,block))):
            raise ValueError('Mean trace kernel launch metadata is incomplete')
        kernels.append({'name':e['name'],'grid':grid,'block':block,'args':args})
    if len(means)!=1 or not kernels:raise ValueError('Mean trace requires one CPU mean and actual CUDA kernel events')
    return {'label':label,'cpu_mean_events':len(means),'kernel_count':len(kernels),'kernels':kernels,
        'runtime_reduce_config_observed':False}


def layout_info(value):
    return {**rms._descriptor(value),'data_ptr_mod16':value.data_ptr()%16}


def validate_raw(raw,old,native,*,engine):
    if engine not in ('official','native'):raise ValueError('Unknown mean runtime')
    expected={}
    for p in rms.PROBES:
        for n in range(4):
            name=f'{p}.n{n}';base='own.'+name
            for field in ('float_input','square'):expected[name+'.'+field]=old[base+'.stage.'+field]
            target=old[base+'.stage.variance'] if engine=='official' else native['fixed_old.'+name+'.stage.variance']
            for field in ('unprofiled_variance','profiled_variance'):expected[name+'.'+field]=target
    if set(raw)!=set(expected) or any(not handoff._exact(raw[k],v) for k,v in expected.items()):
        raise ValueError('Mean raw role/dtype/profiled or unprofiled values differ from historical baseline')
    return {'tensors':len(raw),'profiled_means_exact':16,'unprofiled_means_exact':16}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--engine',choices=('official','native'),required=True);p.add_argument('--history-root',type=Path,required=True)
    p.add_argument('--history-job',default='315256');p.add_argument('--history-source-revision',default='b522e8fd04616c22b680a32a3c183a566c4088a4')
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);rms._sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Mean launch observation requires recorded A800')
    if not torch.__version__.startswith('2.7.1' if args.engine=='official' else '2.11.0'):raise RuntimeError('Mean observation runtime differs')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact current source revision required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.history_root.resolve();task=root.parent.parent
    if root!=task/'results'/('hy-rms-stages-'+args.history_job) or (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.history_source_revision:raise ValueError('Historical mean root/source differs')
    ro=json.loads((root/'official/result.json').read_text());rn=json.loads((root/'native/result.json').read_text())
    old=load_file(str(root/'official/operator_outputs.safetensors'));native=load_file(str(root/'native/operator_outputs.safetensors'))
    proof=handoff.admit_history(ro,old,rn,native,job_id=args.history_job,source_revision=args.history_source_revision)
    rms._record_files_unchanged(ro['source_records'],task/'snapshots')
    if rms.layers._file_record(ro['checkpoint']['path'])!=ro['checkpoint']:raise ValueError('Historical checkpoint changed')
    ref=ro if args.engine=='official' else rn
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    if (runtime!=ref['runtime'] or torch.version.git_version!=ref['torch_git_version']
            or torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction!=ref['bf16_reduced_precision_reduction']
            or rn['current_old_reference']['path']!=str(root/'official')):raise ValueError('Historical mean runtime/git/flags/reference path changed')
    args.output.mkdir(parents=True,exist_ok=False);traces=args.output/'traces';traces.mkdir();raw={};rows={}
    layout=_sibling('frozen_eager_attention')
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for probe in rms.PROBES:
            for n in range(4):
                name=f'{probe}.n{n}';base='own.'+name;meta=ro['norm_metadata'][probe][f'n{n}']
                x=layout.restore_layout(old[base+'.input'],meta['input'],device='cuda');xf=x.float();sq=xf.pow(2)
                raw[name+'.float_input']=xf.cpu().clone();raw[name+'.square']=sq.cpu().clone()
                if any(not handoff._exact(raw[name+'.'+f],old[base+'.stage.'+f]) for f in ('float_input','square')):raise ValueError('Direct mean input differs before profiling')
                unprofiled=sq.mean(-1,keepdim=True);warm=sq.mean(-1,keepdim=True)
                if not handoff._exact(unprofiled,warm):raise ValueError('Unprofiled mean was unstable')
                info={'float_input':layout_info(xf),'square':layout_info(sq)}
                torch.cuda.synchronize();label='mean.'+name
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False,profile_memory=False,with_stack=False,with_flops=False) as profiler:
                    with torch.profiler.record_function(label):profiled=sq.mean(-1,keepdim=True)
                torch.cuda.synchronize();path=traces/(name+'.json');profiler.export_chrome_trace(str(path))
                observed=summarize_trace(json.loads(path.read_text()),label)
                raw[name+'.unprofiled_variance']=unprofiled.cpu().clone();raw[name+'.profiled_variance']=profiled.cpu().clone()
                rows[name]={'input_layouts':info,'trace':str(path.relative_to(args.output)),'observation':observed}
    raw_proof=validate_raw(raw,old,native,engine=args.engine)
    here=Path(__file__).resolve()
    report={'kind':'hy_isolated_mean_launch_observation','status':'diagnostic_complete','engine':args.engine,'job_id':os.environ['SLURM_JOB_ID'],
        'source_revision':args.source_revision,'historical_reference':{'job_id':args.history_job,'source_revision':args.history_source_revision,'root':str(root),'proof':proof},
        'runtime':runtime,'torch_git_version':torch.version.git_version,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'source_records':{n:rms.layers._file_record(f) for n,f in (('driver',here),('stages',Path(rms.__file__)),('history',Path(handoff.__file__)),('layout',here.with_name('frozen_eager_attention.py')))},
        'rows':rows,'raw_proof':raw_proof,'trace_count':len(rows),'warm_calls_per_norm':2,'model_constructed':False,'checkpoint_tensors_read':0,
        'performance_measurement':False,'runtime_reduce_config_observed':False,
        'scope':'Only16fixed-input FP32 mean launches, not direct runtime vectorization/shuffle branch or all-layer/fullpolicy proof. Historical6/7FAILED unchanged.'}
    rms.publish(args.output,raw,report)
    print(json.dumps({'engine':args.engine,'raw_proof':raw_proof,'traces':len(rows),'runtime_reduce_config_observed':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
