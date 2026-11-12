"""Isolated PROJ2 geometry and bias-expression controls, not a policy change."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_proj2_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


merger=_sibling('merger_stages')
ARMS=('batched_bias','per_camera_b1_bias','per_camera_hwd_bias','per_camera_flat_bias','flat147_bias','batched_mm_add','per_camera_mm_add')
MATH_OPS={'aten::linear','aten::addmm','aten::mm','aten::matmul','aten::bmm','aten::add','aten::add_'}


def _operands(x,w,b,arm):
    import torch
    if (arm not in ARMS or x.shape!=(3,7,7,2048) or w.shape!=(2048,2048) or b.shape!=(2048,)
            or any(v.dtype!=torch.bfloat16 or not v.is_contiguous() or v.device!=x.device or v.data_ptr()%16 or not torch.isfinite(v).all() for v in (x,w,b))):
        raise ValueError('PROJ2 requires admitted arm, contiguous aligned BF16 geometry and finite operands')


def _compute(x,w,b,arm):
    import torch
    import torch.nn.functional as F
    if arm=='batched_bias':return F.linear(x,w,b)
    if arm=='flat147_bias':return F.linear(x.reshape(147,2048),w,b).reshape(x.shape)
    if arm=='batched_mm_add':return F.linear(x,w,None)+b
    pieces=[]
    for i in range(3):
        view=x[i:i+1]
        if arm=='per_camera_hwd_bias':view=x[i]
        elif arm=='per_camera_flat_bias':view=x[i].reshape(49,2048)
        value=(F.linear(view,w,None)+b if arm=='per_camera_mm_add' else F.linear(view,w,b))
        pieces.append(value.reshape(1,7,7,2048))
    return torch.cat(pieces,dim=0)


def linear_arm(x,w,b,*,arm):
    _operands(x,w,b,arm)
    return _compute(x,w,b,arm)


def summarize_cpu_events(events,label):
    events=list(events)
    if sum(e.name==label for e in events)!=1:raise ValueError('PROJ2 CPU profile requires unique label')
    math=[{'op':e.name,'input_shapes':e.input_shapes} for e in events if e.name in MATH_OPS]
    if (not any(e['op']=='aten::linear' for e in math) or not any(e['op'] in ('aten::mm','aten::addmm','aten::bmm') for e in math)
            or any(not isinstance(e['input_shapes'],list) or not e['input_shapes'] or not any(e['input_shapes']) for e in math)):
        raise ValueError('PROJ2 CPU profile lacks actual Linear/GEMM input shape observations')
    return {'math_events':math,'op_counts':dict(Counter(e['op'] for e in math)),
        'cuda_activity_requested':False,'cuBLAS_algorithm_observed':False}


def profile_arm(x,w,b,*,arm,label,path):
    import torch
    _operands(x,w,b,arm)
    before=tuple(v.clone() for v in (x,w,b))
    unprofiled=_compute(x,w,b,arm);warm=_compute(x,w,b,arm)
    if not merger._exact(unprofiled,warm):raise ValueError('PROJ2 warm outputs were unstable')
    if x.device.type=='cuda':torch.cuda.synchronize(x.device)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,
            profile_memory=False,with_stack=False,with_flops=False,acc_events=True) as profiler:
        with torch.profiler.record_function(label):value=_compute(x,w,b,arm)
    if x.device.type=='cuda':torch.cuda.synchronize(x.device)
    profiler.export_chrome_trace(str(path))
    lowering=summarize_cpu_events(profiler.events(),label)
    if not merger._exact(value,unprofiled) or any(not merger._exact(v,old) for v,old in zip((x,w,b),before)):
        raise ValueError('PROJ2 profiled output or readonly operands differ')
    return value,{'lowering':lowering,'unprofiled_profiled_exact':True,'unprofiled_output':unprofiled,
        'input_layout':merger.ablation._layout(x),'output_layout':merger.ablation._layout(value),'warm_calls':2}


def admit_support(reports,raw,*,factory):
    if set(reports)!={'official','native'} or set(raw)!={'official','native'}:raise ValueError('PROJ2 support engine set differs')
    for engine in ('official','native'):
        report=reports[engine]
        if (report.get('kind')!='hy_frozen_complete_merger_stages' or report.get('status')!='diagnostic_complete'
                or report.get('engine')!=engine or report.get('job_id')!='317360'
                or report.get('source_revision')!='c990f271da94eceae2eeca0d4b0b66243cf02106'
                or report.get('model_constructed') is not False or report.get('raw_proof')!={'tensors':60,'actual_endpoints_exact':4}):
            raise ValueError('PROJ2 R34 support header identity differs')
        merger.validate_raw(raw[engine])
        for origin in ('gold','native'):
            if not merger._exact(raw[engine][origin+'.official.activation'],raw[engine][origin+'.native_hwd.activation']):raise ValueError('PROJ2 R34 fixed-form activation support differs')
    if any(not merger._exact(v,raw['native'][key]) for key,v in raw['official'].items()):raise ValueError('PROJ2 R34 cross-runtime support differs')
    return factory()


def validate_raw(raw,references):
    import torch
    expected={o+'.activation' for o in ('gold','native')}
    expected.update(o+'.'+phase+'.'+arm for o in ('gold','native') for phase in ('unprofiled','profiled') for arm in ARMS)
    if set(raw)!=expected or any(v.dtype!=torch.bfloat16 or v.shape!=(3,7,7,2048) or not torch.isfinite(v).all() for v in raw.values()):raise ValueError('PROJ2 raw role/dtype/shape/finite differs')
    for origin in ('gold','native'):
        if not merger._exact(raw[origin+'.activation'],references[origin]['activation']):raise ValueError('PROJ2 raw activation differs from admitted support')
        for arm in ARMS:
            if not merger._exact(raw[origin+'.profiled.'+arm],raw[origin+'.unprofiled.'+arm]):raise ValueError('PROJ2 raw profiled output differs')
        for arm,ref in (('batched_bias','native_hwd'),('per_camera_b1_bias','official')):
            if not merger._exact(raw[origin+'.profiled.'+arm],references[origin][ref]):raise ValueError('PROJ2 historical endpoint anchor/baseline differs')
    return {'tensors':len(raw),'unprofiled_profiled_exact':14,'historical_form_anchors_exact':4}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv);merger._sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('PROJ2 diagnostic requires recorded A800')
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact PROJ2 producer source required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    root=args.reference_root.resolve();task=root.parent.parent
    if root!=task/'results/hy-merger-stages-317360' or (root/'SOURCE_GIT_COMMIT').read_text().strip()!='c990f271da94eceae2eeca0d4b0b66243cf02106':raise ValueError('PROJ2 R34 reference root/source differs')
    manifest=json.loads((root/'inputs.json').read_text())
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    history,history_proof=merger.load_history(manifest,engine='native',runtime=runtime,torch_git=torch.version.git_version)
    reports={};raw_support={};support_records={}
    for engine in ('official','native'):
        folder=root/engine;reports[engine]=json.loads((folder/'result.json').read_text());raw_support[engine]=load_file(str(folder/'operator_outputs.safetensors'))
        report=reports[engine]
        if report.get('input_manifest')!=manifest or set(report.get('source_records',{}))!={'driver','official','camera','AST_loader'}:raise ValueError('PROJ2 R34 manifest/source roles differ')
        for name,record in report['source_records'].items():
            expected=(Path(manifest['official_source']['path']) if name=='official' else task/'snapshots/round034/code/examples/vla_autotune/hy_trace'/
                {'driver':'merger_stages.py','camera':'proj1_camera_ablation.py','AST_loader':'frozen_eager_attention.py'}[name])
            if Path(record['path']).resolve()!=expected or merger.file_record(expected)!=record:raise ValueError('PROJ2 R34 actual source path/stat differs')
        support_records[engine]={n:merger.file_record(folder/n) for n in ('result.json','operator_outputs.safetensors')}
    if Path(merger.__file__).read_bytes()!=Path(reports['native']['source_records']['driver']['path']).read_bytes():raise ValueError('PROJ2 Merger support implementation changed')
    ref=reports['native']
    if ref['runtime']!=runtime or ref['torch_git_version']!=torch.version.git_version or ref['bf16_reduced_precision_reduction']!=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:raise ValueError('PROJ2 runtime/Git/reductionflags differ')
    for engine in ('official','native'):merger.admit_endpoints(raw_support[engine],engine=engine,gold=history['gold_merged'],native=history['native_merged'],current_bridge=history['bridge'])
    admit_support(reports,raw_support,factory=lambda:None)
    references={o:{f:raw_support['native'][o+'.'+f+'.proj2'] for f in ('official','native_hwd')} for o in ('gold','native')}
    for origin in references:references[origin]['activation']=raw_support['native'][origin+'.native_hwd.activation']
    with safe_open(manifest['checkpoint']['path'],framework='pt',device='cpu') as sf:
        keys=[merger.CHECKPOINT_PREFIX+'proj2.'+f for f in ('weight','bias')]
        w,b=[sf.get_tensor(key).to('cuda') for key in keys]
    args.output.mkdir(parents=True,exist_ok=False);traces=args.output/'traces';traces.mkdir();raw={};observations={};contrasts={}
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
        for origin in ('gold','native'):
            x=references[origin]['activation'].to('cuda');raw[origin+'.activation']=x.cpu().clone();observations[origin]={};contrasts[origin]={}
            expected=reports['native']['observations'][origin+'.native_hwd']['layouts']['stages']['activation']
            if merger.ablation._layout(x)!=expected:raise ValueError('PROJ2 actual activation layout differs from R34')
            for arm in ARMS:
                value,record=profile_arm(x,w,b,arm=arm,label='proj2.'+origin+'.'+arm,path=traces/(origin+'.'+arm+'.json'))
                raw[origin+'.profiled.'+arm]=value.cpu().clone();raw[origin+'.unprofiled.'+arm]=record.pop('unprofiled_output').cpu().clone()
                observations[origin][arm]=record
                contrasts[origin][arm]={form:merger._difference(raw[origin+'.profiled.'+arm],references[origin][form]) for form in ('official','native_hwd')}
                if arm in ('batched_bias','per_camera_b1_bias'):
                    target='native_hwd' if arm=='batched_bias' else 'official'
                    if not contrasts[origin][arm][target]['exact']:raise ValueError('PROJ2 historical endpoint anchor failed before later arms')
    proof=validate_raw(raw,references)
    summary={arm:{form:sum(contrasts[o][arm][form]['exact'] for o in ('gold','native')) for form in ('official','native_hwd')} for arm in ARMS}
    report={'kind':'hy_frozen_PROJ2_call_geometry','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
        'runtime':runtime,'torch_git_version':torch.version.git_version,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'R34_reference':{'job_id':'317360','source_revision':'c990f271da94eceae2eeca0d4b0b66243cf02106','root':str(root),'records':support_records},
        'checkpoint':manifest['checkpoint'],'selected_checkpoint_keys':keys,'weight_read_events':2,'arms':list(ARMS),'raw_proof':proof,'trace_windows':14,
        'observations':observations,'contrasts':contrasts,'summary':summary,'model_constructed':False,'performance_measurement':False,'cuBLAS_algorithm_observed':False,
        'source_records':{'driver':merger.file_record(__file__),'Merger':merger.file_record(merger.__file__)},
        'scope':'TwofrozenR34activations/operatoronly; CPUatenlowering notCUDAalgorithmevidence, explicitBF16mm-add roundingalternative. No policy/default/gate/robotqualification. Historical6/7FAILED unchanged.'}
    merger.publish(args.output,raw,report)
    print(json.dumps({'raw_proof':proof,'summary':summary,'trace_windows':14,'cuBLAS_algorithm_observed':False}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
