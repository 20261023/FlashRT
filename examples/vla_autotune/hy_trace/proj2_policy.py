"""Single-PROJ2 native quality diagnostic; no Expert or production changes."""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_proj2_policy_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


legacy=_sibling('native_policy_tree');ablation=legacy.ablation;merger=_sibling('merger_stages')
VISION_FIELDS=('vision.merger_input','vision.proj2_activation','vision.proj2_output','vision.merged')
PRECONDITION_KIND='hy_complete_merger_proj2_frozen_precondition'


def adapt_merger(source,proj2_call):
    tree=ast.parse(Path(source).read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='camera_merger']
    if len(nodes)!=1:raise ValueError('PROJ2 AST requires one camera Merger source')
    original=nodes[0];node=copy.deepcopy(original)
    expected_args=ast.parse('def camera_merger(self, x, grid=14): pass').body[0].args
    wanted=ast.parse('x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)').body[0]
    returned=ast.parse('return x.reshape(B, -1, C)').body[0]
    if (node.decorator_list or ast.dump(node.args)!=ast.dump(expected_args) or len(node.body)<2
            or ast.dump(node.body[-2])!=ast.dump(wanted) or ast.dump(node.body[-1])!=ast.dump(returned)
            or sum(ast.dump(n)==ast.dump(wanted.value) for n in ast.walk(node) if isinstance(n,ast.Call))!=1):
        raise ValueError('PROJ2 source signature/unique last Linear/return AST drift')
    node.body[-2].value.func=ast.Name(id='_proj2_call',ctx=ast.Load())
    reverse=copy.deepcopy(node);reverse.body[-2]=copy.deepcopy(wanted)
    if ast.dump(reverse)!=ast.dump(original):raise ValueError('PROJ2 AST roundtrip changed another operation')
    ast.fix_missing_locations(node)
    scope={'per_camera_proj1':ablation.per_camera_proj1,'_layout':ablation._layout,'_proj2_call':proj2_call}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
    return scope['camera_merger']


@contextmanager
def merger_request(pipe,*,enabled,tensors,state,synchronize,source=None):
    import torch
    import torch.nn.functional as F
    if type(enabled) is not bool or getattr(pipe,'_proj2_policy_active',False):raise ValueError('Nested or invalid PROJ2 Merger request')
    if any(k in tensors for k in VISION_FIELDS):raise ValueError('PROJ2 Vision raw roles already present')
    if getattr(pipe,'_camera_proj1_fold_spatial',False) is not True:raise ValueError('PROJ2 diagnostic requires existing HWD PROJ1')
    old=pipe.merger_forward
    if getattr(old,'__func__',None) is not ablation.camera_merger:raise ValueError('PROJ2 must wrap the actual admitted HWD camera Merger')
    missing=object();attrs=('merger_forward','_camera_proj1_encoding_views','_camera_proj1_fold_spatial','_proj2_policy_active')
    own={k:pipe.__dict__.get(k,missing) for k in attrs};hold={'inside':False};primary=None
    state.update(enabled=enabled,complete=False,merger_calls=0,proj2_calls=0,proj2_views=[],tensor_metadata={},cleanup={'passed':False})
    def record(name,value):legacy.common._rec(tensors,state['tensor_metadata'],name,value)
    def callback(x,w,b):
        if not hold['inside'] or state['proj2_calls']:raise ValueError('Unexpected PROJ2 callback count')
        state['proj2_calls']+=1;C=w.shape[0];h=hold['grid']//2
        if (w is not pipe.W._mg_proj2_w or b is not pipe.W._mg_proj2_b
                or x.shape!=(3,h,h,C) or w.shape!=(C,C) or b.shape!=(C,)
                or any(v.dtype!=torch.bfloat16 or v.device!=x.device or not torch.isfinite(v).all() for v in (x,w,b))
                or not x.is_contiguous()):raise ValueError('PROJ2 actual operand identity/geometry/BF16 contract differs')
        before=tuple(v.clone() for v in (x,w,b));state['proj2_activation_layout']=ablation._layout(x)
        def observe(view):
            same=view.untyped_storage().data_ptr()==x.untyped_storage().data_ptr()
            state['proj2_views'].append({**ablation._layout(view),'shares_input_storage':same})
            if not same:raise ValueError('PROJ2 view does not share actual activation storage')
        if enabled:out=ablation.per_camera_proj1(x,w,b,fold_spatial=False,observer=observe)
        else:observe(x);out=F.linear(x,w,b)
        if any(not merger._exact(v,prior) for v,prior in zip((x,w,b),before)):raise ValueError('PROJ2 readonly activation/weight/bias operand mutated')
        if out.shape!=x.shape or out.dtype!=torch.bfloat16 or not torch.isfinite(out).all():raise ValueError('PROJ2 actual output geometry/finite differs')
        record('vision.proj2_activation',x);record('vision.proj2_output',out)
        return out
    adapted=MethodType(adapt_merger(source or ablation.__file__,callback),pipe)
    def forward(x,grid=14):
        if hold['inside'] or state['merger_calls']:raise ValueError('Nested/repeated Merger call in one request')
        if type(grid) is not int or grid<2 or grid%2 or x.ndim!=3 or x.shape[:2]!=(3,grid*grid) or x.dtype!=torch.bfloat16 or not torch.isfinite(x).all():raise ValueError('PROJ2 actual Merger input geometry differs')
        state['merger_calls']+=1;hold.update(inside=True,grid=grid);before=x.clone()
        state['merger_input_layout']=ablation._layout(x)
        try:out=adapted(x,grid=grid)
        finally:hold['inside']=False
        if not merger._exact(x,before):raise ValueError('PROJ2 actual Merger readonly input mutated')
        if state['proj2_calls']!=1 or len(state['proj2_views'])!=(3 if enabled else 1):raise ValueError('PROJ2 actual view/callback counts differ')
        if not merger._exact(tensors['vision.proj2_output'].reshape(out.shape),out.detach().cpu()):raise ValueError('PROJ2 actual returned merged view differs')
        state['proj1_views']=list(pipe._camera_proj1_encoding_views)
        record('vision.merger_input',x);record('vision.merged',out);return out
    try:
        pipe._proj2_policy_active=True;pipe.merger_forward=forward
        yield state
        if pipe.merger_forward is not forward or pipe._camera_proj1_fold_spatial is not True or state['merger_calls']!=1 or state['proj2_calls']!=1:
            raise ValueError('Incomplete PROJ2 call count or request binding drift')
        state['complete']=True
    except BaseException as error:primary=error;raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        for name,value in own.items():
            try:
                if value is missing:pipe.__dict__.pop(name,None)
                else:setattr(pipe,name,value)
            except BaseException as error:errors.append('restore '+name+': '+str(error))
        state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False;state['_failed_owners']=(pipe,old,adapted)
            if primary is not None:primary.add_note('Merger cleanup failed: '+'; '.join(errors))
            else:raise RuntimeError('Merger cleanup failed: '+'; '.join(errors))


def validate_precondition_raw(raw,references):
    import torch
    shapes={'input':(3,196,1152),'control_activation':(3,7,7,2048),'candidate_activation':(3,7,7,2048),
        'control_proj2':(3,7,7,2048),'candidate_proj2':(3,7,7,2048),
        **{k:(3,49,2048) for k in ('original_merged','control_merged','candidate_merged','official_reference')}}
    expected={o+'.'+k:shape for o in ('gold','native') for k,shape in shapes.items()}
    expected.update({'proj1.control':(3,14,14,2048),'proj1.reference':(3,14,14,2048)})
    if set(raw)!=set(expected) or any(v.dtype!=torch.bfloat16 or tuple(v.shape)!=expected[k] or not torch.isfinite(v).all() for k,v in raw.items()):raise ValueError('Complete Merger precondition raw role/shape/dtype/finite differs')
    for origin in ('gold','native'):
        r=references[origin];base=origin+'.'
        pairs=[(raw[base+'input'],r['input']),*( (raw[base+k],r['activation']) for k in ('control_activation','candidate_activation')),
            *( (raw[base+k],r['native']) for k in ('original_merged','control_merged')),
            *( (raw[base+k],r['official']) for k in ('candidate_merged','official_reference')),
            (raw[base+'control_proj2'].reshape(3,49,2048),raw[base+'control_merged']),
            (raw[base+'candidate_proj2'].reshape(3,49,2048),raw[base+'candidate_merged'])]
        if any(not merger._exact(a,b) for a,b in pairs):raise ValueError('Complete Merger precondition input/activation/actual endpoint anchor differs')
    if not merger._exact(raw['proj1.control'],raw['proj1.reference']) or not merger._exact(raw['proj1.reference'],references['proj1']):raise ValueError('PROJ1 child raw equality/actual historical anchor differs')
    return {'tensors':len(raw),'original_and_observed_controls_exact':2,'candidate_endpoints_exact':2,'PROJ1_child_exact':True}


def admit_precondition(report,raw,*,references,job_id,source_revision,checkpoint,runtime,torch_git,factory):
    if (report.get('kind')!=PRECONDITION_KIND or report.get('status')!='operator_precondition_complete' or report.get('operator_exact') is not True
            or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision or report.get('checkpoint')!=checkpoint
            or report.get('runtime')!=runtime or report.get('torch_git_version')!=torch_git or report.get('model_constructed') is not False
            or report.get('weight_read_events')!=8
            or set(report.get('selected_checkpoint_keys',[]))!={merger.CHECKPOINT_PREFIX+k for k in merger.WEIGHTS}):raise ValueError('Complete Merger precondition header/source/runtime identity differs')
    ablation.validate_precondition(report.get('proj1_child',{}),job_id=job_id,source_revision=source_revision,checkpoint=checkpoint['path'],fold_spatial=True)
    if report.get('raw_proof')!=validate_precondition_raw(raw,references):raise ValueError('Complete Merger precondition raw proof differs')
    observations=report.get('observations',{})
    if set(observations)!={'gold','native'}:raise ValueError('Complete Merger precondition observation roles differ')
    for origin in ('gold','native'):
        if set(observations[origin])!={'control','candidate'}:raise ValueError('Complete Merger precondition observed form roles differ')
        for arm,enabled in (('control',False),('candidate',True)):
            state=observations[origin][arm];views=state.get('proj2_views',[])
            if (state.get('enabled') is not enabled or state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True
                    or state.get('merger_calls')!=1 or state.get('proj2_calls')!=1 or len(state.get('proj1_views',[]))!=3
                    or len(views)!=(3 if enabled else 1) or any(v.get('shares_input_storage') is not True for v in views)):
                raise ValueError('Complete Merger precondition actual view/callback/cleanup proof differs')
    return factory()


def qualify_merger_candidate(candidate,control,arrays):
    import torch
    a,b=candidate['tensors'],control['tensors']
    if candidate['scope'].get('enabled') is not False or control['scope'].get('enabled') is not False:raise ValueError('Single PROJ2 experiment must preserve original Expert in both arms')
    changed={'prefix.embeddings','prefix.sorted_embeddings','prefix.k','prefix.v'}
    strict=tuple(k for k in legacy.CONDITIONS if k not in changed)+('vision.merger_input','vision.proj2_activation','step0.latent_before')
    strict+=tuple(f'step{s}.time_embedding' for s in range(10))
    if any(k not in a or k not in b or not merger._exact(a[k],b[k]) for k in strict):raise ValueError('Single PROJ2 fixed input/condition invariant differs')
    if any(candidate['operator_scope'][k]!=control['operator_scope'][k] for k in ('merger_input_layout','proj2_activation_layout')):raise ValueError('Single PROJ2 actual input/activation layout invariant differs')
    for raw in (a,b):
        pe=raw['prefix.embeddings'];mm=raw['prefix.modality_mask'];perm=raw['prefix.permutation']
        if mm.dtype!=torch.bool or mm.shape!=pe.shape[:2] or int(mm.sum())!=147 or perm.dtype!=torch.long:
            raise ValueError('Single PROJ2 actual prefix modality geometry differs')
        expected=torch.cat((mm[0].nonzero().flatten(),(~mm[0]).nonzero().flatten()))
        if not merger._exact(perm,expected) or not merger._exact(raw['prefix.sorted_embeddings'],pe[:,perm]):raise ValueError('Single PROJ2 actual sorted prefix mapping differs')
        if not merger._exact(pe[:,mm[0]].reshape(3,49,2048),raw['vision.merged']) or not merger._exact(raw['vision.proj2_output'].reshape(3,49,2048),raw['vision.merged']):raise ValueError('Single PROJ2 own visual token/merged bridge differs')
    mm=a['prefix.modality_mask'][0]
    if not merger._exact(a['prefix.embeddings'][:,~mm],b['prefix.embeddings'][:,~mm]):raise ValueError('Single PROJ2 changed non-image prefix embeddings')
    actions=a['actions'][0].numpy();legacy.validator.validate_actions(actions)
    result=legacy.validator.oracle_comparison(actions,arrays,sampler='euler',num_steps=10)
    result.update(fixed_inputs_and_non_image_prefix_exact_control=True,
        effects_vs_control={k:merger._difference(a[k],b[k]) for k in ('vision.proj2_output','vision.merged','prefix.k','prefix.v','actions')})
    return result


def complete_frozen_controls(weights,history,references,*,device,synchronize):
    import math
    import torch
    grid=math.isqrt(history['native_metadata']['shape'][1]);D=history['native_metadata']['shape'][-1]
    if grid*grid!=history['native_metadata']['shape'][1] or grid%2:raise ValueError('Frozen complete Merger grid differs')
    pipe=merger.native_owner(weights);pipe.merger_forward=MethodType(ablation.camera_merger,pipe)
    raw={};observations={};gold_input=None
    for origin in ('gold','native'):
        values=references[origin]['input'].reshape(3,grid,grid,D)
        _,x,_=merger.rebuild_inputs(tuple(values.split(1)),history['official_strides'],history['native_metadata'],device=device)
        if origin=='gold':gold_input=x
        original=pipe.merger_forward(x,grid=grid).detach().cpu().clone()
        if not merger._exact(original,references[origin]['native']):raise ValueError('Original complete HWD Merger no longer reproduces R34 endpoint')
        raw[origin+'.original_merged']=original;raw[origin+'.input']=x.detach().cpu().clone()
        observations[origin]={}
        for name,enabled in (('control',False),('candidate',True)):
            capture={};state={}
            with merger_request(pipe,enabled=enabled,tensors=capture,state=state,synchronize=synchronize):out=pipe.merger_forward(x,grid=grid)
            if not merger._exact(capture['vision.merger_input'],references[origin]['input']) or not merger._exact(capture['vision.proj2_activation'],references[origin]['activation']):raise ValueError('Observed complete Merger actual input/activation differs')
            target=references[origin]['official' if enabled else 'native']
            if not merger._exact(capture['vision.merged'],target):raise ValueError('Observed complete Merger actual endpoint differs')
            raw[origin+'.'+name+'_activation']=capture['vision.proj2_activation']
            raw[origin+'.'+name+'_proj2']=capture['vision.proj2_output'];raw[origin+'.'+name+'_merged']=capture['vision.merged']
            observations[origin][name]=state
        raw[origin+'.official_reference']=references[origin]['official'].clone()
    views=[]
    actual=ablation.per_camera_proj1(gold_input.reshape(3,grid,grid,D),weights['proj1.weight'],weights['proj1.bias'],fold_spatial=True,
        observer=lambda value:views.append(ablation._layout(value)))
    raw['proj1.control']=actual.detach().cpu().clone();raw['proj1.reference']=references['proj1'].clone()
    if len(views)!=3 or not merger._exact(raw['proj1.control'],raw['proj1.reference']):raise ValueError('Fresh HWD PROJ1 child fails historical Gold output anchor')
    return raw,observations,views


def execute_policy_request(backend,arrays,*,output,arm,variant,index,synchronize):
    def context(tensors):return merger_request(backend.frontend.pipe,enabled=arm=='candidate',tensors=tensors,state={},synchronize=synchronize)
    return legacy.execute_request(backend,arrays,output=output,arm=arm,variant=variant,index=index,
        expert_enabled=False,separate_block=None,synchronize=synchronize,request_context=context)


def _source_records():
    native=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    return {n:merger.file_record(p) for n,p in (('driver',__file__),('shared',legacy.__file__),('camera',ablation.__file__),
        ('policy_scope',legacy.policy.__file__),('Merger',merger.__file__),('native',native))}


def load_references(root,*,runtime,torch_git,reduction_flag):
    from safetensors import safe_open
    from safetensors.torch import load_file
    root=Path(root).resolve();task=root.parent.parent;calls=_sibling('proj2_calls')
    if root!=task/'results/hy-proj2-calls-317555' or (root/'SOURCE_GIT_COMMIT').read_text().strip()!='32ea141e346e4faea0a8380089a9d02c5359f074':raise ValueError('R36 PROJ2 source-support root/marker differs')
    sr=json.loads((root/'native/result.json').read_text())
    if (sr.get('kind')!='hy_frozen_PROJ2_call_geometry' or sr.get('status')!='diagnostic_complete' or sr.get('job_id')!='317555'
            or sr.get('source_revision')!='32ea141e346e4faea0a8380089a9d02c5359f074'
            or sr.get('runtime')!=runtime or sr.get('torch_git_version')!=torch_git or sr.get('bf16_reduced_precision_reduction')!=reduction_flag
            or sr.get('raw_proof')!={'tensors':30,'unprofiled_profiled_exact':14,'historical_form_anchors_exact':4}):raise ValueError('R36 PROJ2 actual support/runtime/flags differ')
    if set(sr.get('source_records',{}))!={'driver','Merger'}:raise ValueError('R36 R35 source role set differs')
    for role,filename in (('driver','proj2_calls.py'),('Merger','merger_stages.py')):
        record=sr['source_records'][role];expected=task/'snapshots/round035/code/examples/vla_autotune/hy_trace'/filename
        if Path(record['path']).resolve()!=expected or merger.file_record(expected)!=record:raise ValueError('R36 R35 source path/stat differs')
        if Path(calls.__file__ if role=='driver' else merger.__file__).read_bytes()!=expected.read_bytes():raise ValueError('R36 source-support implementation changed')
    bound=sr['R34_reference'];r34=task/'results/hy-merger-stages-317360'
    if bound.get('job_id')!='317360' or bound.get('source_revision')!='c990f271da94eceae2eeca0d4b0b66243cf02106' or Path(bound['root']).resolve()!=r34:
        raise ValueError('R36 complete Merger support identity differs')
    if (r34/'SOURCE_GIT_COMMIT').read_text().strip()!='c990f271da94eceae2eeca0d4b0b66243cf02106':raise ValueError('R36 R34 source marker differs')
    manifest=json.loads((r34/'inputs.json').read_text());history,history_proof=merger.load_history(manifest,engine='native',runtime=runtime,torch_git=torch_git)
    reports={};arrays={}
    for engine in ('official','native'):
        records=bound['records'][engine]
        if set(records)!={'result.json','operator_outputs.safetensors'}:raise ValueError('R36 R34 record roles differ')
        for name,record in records.items():
            expected=r34/engine/name
            if Path(record['path']).resolve()!=expected or merger.file_record(expected)!=record:raise ValueError('R36 R34 recorded actual file path/stat differs')
        reports[engine]=json.loads((r34/engine/'result.json').read_text());arrays[engine]=load_file(str(r34/engine/'operator_outputs.safetensors'))
        if reports[engine].get('input_manifest')!=manifest:raise ValueError('R36 R34 input manifest pairing differs')
        for record in reports[engine]['source_records'].values():
            if merger.file_record(record['path'])!=record:raise ValueError('R36 R34 source stat changed')
        merger.admit_endpoints(arrays[engine],engine=engine,gold=history['gold_merged'],native=history['native_merged'],current_bridge=history['bridge'])
    calls.admit_support(reports,arrays,factory=lambda:None)
    r35=load_file(str(root/'native/operator_outputs.safetensors'))
    cap={o:{'activation':arrays['native'][o+'.native_hwd.activation'],**{f:arrays['native'][o+'.'+f+'.proj2'] for f in ('official','native_hwd')}} for o in ('gold','native')}
    calls.validate_raw(r35,cap)
    references={o:{'input':arrays['native'][o+'.native_hwd.input'].reshape(3,196,1152),'activation':cap[o]['activation'],
        'native':arrays['native'][o+'.native_hwd.actual_endpoint'],'official':arrays['native'][o+'.official.actual_endpoint']} for o in ('gold','native')}
    with safe_open(manifest['records']['gold.baseline.raw']['path'],framework='pt',device='cpu') as f:
        import torch
        references['proj1']=torch.cat([f.get_tensor(f'vision.proj1.{i}') for i in range(3)])
    if not merger._exact(references['proj1'],arrays['native']['gold.native_hwd.proj1']):raise ValueError('R36 fresh PROJ1 child reference differs from R34')
    support={'R35':{'job_id':'317555','source_revision':'32ea141e346e4faea0a8380089a9d02c5359f074','root':str(root),
        'records':{n:merger.file_record(root/'native'/n) for n in ('result.json','operator_outputs.safetensors')}},
        'R34':bound,'input_manifest':manifest,'history_proof':{k:v for k,v in history_proof.items() if k!='runtime_reference'}}
    return task,history,references,support


def _validate_request(record):
    import torch
    raw=record['tensors'];expected={*legacy.CONDITIONS,'actions',*(f'step{i}.{f}' for i in range(10) for f in ('latent_before','time','time_embedding','velocity')),*VISION_FIELDS}
    shapes={'vision.merger_input':(3,196,1152),'vision.proj2_activation':(3,7,7,2048),'vision.proj2_output':(3,7,7,2048),'vision.merged':(3,49,2048)}
    if set(raw)!=expected or len(raw)!=64 or any(not torch.isfinite(v).all() for v in raw.values()):raise ValueError('R36 full64 raw role/finite contract differs')
    if any(raw[k].shape!=shape or raw[k].dtype!=torch.bfloat16 for k,shape in shapes.items()):raise ValueError('R36 actual Vision field geometry/dtype differs')
    state=record['operator_scope']
    if state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True or state['merger_calls']!=1 or state['proj2_calls']!=1:
        raise ValueError('R36 actual Merger callback/cleanup incomplete')
    count=3 if state['enabled'] else 1
    if len(state['proj2_views'])!=count or len(state['proj1_views'])!=3 or not all(v['shares_input_storage'] for v in state['proj2_views']):raise ValueError('R36 actual projection view counts/alias evidence differs')
    if record['scope'].get('enabled') is not False or any(s['norm']['original_norm_calls']!=129 for s in record['scope']['steps']):raise ValueError('R36 unexpectedly changed original Expert')


def run_preflight(args,runtime,torch_git,reduction_flag):
    import torch
    from safetensors import safe_open
    task,history,refs,support=load_references(args.reference_root,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag)
    checkpoint=support['input_manifest']['checkpoint'];weights={}
    with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),safe_open(checkpoint['path'],framework='pt',device='cpu') as sf:
        for name in merger.WEIGHTS:
            value=sf.get_tensor(merger.CHECKPOINT_PREFIX+name)
            if value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('R36 precondition selected weights must remain finite BF16')
            weights[name]=value.to('cuda')
        if merger._weights(weights)!=(1152,2048) or not merger._exact(weights['proj1.weight'].cpu(),history['proj1_weight']) or not merger._exact(weights['proj1.bias'].cpu(),history['proj1_bias']):
            raise ValueError('R36 precondition checkpoint geometry/Gold PROJ1 weight pairing differs')
        raw,observations,p1views=complete_frozen_controls(weights,history,refs,device='cuda',synchronize=torch.cuda.synchronize)
    proof=validate_precondition_raw(raw,refs)
    child={'kind':ablation._PREFLIGHT_KIND,'operator_exact':True,'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
        'checkpoint':checkpoint,'fold_spatial':True,'model_constructed':False,'projection_views':p1views,
        'scope':'FreshactualHWDPROJ1vsGoldsavedoutputonly, noSDPAprofiling/wholepolicyclaim.'}
    report={'kind':PRECONDITION_KIND,'status':'operator_precondition_complete','operator_exact':True,'job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
        'runtime':runtime,'torch_git_version':torch_git,'bf16_reduced_precision_reduction':reduction_flag,'checkpoint':checkpoint,
        'model_constructed':False,'performance_measurement':False,'weight_read_events':len(weights),'selected_checkpoint_keys':[merger.CHECKPOINT_PREFIX+k for k in weights],
        'source_support':support,'source_records':_source_records(),'proj1_child':child,'raw_proof':proof,'observations':observations,
        'scope':'ActualcompletefrozenMergerendpoints only; no nativepolicy/robot/performance/defaultqualification. Historical6/7FAILED unchanged.'}
    admit_precondition(report,raw,references=refs,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,checkpoint=checkpoint,runtime=runtime,torch_git=torch_git,factory=lambda:None)
    args.output.mkdir(parents=True,exist_ok=False);merger.publish(args.output,raw,report)
    print(json.dumps({'arm':'preflight','raw_proof':proof,'model_constructed':False}),flush=True)
    return 0


def run_policy(args,runtime,torch_git,reduction_flag):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from flash_rt.vla import BackendConfig,load_backend
    from flash_rt.frontends.torch.hyvla_probeflow import HyVLAFlowFrontend
    if any(x is None for x in (args.precondition_report,args.fixtures,args.prior)):raise ValueError('R36 policy requires complete precondition/fixtures/prior')
    task,history,refs,support=load_references(args.reference_root,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag)
    checkpoint=support['input_manifest']['checkpoint'];checkpoint_dir=Path(checkpoint['path']).parent
    pre_path=args.precondition_report.resolve()
    if pre_path!=task/'results'/('hy-proj2-policy-'+os.environ['SLURM_JOB_ID'])/'precondition/result.json':raise ValueError('R36 current precondition report/job path differs')
    pre=json.loads(pre_path.read_text());pre_raw=load_file(str(pre_path.with_name('operator_outputs.safetensors')))
    if pre.get('source_support')!=support or pre.get('source_records')!=_source_records() or pre.get('bf16_reduced_precision_reduction')!=reduction_flag:
        raise ValueError('R36 complete precondition source/support/flags changed')
    _,directory=legacy.validator.checkpoint_record(checkpoint_dir)
    data,priors,inputs,old,old_records=legacy.load_policy_inputs(args.fixtures,args.prior,task=task,checkpoint=directory,weight_stat=checkpoint,runtime=runtime)
    config=BackendConfig(model='hyvla',checkpoint=str(checkpoint_dir),hardware='cuda_sm80',precision='bf16',sampler='euler',num_steps=10,
        options={'use_cuda_graph':False,'numerical_profile':'official_local_fp32'})
    backend_state={};requests=[]
    context=admit_precondition(pre,pre_raw,references=refs,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,checkpoint=checkpoint,runtime=runtime,torch_git=torch_git,
        factory=lambda:ablation.scoped_backend(load_backend,config,arm='camera_proj1',preflight=pre['proj1_child'],state=backend_state,synchronize=torch.cuda.synchronize,
            expected_frontend_type=HyVLAFlowFrontend,fold_spatial=True,execution='diagnostic_eager'))
    del pre_raw,refs,history
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False),context as backend:
            sources=legacy.validator.model_sources(backend)
            def execute(arm,variant,index):
                record=execute_policy_request(backend,data[variant],output=args.output,arm=arm,variant=variant,index=index,synchronize=torch.cuda.synchronize)
                _validate_request(record);requests.append(record);return record
            def control(variant,pair):
                if not np.array_equal(pair[0]['tensors']['actions'][0].numpy(),priors[variant]['actions32']):raise ValueError('R36 unchanged HWD control differs from historical actions')
                return {'historical_actions_exact':True,'historical_job_id':'309681','step_evidence':legacy.legacy_step_check(variant,pair[0]['tensors'],old.get(variant)),
                    'official_oracle':legacy.validator.oracle_comparison(pair[0]['tensors']['actions'][0].numpy(),data[variant],sampler='euler',num_steps=10)}
            matrix=legacy.run_matrix(execute,control,lambda case,pair,prior:qualify_merger_candidate(pair[0],prior,data[case]))
        counts=legacy.gate_counts(matrix)
        if len(requests)!=28 or sum(len(r['tensors']) for r in requests)!=1792:raise ValueError('R36 complete native request/raw coverage differs')
        summary={arm:{case:{'replay':row['replay'],'qualification':row['qualification'],'request_paths':[r['path'] for r in row['requests']]}
            for case,row in cases.items()} for arm,cases in matrix.items()}
        report={'kind':'hy_native_policy_single_PROJ2_diagnostic','status':'diagnostic_complete','job_id':os.environ['SLURM_JOB_ID'],'source_revision':args.source_revision,
            'runtime':runtime,'torch_git_version':torch_git,'bf16_reduced_precision_reduction':reduction_flag,'checkpoint':directory,'source_records':_source_records(),
            'source_support':support,'complete_precondition':{'path':str(pre_path),'report':pre,'files':{n:merger.file_record(pre_path.parent/n) for n in ('result.json','operator_outputs.safetensors')}},
            'inputs':inputs,'legacy_step_traces':old_records,'model_sources':sources,'backend_cleanup':backend_state['cleanup'],'matrix':summary,'original_padded_gate_passes':counts,
            'requests':len(requests),'actual_nfe':sum(r['sampler_metadata']['nfe'] for r in requests),'raw_tensors':sum(len(r['tensors']) for r in requests),
            'candidate_only_change':'merger.PROJ2 biasedLinearB3_to_threeB1','Expert_changed':False,'common_projection':'PROJ1per_cameraHWD',
            'model_constructed':True,'weight_instances':1,'graphs_enabled':False,'performance_measurement':False,'production_default_changed':False,
            'historical_policy_result_unchanged':{'job_id':'309681','padded_passes':6,'cases':7},
            'scope':'Existingseven-fixture proxy/nativeKV/free-running only, no robot/speed/generalization/defaultclaim. Upstreamvisiongap nottargeted; no R33Experttriple.'}
        legacy.publish_completed(args.output/'result.json',report,backend_state,requests)
        print(json.dumps({'arm':'policy','original_padded_gate_passes':counts,'raw_tensors':1792,'actual_nfe':report['actual_nfe'],'Expert_changed':False}),flush=True)
    except BaseException as error:
        (args.output/'failure.json').write_text(json.dumps({'status':'unqualified_single_PROJ2_diagnostic_failure','completed_request_paths':[r['path'] for r in requests],
            'error_type':type(error).__name__,'error':str(error),'backend_cleanup':backend_state.get('cleanup')},indent=2)+'\n')
        raise
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--arm',choices=('preflight','policy'),required=True);p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--source-revision',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--precondition-report',type=Path);p.add_argument('--fixtures',type=Path);p.add_argument('--prior',type=Path)
    args=p.parse_args(argv);legacy.common.require_step()
    import platform,sys
    import numpy as np
    import torch
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact R36 source commit required')
    legacy.validator.require_cuda(torch);torch.set_num_threads(2);torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    function=run_preflight if args.arm=='preflight' else run_policy
    return function(args,runtime,torch.version.git_version,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__=='__main__':raise SystemExit(main())
