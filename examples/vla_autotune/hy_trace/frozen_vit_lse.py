"""Weightless, fixed actual QKV Flash LSE observation; never a policy change."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path

SCALE=72**-.5
_ACTIVE=False


def _same(a,b):
    import torch
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)


def _layout(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype),'live_stride':list(value.stride()),'storage_offset':value.storage_offset()}


def _qkv(value):
    import torch
    if (value.ndim!=3 or value.shape[0] not in (6,18) or value.shape[1:]!=(196,3456)
            or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all()):raise ValueError('QKV requires fixed finite BF16 B6/B18, N196, H16/D72 geometry')
    span=1
    for stride,size in sorted((s,n) for s,n in zip(value.stride(),value.shape) if n>1):
        if stride<span:raise ValueError('QKV live layout overlaps')
        span+=stride*(size-1)


def split_qkv(value,*,engine):
    _qkv(value)
    if engine not in ('official','native'):raise ValueError('Unknown QKV engine')
    B=value.shape[0];parts=value.reshape(B,196,3,16,72).permute(2,0,3,1,4).unbind(0)
    if any(p.untyped_storage().data_ptr()!=value.untyped_storage().data_ptr() for p in parts):raise ValueError('QKV reshape would copy actual live layout')
    return tuple(p.permute(0,2,1,3) for p in parts) if engine=='official' else parts


def reassemble_qkv(parts,*,engine):
    import torch
    if engine not in ('official','native') or len(parts)!=3:raise ValueError('QKV inverse engine/role coverage differs')
    values=parts if engine=='official' else tuple(p.permute(0,2,1,3) for p in parts)
    shape=values[0].shape
    if len(shape)!=4 or shape[1:]!=(196,16,72) or any(p.shape!=shape or p.dtype!=torch.bfloat16 for p in values):raise ValueError('QKV inverse shape/BF16 contract differs')
    return torch.stack(values,dim=2).reshape(shape[0],196,3456)


def _context(value,*,engine,batch):
    import torch
    shape=(batch,196,16,72) if engine=='official' else (batch,16,196,72)
    if not isinstance(value,torch.Tensor) or value.shape!=shape or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():
        raise ValueError('Actual API context shape/BF16/finite differs')
    return value.reshape(batch,196,1152) if engine=='official' else value.transpose(1,2).reshape(batch,196,1152)


def _return_metadata(value):
    import torch
    if value is None:return {'type':'NoneType','undefined_tensor':True}
    if isinstance(value,torch.Tensor):return {**_layout(value),'numel':value.numel(),'device':str(value.device)}
    if isinstance(value,int):return {'type':'int','value':int(value)}
    raise ValueError('Unsupported actual API return type')


def parse_capture_result(result,*,engine,batch):
    import torch
    count={'official':3,'native':9}.get(engine)
    if count is None or not isinstance(result,tuple) or len(result)!=count:raise ValueError('Actual LSE API return arity differs')
    context=_context(result[0],engine=engine,batch=batch);value=result[1]
    if (not isinstance(value,torch.Tensor) or value.dtype!=torch.float32 or value.shape!=(batch,16,196)
            or value.stride()!=(3136,196,1) or not torch.isfinite(value).all()):raise ValueError('LSE must be actual finite unpadded contiguous FP32 B/H/196')
    if engine=='native' and (result[2] is not None or result[3] is not None or any(not isinstance(result[i],torch.Tensor) for i in (6,7))
            or type(result[4]) is not int or type(result[5]) is not int or (result[4],result[5])!=(196,196)):
        raise ValueError('Native Flash LSE return schema/max sequence geometry differs')
    if not isinstance(result[-1],torch.Tensor) or result[-1].numel()!=0:raise ValueError('Actual LSE capture requires empty debug/S_dmask at dropout0')
    return context,value,{'return_count':count,'returns':[_return_metadata(v) for v in result],'debug_numel':0}


def capture_lse(engine,qkv,*,plain,capture,targets,synchronize):
    import torch
    global _ACTIVE
    if _ACTIVE:raise ValueError('Nested LSE observation is active')
    if engine not in ('official','native'):raise ValueError('Unknown LSE capture engine')
    values=qkv if engine=='official' else (qkv,)
    if not isinstance(values,tuple) or len(values)!=(3 if engine=='official' else 1):raise ValueError('Actual QKV camera/input coverage differs')
    for value in values:_qkv(value)
    B=6 if engine=='official' else 18
    if any(v.shape[0]!=B or v.device!=values[0].device for v in values):raise ValueError('Actual QKV engine/batch/device pairing differs')
    if targets.shape!=(18,196,1152) or targets.dtype!=torch.bfloat16 or not torch.isfinite(targets).all():raise ValueError('R39 target context geometry/finite differs')
    before=tuple(v.clone() for v in values);raw={};pending={};state={'complete':False,'cleanup':{'passed':False},'layouts':{},'groups':{},'partial_layouts':{},
        'source_QKV_layouts':[_layout(v) for v in values],
        'plain_calls':0,'capture_calls':0,'learned_weight_read_events':0,'readonly_QKV_exact':False};primary=None
    def canonical():return torch.cat([v.detach().cpu() for v in values],0) if engine=='official' else values[0].detach().cpu()
    def record(name,value):
        if name in raw:raise ValueError('Duplicate LSE raw role')
        state['layouts'][name]=_layout(value);raw[name]=value.detach().cpu().contiguous().clone()
    def readonly():
        if any(not _same(v,b) for v,b in zip(values,before)):raise ValueError('Actual QKV readonly control failed')
    try:
        _ACTIVE=True;record('input.qkv',canonical())
        for size in (6,18):
            group='g'+str(size);state['groups'][group]=[];controls=[];captures=[];lses=[]
            if size==6:operands=values if engine=='official' else tuple(values[0][i:i+6] for i in (0,6,12))
            else:operands=(torch.cat(before,0),) if engine=='official' else values
            for index,value in enumerate(operands):
                operand_before=value.clone()
                inputs=split_qkv(value,engine=engine);ref=targets[index*6:(index+1)*6] if size==6 else targets
                def stage(name,tensor):
                    role=f'partial.{group}.r{index}.'+name
                    state['partial_layouts'][role]=_layout(tensor);pending[role]=tensor.detach().cpu().contiguous().clone()
                source_ptrs={v.untyped_storage().data_ptr() for v in values}
                entry={'qkv_layout':_layout(value),'API_operand_layouts':[_layout(v) for v in inputs],
                    'shares_original_QKV_storage':value.untyped_storage().data_ptr() in source_ptrs,
                    'grouping_scope':'actual_R39_camera6' if engine=='official' and size==6 else 'actual_R39_combined18' if engine=='native' and size==18 else 'new_grouping_control'}
                state['groups'][group].append(entry)
                kwargs=({'dropout_p':0.,'softmax_scale':SCALE,'causal':False,'window_size':(-1,-1),'softcap':0.,'alibi_slopes':None,
                    'deterministic':False,'return_attn_probs':False} if engine=='official' else {'dropout_p':0.,'is_causal':False,'scale':SCALE})
                state['plain_calls']+=1;plain_out=plain(*inputs,**kwargs);entry['plain_context_layout']=_layout(plain_out)
                control=_context(plain_out,engine=engine,batch=size);stage('context_control',control);readonly()
                if not _same(value,operand_before):raise ValueError('Actual plain QKV operand readonly control failed')
                if not _same(control.detach().cpu(),ref.detach().cpu()):raise ValueError('Plain API context differs from R39 hard control')
                controls.append(control.detach().cpu());entry['plain_R39_context_exact']=True
                capture_kwargs={**kwargs,'return_attn_probs':True} if engine=='official' else {**kwargs,'return_debug_mask':False}
                state['capture_calls']+=1;result=capture(*inputs,**capture_kwargs)
                # Preserve returned tensors before qualification. A malformed
                # LSE (including NaN/padding) is precisely the failure evidence
                # we need; keep its original shape and never normalize it.
                evidence_errors=[];entry['unqualified_capture_returns']=[]
                for position,returned in enumerate(result if isinstance(result,(tuple,list)) else ()):
                    try:metadata=_return_metadata(returned)
                    except BaseException:metadata={'type':type(returned).__name__}
                    entry['unqualified_capture_returns'].append(metadata)
                    if position in (0,1) and isinstance(returned,torch.Tensor):
                        try:stage('capture_return'+str(position),returned)
                        except BaseException as error:evidence_errors.append('capture return snapshot failed: '+str(error))
                entry['return_evidence_errors']=evidence_errors
                try:observed,normalizer,meta=parse_capture_result(result,engine=engine,batch=size)
                except BaseException as error:
                    for note in evidence_errors:error.add_note(note)
                    raise
                if evidence_errors:raise RuntimeError('LSE capture return evidence failed: '+'; '.join(evidence_errors))
                entry['capture_returns']=meta
                stage('context_lse_capture',observed);stage('lse',normalizer);readonly()
                if not _same(value,operand_before):raise ValueError('Actual capture QKV operand readonly control failed')
                if not _same(observed.detach().cpu(),ref.detach().cpu()):raise ValueError('LSE capture context differs from R39 hard control')
                entry['capture_R39_context_exact']=True;captures.append(observed.detach().cpu());lses.append(normalizer.detach().cpu())
            record(group+'.context_control',torch.cat(controls,0));record(group+'.context_lse_capture',torch.cat(captures,0));record(group+'.lse',torch.cat(lses,0))
            pending.clear();state['partial_layouts'].clear()
        record('after.qkv',canonical());readonly();state['readonly_QKV_exact']=True
        if len(raw)!=8 or state['plain_calls']!=4 or state['capture_calls']!=4:raise ValueError('LSE actual role/call coverage incomplete')
        state['complete']=True
    except BaseException as error:
        raw.update(pending);primary=error;error.lse_partial=(raw,state);raise
    finally:
        errors=[]
        try:
            readonly();state['readonly_QKV_exact']=True
        except BaseException as error:errors.append('readonly control failed: '+str(error))
        try:
            if 'input.qkv' in raw and 'after.qkv' not in raw:record('after.qkv',canonical())
        except BaseException as error:errors.append('after-QKV evidence failed: '+str(error))
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        _ACTIVE=False;state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False
            if primary is not None:primary.add_note('; '.join(errors))
            else:
                error=RuntimeError('LSE cleanup failed: '+'; '.join(errors));error.lse_partial=(raw,state);raise error
    return raw,state


def compare_lse(a,b):
    import torch
    if (a.dtype!=torch.float32 or b.dtype!=torch.float32 or a.shape!=b.shape
            or not torch.isfinite(a).all() or not torch.isfinite(b).all()):raise ValueError('LSE comparison requires equal finite FP32 shapes')
    x,y=a.detach().cpu(),b.detach().cpu()
    def stats(left,right):
        d=left.double()-right.double();numeric=int(torch.count_nonzero(left!=right))
        bits=int(torch.count_nonzero(left.contiguous().view(torch.int32)!=right.contiguous().view(torch.int32)))
        return {'numeric_exact':numeric==0,'numeric_unequal':numeric,'bit_exact':bits==0,'bit_unequal':bits,
            'max_abs':float(d.abs().max()),'rmse':float(d.square().mean().sqrt()),'l2_norm':float(d.norm())}
    result=stats(x,y)
    if x.shape==(18,16,196):
        result['per_camera']=[stats(x[i:i+6],y[i:i+6]) for i in (0,6,12)]
        result['per_frame']=[{'camera':i//6,'frame':i%6,**stats(x[i],y[i])} for i in range(18)]
        result['per_head']=[stats(x[:,h],y[:,h]) for h in range(16)]
    return result


def row_cross_tab(lse_a,lse_b,context_a,context_b):
    import torch
    compare_lse(lse_a,lse_b)
    if (lse_a.shape!=(18,16,196) or context_a.shape!=(18,196,1152) or context_b.shape!=context_a.shape
            or context_a.dtype!=torch.bfloat16 or context_b.dtype!=torch.bfloat16
            or not torch.isfinite(context_a).all() or not torch.isfinite(context_b).all()):raise ValueError('LSE/context row alignment requires physical B/H/N/D geometry')
    left=lse_a.detach().cpu()!=lse_b.detach().cpu()
    a=context_a.detach().cpu().reshape(18,196,16,72).permute(0,2,1,3)
    b=context_b.detach().cpu().reshape(18,196,16,72).permute(0,2,1,3)
    right=(a!=b).any(-1)
    def tab(x,y):
        return {'rows':x.numel(),'both_different':int(torch.count_nonzero(x&y)),'lse_only':int(torch.count_nonzero(x&~y)),
            'context_only':int(torch.count_nonzero(~x&y)),'both_equal':int(torch.count_nonzero(~x&~y))}
    result=tab(left,right)
    result['per_camera']=[tab(left[i:i+6],right[i:i+6]) for i in (0,6,12)]
    result['per_frame']=[{'camera':i//6,'frame':i%6,**tab(left[i],right[i])} for i in range(18)]
    result['per_head']=[tab(left[:,h],right[:,h]) for h in range(16)]
    result['scope']='Aligned numeric row overlap only; not causation or an explained-policy-error fraction.'
    return result


def reference_lse(qkv):
    import torch
    _qkv(qkv)
    q,k,_=split_qkv(qkv.detach().cpu(),engine='native')
    scale=float(torch.tensor(SCALE,dtype=torch.float32))
    # CPU FP64 mathematical reference only: not a replay of fused FP32/Tensor
    # Core ordering. Batch chunks bound temporary score storage.
    return torch.cat([torch.logsumexp((q[i:i+1].double()@k[i:i+1].double().transpose(-1,-2))*scale,-1)
        for i in range(q.shape[0])],0)


def validate_native_schema(schema):
    arguments=[{'name':a.name,'type':str(a.type),'has_default':a.has_default_value(),'default':a.default_value,'kwarg_only':a.kwarg_only} for a in schema.arguments]
    returns=[{'name':r.name,'type':str(r.type)} for r in schema.returns]
    arg_specs=[('query','Tensor',False,None,False),('key','Tensor',False,None,False),('value','Tensor',False,None,False),
        ('dropout_p','float',True,0.,False),('is_causal','bool',True,False,False),('return_debug_mask','bool',True,False,False),
        ('scale','Optional[float]',True,None,True)]
    expected=[dict(zip(('name','type','has_default','default','kwarg_only'),v)) for v in arg_specs]
    result_specs=[('output','Tensor'),('logsumexp','Tensor'),('cum_seq_q','Tensor'),('cum_seq_k','Tensor'),('max_q','int'),
        ('max_k','int'),('rng_state','Tensor'),('unused','Tensor'),('debug_attn_mask','Tensor')]
    if (schema.name!='aten::_scaled_dot_product_flash_attention' or schema.overload_name!='' or arguments!=expected
            or returns!=[{'name':n,'type':t} for n,t in result_specs]):raise ValueError('Version-bound ATen Flash schema/signature differs')
    return {'schema':str(schema),'arguments':arguments,'returns':returns}


def validate_public_signature(function):
    import inspect
    parameters=list(inspect.signature(function).parameters.values())
    names=('q','k','v','dropout_p','softmax_scale','causal','window_size','softcap','alibi_slopes','deterministic','return_attn_probs')
    defaults=(inspect.Parameter.empty,)*3+(0.,None,False,(-1,-1),0.,None,False,False)
    if (tuple(p.name for p in parameters)!=names or tuple(p.default for p in parameters)!=defaults
            or any(p.kind!=inspect.Parameter.POSITIONAL_OR_KEYWORD for p in parameters)):
        raise ValueError('Installed FA2 public signature differs from2.7.4.post1')
    return {'signature':str(inspect.signature(function)),'arguments':list(names),'dropout_p':0.,'return_attn_probs_default':False}


def validate_capture(raw,state,*,engine,qkv_reference,target_reference):
    import torch
    if engine not in ('official','native'):raise ValueError('Unknown LSE capture engine')
    specs={'input.qkv':((18,196,3456),torch.bfloat16),'after.qkv':((18,196,3456),torch.bfloat16)}
    for group in ('g6','g18'):
        specs.update({group+'.context_control':((18,196,1152),torch.bfloat16),group+'.context_lse_capture':((18,196,1152),torch.bfloat16),
            group+'.lse':((18,16,196),torch.float32)})
    if set(raw)!=set(specs) or any(v.device.type!='cpu' or v.shape!=specs[k][0] or v.dtype!=specs[k][1] or not torch.isfinite(v).all() for k,v in raw.items()):
        raise ValueError('LSE capture raw role/shape/dtype/finite coverage differs')
    if (state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True or state.get('plain_calls')!=4 or state.get('capture_calls')!=4
            or state.get('learned_weight_read_events')!=0 or state.get('readonly_QKV_exact') is not True or set(state.get('layouts',{}))!=set(raw)):
        raise ValueError('LSE capture calls/cleanup/weight-free control differs')
    if not _same(raw['input.qkv'],qkv_reference) or not _same(raw['after.qkv'],qkv_reference):raise ValueError('LSE QKV source/readonly control differs')
    for group in ('g6','g18'):
        for field in ('context_control','context_lse_capture'):
            if not _same(raw[group+'.'+field],target_reference):raise ValueError('LSE actual context control differs from R39')
    sources=state.get('source_QKV_layouts',[])
    if len(sources)!=(3 if engine=='official' else 1) or set(state.get('groups',{}))!={'g6','g18'}:raise ValueError('LSE source/capture view coverage differs')
    for group,size,count in (('g6',6,3),('g18',18,1)):
        rows=state['groups'][group]
        if len(rows)!=count:raise ValueError('LSE capture grouping call coverage differs')
        for i,entry in enumerate(rows):
            layout=entry['qkv_layout']
            source=sources[i] if engine=='official' and size==6 else sources[0]
            expected_offset=source['storage_offset']+(i*6*source['live_stride'][0] if engine=='native' and size==6 else 0)
            if engine=='official' and size==18:expected_offset=0
            if (layout['shape']!=[size,196,3456] or layout['dtype']!='torch.bfloat16' or layout['live_stride']!=source['live_stride']
                    or layout['storage_offset']!=expected_offset or entry.get('plain_R39_context_exact') is not True or entry.get('capture_R39_context_exact') is not True
                    or entry.get('shares_original_QKV_storage') is not (engine=='native' or size==6)):
                raise ValueError('LSE actual QKV view/offset or context control proof differs')
            operands=entry.get('API_operand_layouts',[]);stride=layout['live_stride']
            expected_shape=[size,196,16,72] if engine=='official' else [size,16,196,72]
            expected_stride=[stride[0],stride[1],72,1] if engine=='official' else [stride[0],72,stride[1],1]
            if len(operands)!=3 or any(v!={'shape':expected_shape,'dtype':'torch.bfloat16','live_stride':expected_stride,
                    'storage_offset':expected_offset+j*1152} for j,v in enumerate(operands)):
                raise ValueError('LSE actual API operand layout/axis metadata differs')
            meta=entry.get('capture_returns',{});returns=meta.get('returns',[]);expected_count=3 if engine=='official' else 9
            if meta.get('return_count')!=expected_count or len(returns)!=expected_count or meta.get('debug_numel')!=0:
                raise ValueError('LSE actual return metadata coverage differs')
            if (returns[0].get('shape')!=expected_shape or returns[0].get('dtype')!='torch.bfloat16'
                    or returns[1].get('shape')!=[size,16,196] or returns[1].get('dtype')!='torch.float32'
                    or returns[1].get('live_stride')!=[3136,196,1] or returns[-1].get('numel')!=0):
                raise ValueError('LSE actual context/LSE/debug return layout metadata differs')
            if engine=='native' and (returns[4]!={'type':'int','value':196} or returns[5]!={'type':'int','value':196}
                    or returns[2:4]!=[{'type':'NoneType','undefined_tensor':True}]*2
                    or any('shape' not in returns[j] for j in (6,7))):raise ValueError('LSE native return schema metadata differs')
    return {'tensors':8,'BF16_arrays':6,'FP32_LSE_arrays':2,'QKV_readonly_exact':True,'all_contexts_exact_R39':True,'learned_weight_read_events':0}


def file_record(path):
    path=Path(path).resolve();s=path.stat();return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def admit_before_api(manifest,*,engine,runtime,torch_git,reduction_flag,current_native,current_bridge,current_prefix,factory):
    if manifest.get('kind')!='hy_actual_QKV_weightless_LSE_inputs' or engine not in ('official','native'):raise ValueError('R40 input kind/engine differs')
    task=Path(manifest['task_root']).resolve();root=task/'results/hy-vit-attention-bridge-319502'
    names={'SOURCE_GIT_COMMIT','inputs.json','official/result.json','official/operator_outputs.safetensors','native/result.json',
        'native/operator_outputs.safetensors','native/projection_interaction.safetensors'}
    records=manifest.get('R39_records',{})
    if set(records)!=names:raise ValueError('R39 support record coverage differs')
    for name,rec in records.items():
        if Path(rec['path']).resolve()!=root/name or file_record(root/name)!=rec:raise ValueError('R39 actual support record path/stat differs')
    if (root/'SOURCE_GIT_COMMIT').read_text().strip()!='e058161ad2b58808d3acd07e9abf18b9b7e434d8':raise ValueError('R39 frozen source marker differs')
    frozen=task/'snapshots/round039/code/examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py';rec=manifest['frozen_bridge']
    if Path(rec['path']).resolve()!=frozen or file_record(frozen)!=rec or frozen.read_bytes()!=Path(current_bridge).read_bytes():
        raise ValueError('R39 bridge original namespace/source path/stat differs')
    current=task/'snapshots/round040/code/flash_rt/models/hyvla/pipeline_thor.py';rec=manifest['current_native']
    if Path(current_native).resolve()!=current or Path(rec['path']).resolve()!=current or file_record(current)!=rec:raise ValueError('Current native source path/stat differs')
    old_root=task/'snapshots/round039/code';native39=old_root/'flash_rt/models/hyvla/pipeline_thor.py'
    bridge=_module(frozen,'hy_lse_original_R39_bridge');inputs=json.loads((root/'inputs.json').read_text())
    def metadata_consumer(prefix,parents):return prefix,parents
    prefix,r38=bridge.admit_before_weights(inputs,engine=engine,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag,
        current_native=native39,current_prefix=current_prefix,factory=metadata_consumer)
    if current.read_bytes()!=native39.read_bytes():raise ValueError('Current native source bytes differ from admitted R39')
    sources=bridge._source_records(inputs,{'R38_reports':r38});r39={e:json.loads((root/e/'result.json').read_text()) for e in ('official','native')}
    for e,report in r39.items():
        obs=report.get('observations',{});parent=r38[e]
        if (report.get('kind')!='hy_actual_vit_attention_context_projection_bridge' or report.get('status')!='bridge_complete'
                or report.get('engine')!=e or report.get('job_id')!='319502' or report.get('source_revision')!='e058161ad2b58808d3acd07e9abf18b9b7e434d8'
                or report.get('runtime')!=parent['runtime'] or report.get('torch_git_version')!=parent['torch_git_version']
                or report.get('bf16_reduced_precision_reduction')!=parent['bf16_reduced_precision_reduction'] or report.get('input_manifest')!=inputs
                or report.get('source_records')!=sources or report.get('checkpoint')!=parent['checkpoint'] or report.get('weight_read_events')!=4
                or len(report.get('selected_checkpoint_keys',[]))!=4 or set(report['selected_checkpoint_keys'])!={bridge.PREFIX+k for k in bridge.attention_keys()}
                or report.get('full_model_constructed') is not False or report.get('full_prefix_constructed') is not False
                or report.get('attention_module_constructed') is not (e=='official') or report.get('performance_measurement') is not False
                or obs.get('actual',{}).get('complete') is not True or obs['actual'].get('cleanup',{}).get('passed') is not True):
            raise ValueError('R39 bridge source/runtime/checkpoint/weight/complete proof differs')
        if (e=='official' and obs.get('cuts') is not None) or (e=='native' and (obs.get('cuts',{}).get('complete') is not True or obs['cuts'].get('cleanup',{}).get('passed') is not True)):
            raise ValueError('R39 cut cleanup proof differs')
    return factory(bridge,prefix,r39,r38)


def nonweight_roles(kind):
    fields=('norm1','qkv','context','projection','api','control','self_projection')
    if kind=='R40.actual':return {'input.qkv','after.qkv'}|{g+'.'+f for g in ('g6','g18') for f in ('context_control','context_lse_capture','lse')}
    if kind=='R39.official':return {f'camera{i}.'+f for i in range(3) for f in fields}
    if kind=='R39.native':return set(fields)|{f'cuts.{cell}.{f}' for cell in ('N18','N6','O18','O6') for f in ('input','output')}|{'cuts.restored_context','cuts.restore_projection'}
    if kind=='R39.interaction':return {'projection_interaction'}
    if kind not in ('R38.official','R38.native'):raise ValueError('Unknown nonweight input role class')
    roles={'prepared.images'};fields=('input','norm1_input','norm1','qkv','attention','residual','norm2','mlp','output')
    for i in range(3 if kind=='R38.official' else 1):
        pre=f'camera{i}.' if kind=='R38.official' else ''
        roles.update(pre+k for k in ('patch','position','block3.time_pe','block3.temporal_v'))
        roles.update(pre+f'block{j}.'+f for j in range(4) for f in fields)
        if kind=='R38.official':roles.add(pre+'api_current_frame')
    if kind=='R38.native':roles.add('control.block3.output')
    return roles


def read_nonweight(record,*,kind,opener=None):
    if opener is None:
        from safetensors import safe_open
        opener=safe_open
    if file_record(record['path'])!=record:raise ValueError('Nonweight input record stat differs')
    expected=nonweight_roles(kind)
    with opener(record['path'],framework='pt',device='cpu') as sf:
        if set(sf.keys())!=expected:raise ValueError('Nonweight input role set differs; no learned trace roles may be read')
        return {name:sf.get_tensor(name) for name in sorted(expected)}


def load_inputs(manifest,*,engine,runtime,torch_git,reduction_flag,current_native=None,current_bridge=None,current_prefix=None,consumer=lambda value:value):
    import torch
    code=Path(__file__).resolve().parents[3]
    current_native=current_native or code/'flash_rt/models/hyvla/pipeline_thor.py'
    current_bridge=current_bridge or Path(__file__).with_name('frozen_vit_attention_bridge.py')
    current_prefix=current_prefix or Path(__file__).with_name('frozen_vit_prefix.py')
    task=Path(manifest['task_root']).resolve();root39=task/'results/hy-vit-attention-bridge-319502'
    def read(bridge,prefix,r39,r38):
        inputs39=json.loads((root39/'inputs.json').read_text());raw38={};raw39={};events=[]
        def selected(record,kind):
            values=read_nonweight(record,kind=kind);events.extend({'file':record['path'],'role':k} for k in values);return values
        for e in ('official','native'):
            raw38[e]=selected(inputs39['R38_records'][e+'/operator_outputs.safetensors'],'R38.'+e)
            proof=prefix.validate_raw(raw38[e],engine=e)
            if proof!=r38[e]['raw_proof']:raise ValueError('R38 actual raw proof differs')
        refs={}
        for e in ('official','native'):
            refs[e]={name:(tuple(raw38[e][f'camera{i}.block0.'+field] for i in range(3)) if e=='official' else raw38[e]['block0.'+field])
                for name,field in (('norm1','norm1'),('qkv','qkv'),('post','attention'))}
            raw39[e]=selected(manifest['R39_records'][e+'/operator_outputs.safetensors'],'R39.'+e)
            names=[f'camera{i}.qkv' for i in range(3)] if e=='official' else ['qkv'];B=6 if e=='official' else 18
            for name in names:
                layout=r39[e]['observations']['actual']['layouts'][name]
                if layout!={'shape':[B,196,3456],'dtype':'torch.bfloat16','live_stride':[677376,3456,1],'storage_offset':0}:
                    raise ValueError('R39 actual QKV layout/offset differs from frozen operands')
        common=torch.cat([raw39['official'][f'camera{i}.qkv'] for i in range(3)],0)
        if not _same(common,raw39['native']['qkv']):raise ValueError('R39 common actual QKV values differ')
        contexts={'official':torch.cat([raw39['official'][f'camera{i}.context'] for i in range(3)],0),'native':raw39['native']['context']}
        I=selected(manifest['R39_records']['native/projection_interaction.safetensors'],'R39.interaction')['projection_interaction']
        for e in ('official','native'):
            obs=r39[e]['observations'];proof=bridge.validate_bundle(raw39[e],engine=e,actual_state=obs['actual'],references=refs[e],
                cut_state=obs.get('cuts'),old_context=contexts['official'],old_reference=torch.cat(refs['official']['post'],0),interaction_array=I)
            if proof!=r39[e]['raw_proof']:raise ValueError('R39 actual context/interaction raw proof differs')
        values={'QKV':common,'qkv_by_engine':{'official':tuple(raw39['official'][f'camera{i}.qkv'] for i in range(3)),'native':raw39['native']['qkv']},
            'qkv_layouts':{e:tuple(r39[e]['observations']['actual']['layouts'][f'camera{i}.qkv'] for i in range(3)) if e=='official'
                else r39[e]['observations']['actual']['layouts']['qkv'] for e in ('official','native')},
            'contexts':contexts,'R39_reports':r39,'R38_reports':r38,'checkpoint_identity_stat_only':r39[engine]['checkpoint'],
            'learned_weight_read_events':0,'nonweight_tensor_read_events':len(events),'selected_input_roles':events}
        if len(events)!=205:raise ValueError('Nonweight support input coverage differs')
        return consumer(values)
    return admit_before_api(manifest,engine=engine,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag,current_native=current_native,
        current_bridge=current_bridge,current_prefix=current_prefix,factory=read)


def check_FA2_files(records):
    if set(records)!={'package','interface','binary'}:raise ValueError('Installed FA2 source record roles differ')
    for record in records.values():
        if file_record(record['path'])!=record:raise ValueError('Installed FA2 source/binary stat changed')
    package=Path(records['package']['path']);interface=Path(records['interface']['path']);binary=Path(records['binary']['path'])
    if package.name!='__init__.py' or interface!=package.with_name('flash_attn_interface.py') or binary.parent!=package.parent.parent or not binary.name.startswith('flash_attn_2_cuda') or binary.suffix!='.so':
        raise ValueError('Installed FA2 source/binary paths differ')
    init=ast.parse(package.read_text());versions=[n.value.value for n in init.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
        and any(isinstance(t,ast.Name) and t.id=='__version__' for t in n.targets)]
    if versions!=['2.7.4.post1']:raise ValueError('Installed FA2 version differs')
    tree=ast.parse(interface.read_text());functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='flash_attn_func']
    classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FlashAttnFunc']
    if len(functions)!=1 or len(classes)!=1:raise ValueError('Installed FA2 source definitions differ')
    args=functions[0].args;names=[a.arg for a in args.args]
    if (names!=['q','k','v','dropout_p','softmax_scale','causal','window_size','softcap','alibi_slopes','deterministic','return_attn_probs']
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
            or [ast.literal_eval(v) for v in args.defaults]!=[0.,None,False,(-1,-1),0.,None,False,False]):raise ValueError('Installed FA2 public source signature differs')
    forward=[n for n in classes[0].body if isinstance(n,ast.FunctionDef) and n.name=='forward']
    if len(forward)!=1:raise ValueError('Installed FA2 forward source differs')
    calls=[n for n in ast.walk(forward[0]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_wrapped_flash_attn_forward']
    flags=[kw.value for n in calls for kw in n.keywords if kw.arg=='return_softmax']
    if len(calls)!=1 or len(flags)!=1 or ast.dump(flags[0])!=ast.dump(ast.parse('return_softmax and dropout_p > 0',mode='eval').body):
        raise ValueError('Installed FA2 dropout0 return_softmax source branch differs')
    return {'version':'2.7.4.post1','dropout0_return_softmax_false':True,'records':records,'scope':'Installed wrapper AST and binary stat, not a compiled-branch proof.'}


def admit_official_lse(report,raw,*,job_id,source_revision,runtime,torch_git,reduction_flag,inputs,source_records,qkv_reference,target_reference,factory):
    if (report.get('kind')!='hy_weightless_actual_QKV_LSE' or report.get('status')!='LSE_capture_complete' or report.get('engine')!='official'
            or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision or report.get('runtime')!=runtime
            or report.get('torch_git_version')!=torch_git or report.get('bf16_reduced_precision_reduction')!=reduction_flag
            or report.get('input_manifest')!=inputs or report.get('source_records')!=source_records or report.get('learned_weight_read_events')!=0
            or report.get('model_constructed') is not False):raise ValueError('Same-job official LSE predecessor source/runtime/weight-free header differs')
    proof=validate_capture(raw,report.get('observations',{}),engine='official',qkv_reference=qkv_reference,target_reference=target_reference)
    if proof!=report.get('raw_proof'):raise ValueError('Same-job official LSE predecessor raw proof differs')
    return factory()


def qualified_observation(engine,qkv,*,plain,capture,targets,synchronize):
    raw={};state={};reference=None
    try:
        raw,state=capture_lse(engine,qkv,plain=plain,capture=capture,targets=targets,synchronize=synchronize)
        if engine=='native':reference=reference_lse(raw['input.qkv'])
        proof=validate_capture(raw,state,engine=engine,qkv_reference=raw['input.qkv'],target_reference=targets.detach().cpu())
        return raw,state,reference,proof
    except BaseException as error:
        if raw:error.lse_partial=(raw,state)
        if reference is not None:error.lse_reference=reference
        raise


def write_failure(output,error,*,raw=None,observations=None,reference=None):
    from safetensors.torch import save_file
    values=dict(raw or {});state=dict(observations or {});partial=getattr(error,'lse_partial',None)
    if partial is not None:values.update(partial[0]);state.update(partial[1])
    if reference is None:reference=getattr(error,'lse_reference',None)
    errors=[]
    for name,arrays in (('unqualified_partial.safetensors',values),('unqualified_reference.safetensors',{'mathematical_LSE':reference} if reference is not None else {})):
        if arrays:
            try:save_file({k:v.contiguous() for k,v in arrays.items()},str(Path(output)/name))
            except BaseException as secondary:errors.append(str(secondary));error.add_note('LSE evidence write failed: '+str(secondary))
    record={'status':'unqualified_LSE_observation_failure','qualified':False,'error_type':type(error).__name__,'error':str(error),
        'observations':state,'captured_raw_roles':list(values),'evidence_write_errors':errors}
    try:(Path(output)/'failure.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    except BaseException as secondary:error.add_note('LSE failure metadata write failed: '+str(secondary))


def publish_capture(output,raw,report,*,qkv_reference,target_reference,reference=None):
    import torch
    from safetensors.torch import save_file
    proof=validate_capture(raw,report['observations'],engine=report['engine'],qkv_reference=qkv_reference,target_reference=target_reference)
    if proof!=report.get('raw_proof',proof):raise ValueError('LSE raw proof differs before publication')
    if report['engine']=='native' and (reference is None or reference.dtype!=torch.float64 or reference.device.type!='cpu'
            or reference.shape!=(18,16,196) or not torch.isfinite(reference).all()):raise ValueError('LSE CPU mathematical reference missing or malformed')
    save_file({k:v.contiguous() for k,v in raw.items()},str(Path(output)/'operator_outputs.safetensors'))
    if report['engine']=='native':save_file({'mathematical_LSE':reference.contiguous()},str(Path(output)/'mathematical_LSE.safetensors'))
    path=Path(output)/'result.json.tmp';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');path.replace(Path(output)/'result.json')


def _bind_apis(engine,manifest,support):
    import importlib
    import sys
    import torch
    records=manifest['FA2_files'];old_runtime=support['R39_reports']['official']['runtime']
    environment=Path(old_runtime['executable']).parent.parent;minor='.'.join(old_runtime['python'].split('.')[:2])
    site=environment/'lib'/('python'+minor)/'site-packages'
    if (Path(records['package']['path']).resolve()!=site/'flash_attn/__init__.py'
            or Path(records['interface']['path']).resolve()!=site/'flash_attn/flash_attn_interface.py'
            or Path(records['binary']['path']).resolve().parent!=site):raise ValueError('Installed FA2 paths are not bound to actual old R39 environment')
    package_proof=check_FA2_files(records)
    if engine=='official':
        package=importlib.import_module('flash_attn');interface=importlib.import_module('flash_attn.flash_attn_interface');binary=importlib.import_module('flash_attn_2_cuda')
        if (package.__version__!='2.7.4.post1' or file_record(package.__file__)!=records['package']
                or file_record(interface.__file__)!=records['interface'] or file_record(binary.__file__)!=records['binary']):raise ValueError('Imported FA2 module/binary identity differs')
        fn=interface.flash_attn_func;proof=validate_public_signature(fn)
        return fn,fn,{'FA2':package_proof,'public_signature':proof}
    if any(n=='flash_attn' or n.startswith('flash_attn.') or n=='hy_vla' or n.startswith('hy_vla.') for n in sys.modules):
        raise ValueError('Native weightless LSE runtime must not import old FA2/Hy package')
    op=torch.ops.aten._scaled_dot_product_flash_attention.default;schema=validate_native_schema(op._schema)
    binary=Path(torch.__file__).parent/'lib/libtorch_cuda.so'
    return torch.nn.functional.scaled_dot_product_attention,op,{'ATen_Flash_schema':schema,'torch_module':file_record(torch.__file__),'torch_cuda_binary_stat':file_record(binary)}


def restore_QKV(support,*,engine,device):
    source=support['qkv_by_engine'][engine];meta=support['qkv_layouts'][engine]
    values=source if engine=='official' else (source,);layouts=meta if engine=='official' else (meta,)
    if len(values)!=len(layouts) or len(values)!=(3 if engine=='official' else 1):raise ValueError('Actual QKV camera/layout coverage differs')
    restored=[]
    for value,layout in zip(values,layouts):
        tensor=value.to(device=device)
        if _layout(tensor)!=layout:raise ValueError('Actual R39 QKV stride/offset restoration differs; no contiguous fallback')
        restored.append(tensor)
    return tuple(restored) if engine=='official' else restored[0]


def _source_records(manifest,support):
    return {'driver':file_record(__file__),'native':manifest['current_native'],'frozen_R39_bridge':manifest['frozen_bridge'],
        'R39_components':support['R39_reports']['official']['source_records'],'FA2_files':manifest['FA2_files']}


def _reference_stats(value,reference):
    x=value.double();d=x-reference
    return {'max_abs':float(d.abs().max()),'rmse':float(d.square().mean().sqrt()),'l2_norm':float(d.norm()),
        'scope':'CPU FP64 mathematical reference distance, not fused-kernel correctness or quality ranking.'}


def run(args,runtime,torch_git,reduction_flag):
    import torch
    manifest=json.loads(args.inputs.read_text());task=Path(manifest['task_root']).resolve();root=task/'results'/('hy-vit-lse-'+os.environ['SLURM_JOB_ID'])
    if (args.inputs.resolve()!=root/'inputs.json' or args.output.resolve()!=root/args.engine
            or (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.source_revision):raise ValueError('R40 source/job/input/output root differs')
    support=load_inputs(manifest,engine=args.engine,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag)
    sources=_source_records(manifest,support);old_raw=None
    if args.engine=='native':
        if args.official_lse is None or args.official_lse.resolve()!=root/'official':raise ValueError('Native LSE requires same-job actual official predecessor')
        report=json.loads((args.official_lse/'result.json').read_text());old_raw=read_nonweight(file_record(args.official_lse/'operator_outputs.safetensors'),kind='R40.actual')
        ref=support['R39_reports']['official']
        admit_official_lse(report,old_raw,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,runtime=ref['runtime'],torch_git=ref['torch_git_version'],
            reduction_flag=ref['bf16_reduced_precision_reduction'],inputs=manifest,source_records=sources,qkv_reference=support['QKV'],
            target_reference=support['contexts']['official'],factory=lambda:None)
    args.output.mkdir(parents=True,exist_ok=False);raw={};state={};reference=None
    try:
        plain,capture,API_proof=_bind_apis(args.engine,manifest,support)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            qkv=restore_QKV(support,engine=args.engine,device='cuda')
            raw,state,reference,proof=qualified_observation(args.engine,qkv,plain=plain,capture=capture,targets=support['contexts'][args.engine],synchronize=torch.cuda.synchronize)
        analysis={'group6_vs_group18':compare_lse(raw['g6.lse'],raw['g18.lse'])}
        if args.engine=='native':
            analysis['old_vs_native']={g:compare_lse(old_raw[g+'.lse'],raw[g+'.lse']) for g in ('g6','g18')}
            analysis['row_overlap']={g:row_cross_tab(old_raw[g+'.lse'],raw[g+'.lse'],support['contexts']['official'],support['contexts']['native']) for g in ('g6','g18')}
            analysis['mathematical_reference_distances']={e+'.'+g:_reference_stats(values[g+'.lse'],reference)
                for e,values in (('official',old_raw),('native',raw)) for g in ('g6','g18')}
        result={'kind':'hy_weightless_actual_QKV_LSE','status':'LSE_capture_complete','engine':args.engine,'job_id':os.environ['SLURM_JOB_ID'],
            'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch_git,'bf16_reduced_precision_reduction':reduction_flag,
            'input_manifest':manifest,'source_records':sources,'API_binding':API_proof,'observations':state,'raw_proof':proof,'analysis':analysis,
            'model_constructed':False,'learned_weight_read_events':0,'checkpoint_identity_stat_only':support['checkpoint_identity_stat_only'],
            'nonweight_tensor_read_events':support['nonweight_tensor_read_events'],'selected_input_roles':support['selected_input_roles'],
            'performance_measurement':False,'softmax_scale_logical_head_dim':72,'softmax_scale_python':SCALE,
            'CPU_reference_scale_promoted_FP32':float(torch.tensor(SCALE,dtype=torch.float32)),
            'scope':'Fixed actual QKV LSE scalar observation; equal LSE does not prove equal scores/probabilities; unequal LSE does not establish a primitive or policy cause.'}
        if args.engine=='native':result['official_predecessor']={name:file_record(args.official_lse/name) for name in ('result.json','operator_outputs.safetensors')}
        publish_capture(args.output,raw,result,qkv_reference=support['QKV'],target_reference=support['contexts'][args.engine],reference=reference)
        print(json.dumps({'engine':args.engine,'actual_arrays':len(raw),'CPU_reference_arrays':int(reference is not None),'learned_weight_reads':0,
            'group_LSE_numeric_unequal':analysis['group6_vs_group18']['numeric_unequal']}),flush=True)
    except BaseException as error:
        write_failure(args.output,error,raw=raw,observations=state,reference=reference);raise
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--official-lse',type=Path);parser.add_argument('--source-revision',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);_module(Path(__file__).with_name('diagnose.py'),'hy_lse_allocation_guard').require_step()
    import platform,sys
    import numpy as np
    import torch
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact R40 source revision required')
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Weightless LSE requires admitted A800')
    torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    return run(args,runtime,torch.version.git_version,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__=='__main__':raise SystemExit(main())
