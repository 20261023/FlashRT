"""Weightless, fixed actual QKV Flash LSE observation; never a policy change."""
from __future__ import annotations

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
    if engine=='native' and (any(not isinstance(result[i],torch.Tensor) for i in (2,3,6,7))
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
                observed,normalizer,meta=parse_capture_result(result,engine=engine,batch=size);entry['capture_returns']=meta
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
                    or any('shape' not in returns[j] for j in (2,3,6,7))):raise ValueError('LSE native return schema metadata differs')
    return {'tensors':8,'BF16_arrays':6,'FP32_LSE_arrays':2,'QKV_readonly_exact':True,'all_contexts_exact_R39':True,'learned_weight_read_events':0}
