"""Actual block0 attention context and local projection cuts, not a policy."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import MethodType

PREFIX='model.dual_tower.vlm.model.visual.vision_tower.blocks.0.attn.'
ACTUAL_FIELDS=('norm1','qkv','context','projection','api','control','self_projection')


def attention_keys():
    return {'qkv.weight':(3456,1152),'qkv.bias':(3456,),'proj.weight':(1152,1152),'proj.bias':(1152,)}


def read_attention_weights(reader,*,device):
    import torch
    result={}
    for key,shape in attention_keys().items():
        value=reader.get_tensor(PREFIX+key)
        if value.dtype!=torch.bfloat16 or value.shape!=shape or not torch.isfinite(value).all():raise ValueError('Attention weight requires admitted finite BF16 shape: '+key)
        result[key]=value.to(device=device)
    return result


def _same(a,b):
    import torch
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)


def _layout(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype),'live_stride':list(value.stride()),'storage_offset':value.storage_offset()}


def _record(raw,state,name,value):
    import torch
    if name in raw or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('Duplicate or invalid attention bridge field: '+name)
    state['layouts'][name]=_layout(value);raw[name]=value.detach().cpu().contiguous().clone()


def load_native_methods(source,*,observer=None):
    import torch
    import torch.nn.functional as F
    tree=ast.parse(Path(source).read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HyVLAThorBF16Pipeline']
    signatures={'_vit_qkv':'self, h','_vit_spatial_attn':'self, q, k, v'}
    selected=[n for c in classes for n in c.body if isinstance(n,ast.FunctionDef) and n.name in signatures]
    if len(classes)!=1 or len(selected)!=2 or {n.name for n in selected}!=set(signatures):raise ValueError('Attention source requires two unique actual methods')
    for n in selected:
        if n.decorator_list or ast.dump(n.args)!=ast.dump(ast.parse(f'def f({signatures[n.name]}):pass').body[0].args):raise ValueError('Attention native method signature/source drift')
    nodes=copy.deepcopy(selected)
    if observer is not None:
        for node in nodes:
            calls=[n for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                and isinstance(n.func.value,ast.Name) and n.func.value.id=='F' and n.func.attr=='linear']
            if len(calls)!=1:raise ValueError('Attention observation AST Linear call count drift')
            calls[0].func=ast.Name(id='_bridge_qkv' if node.name=='_vit_qkv' else '_bridge_projection',ctx=ast.Load())
        reverse=copy.deepcopy(nodes)
        for node in reverse:
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id in ('_bridge_qkv','_bridge_projection'):
                    call.func=ast.Attribute(value=ast.Name(id='F',ctx=ast.Load()),attr='linear',ctx=ast.Load())
        if ast.dump(ast.Module(body=reverse,type_ignores=[]))!=ast.dump(ast.Module(body=selected,type_ignores=[])):raise ValueError('Attention inverse observation AST changed math')
    def qkv(*a,**kw):
        out=F.linear(*a,**kw);observer('qkv',a,kw,out);return out
    def projection(*a,**kw):
        observer('context',a,kw,None);out=F.linear(*a,**kw);observer('projection',a,kw,out);return out
    scope={'torch':torch,'F':F,'_bridge_qkv':qkv,'_bridge_projection':projection}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),scope)
    return {name:scope[name] for name in signatures}


def _weights(owner,engine):
    if engine=='official':return [owner.qkv.weight,owner.qkv.bias,owner.proj.weight,owner.proj.bias]
    return [owner._vit_qkv_w_cur,owner._vit_qkv_b_cur,owner._vit_proj_w_cur,owner._vit_proj_b_cur]


def capture_actual(engine,owner,norm1,*,synchronize):
    import torch
    import torch.nn.functional as F
    if engine not in ('official','native'):raise ValueError('Unknown attention bridge engine')
    if getattr(owner,'_attention_bridge_active',False):raise ValueError('Nested attention bridge capture is active')
    if engine=='native' and getattr(owner,'_vit_f8',None) is not False:raise ValueError('Attention bridge requires original BF16 non-FP8 path')
    values=norm1 if engine=='official' else (norm1,);weights=_weights(owner,engine);C=weights[0].shape[1]
    if (not isinstance(values,tuple) or len(values)!=(3 if engine=='official' else 1)
            or any(v.dtype!=torch.bfloat16 or v.shape!=((6 if engine=='official' else 18),196,C) or not torch.isfinite(v).all() for v in values)
            or any(w.dtype!=torch.bfloat16 or w.device!=values[0].device or not torch.isfinite(w).all() for w in weights)):
        raise ValueError('Actual attention norm1/weight BF16 geometry differs')
    if engine=='official' and any(m.training for m in owner.modules()):raise ValueError('Official actual attention must remain eval')
    raw={};state={'engine':engine,'complete':False,'cleanup':{'passed':False},'layouts':{},'api_calls':{'control':0,'observed':0},
        'observed_qkv_calls':0,'observed_projection_calls':0,'context_aliases_fixed_inputs':False}
    before=[v.clone() for v in values];weight_before=[v.clone() for v in weights];handles=[];saved={};missing=object();primary=None
    active=owner.__dict__.get('_attention_bridge_active',missing);hold={'camera':None,'qkv':None};live=[]
    def stem():return f'camera{hold["camera"]}.' if engine=='official' else ''
    def record(key,value):_record(raw,state,stem()+key,value)
    def api(value):
        if engine=='official':return owner(value)
        return owner._vit_spatial_attn(*owner._vit_qkv(value))
    def context(value):
        protected=(values[hold['camera']],hold['qkv'],*weights) if engine=='official' else (norm1,hold['qkv'],*weights)
        if any(value.untyped_storage().data_ptr()==v.untyped_storage().data_ptr() for v in protected):
            state['context_aliases_fixed_inputs']=True
            raise ValueError('Actual context aliases fixed norm1/QKV/weight storage')
        record('context',value);live.append(value)
    def event(kind,a,kw,out):
        if kind=='qkv':state['observed_qkv_calls']+=1;hold['qkv']=out;record('qkv',out)
        elif kind=='context':context(a[0])
        else:state['observed_projection_calls']+=1;record('projection',out)
    try:
        owner._attention_bridge_active=True
        for i,value in enumerate(values):
            hold['camera']=i;record('norm1',value);out=api(value);state['api_calls']['control']+=1;record('control',out)
        if engine=='official':
            def qkv_hook(m,a,out):event('qkv',a,{},out)
            def context_hook(m,a):event('context',a,{},None)
            def projection_hook(m,a,out):event('projection',a,{},out)
            handles=[owner.qkv.register_forward_hook(qkv_hook),owner.proj.register_forward_pre_hook(context_hook),owner.proj.register_forward_hook(projection_hook)]
        else:
            for name,method in load_native_methods(owner._bridge_source,observer=event).items():
                saved[name]=owner.__dict__.get(name,missing);setattr(owner,name,MethodType(method,owner))
        for i,value in enumerate(values):
            hold.update(camera=i,qkv=None);out=api(value);state['api_calls']['observed']+=1;record('api',out)
        for handle in handles:handle.remove()
        handles=[]
        for name,value in saved.items():
            if value is missing:owner.__dict__.pop(name,None)
            else:setattr(owner,name,value)
        saved={}
        if len(live)!=len(values):raise ValueError('Actual context call coverage differs')
        for i,value in enumerate(live):
            hold['camera']=i;out=owner.proj(value) if engine=='official' else F.linear(value,owner._vit_proj_w_cur,owner._vit_proj_b_cur)
            record('self_projection',out);base=stem()
            if any(not _same(raw[base+'api'],raw[base+k]) for k in ('projection','control','self_projection')):raise ValueError('Actual attention API/projection/control/self replay differs')
        if (state['observed_qkv_calls']!=len(values) or state['observed_projection_calls']!=len(values) or len(raw)!=7*len(values)
                or any(not _same(v,b) for v,b in zip(values,before)) or any(not _same(w,b) for w,b in zip(weights,weight_before))):
            raise ValueError('Actual attention call coverage or readonly input/weight changed')
        state['complete']=True
    except BaseException as error:
        primary=error;error.bridge_partial=(raw,state);raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        for handle in handles:
            try:handle.remove()
            except BaseException as error:errors.append('hook cleanup failed: '+str(error))
        for name,value in saved.items():
            try:
                if value is missing:owner.__dict__.pop(name,None)
                else:setattr(owner,name,value)
            except BaseException as error:errors.append('method cleanup failed: '+str(error))
        if active is missing:owner.__dict__.pop('_attention_bridge_active',None)
        else:owner._attention_bridge_active=active
        state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False
            if primary is not None:primary.add_note('; '.join(errors))
            else:
                error=RuntimeError('Actual attention cleanup failed: '+'; '.join(errors));error.bridge_partial=(raw,state);raise error
    return raw,state,tuple(live) if engine=='official' else live[0]


def _cut_operands(context,old,weights,native_reference):
    import torch
    if context.ndim!=3:raise ValueError('Projection cut context geometry differs')
    B,N,C=context.shape;spec={'qkv.weight':(3*C,C),'qkv.bias':(3*C,),'proj.weight':(C,C),'proj.bias':(C,)}
    if ((B,N)!=(18,196) or context.dtype!=torch.bfloat16 or old.dtype!=torch.bfloat16 or old.shape!=context.shape
            or native_reference.dtype!=torch.bfloat16 or native_reference.shape!=context.shape or set(weights)!=set(spec)
            or any(v.dtype!=torch.bfloat16 or v.shape!=spec[k] or v.device!=context.device or not torch.isfinite(v).all() for k,v in weights.items())
            or any(not torch.isfinite(v).all() for v in (context,old,native_reference))):raise ValueError('Projection cut BF16/finite/weight geometry differs')
    span=1
    for stride,size in sorted((s,n) for s,n in zip(context.stride(),context.shape) if n>1):
        if stride<span:raise ValueError('Projection cut context layout overlaps')
        span+=stride*(size-1)
    if any(context.untyped_storage().data_ptr()==v.untyped_storage().data_ptr() for v in weights.values()):raise ValueError('Projection cut context aliases weight storage')


def projection_cuts(live_context,old_context,weights,*,native_reference,synchronize):
    import torch
    import torch.nn.functional as F
    _cut_operands(live_context,old_context,weights,native_reference)
    context=live_context;before=context.clone();old=old_context.to(device=context.device).clone();old_before=old_context.clone()
    initial_weights={k:v.clone() for k,v in weights.items()};initial_layout=_layout(context);storage=context.untyped_storage().data_ptr()
    raw={};state={'complete':False,'cleanup':{'passed':False},'layouts':{},'cells':{},'original_context_layout':initial_layout,
        'projection_calls':0,'restore_projection_calls':0,'restored_context_exact':False,'controlled_context_value_substitution':True};primary=None
    def record(key,value):_record(raw,state,'cuts.'+key,value)
    try:
        for cell in ('N18','N6','O18','O6'):
            value=old if cell.startswith('O') else before;context.copy_(value)
            if not _same(context,value) or _layout(context)!=initial_layout:raise ValueError('Projection cut value/layout substitution failed')
            record(cell+'.input',context);state['cells'][cell]={'views':[],'output_parts':[]};outputs=[]
            operands=[context[i:i+6] for i in (0,6,12)] if cell.endswith('6') else [context]
            for view in operands:
                shares=view.untyped_storage().data_ptr()==storage;state['cells'][cell]['views'].append({**_layout(view),'shares_live_storage':shares})
                if not shares:raise ValueError('Projection cut operand no longer shares actual context storage')
                state['projection_calls']+=1;out=F.linear(view,weights['proj.weight'],weights['proj.bias'])
                state['cells'][cell]['output_parts'].append(_layout(out));outputs.append(out)
            if len(outputs)==1:record(cell+'.output',outputs[0])
            else:
                # Only the result artifact is concatenated; the actual GEMM
                # operands are the three views above, never a new input clone.
                record(cell+'.output',torch.cat([v.detach().cpu() for v in outputs],0))
            state['cells'][cell]['output_concatenation_is_CPU_artifact']=len(outputs)==3
            if not _same(context,value):raise ValueError('Projection cut readonly operand changed within Linear')
            if cell=='N18' and not _same(raw['cuts.N18.output'],native_reference.detach().cpu()):raise ValueError('N18 native self-value control differs before other cuts')
        if any(not _same(v,initial_weights[k]) for k,v in weights.items()):raise ValueError('Projection cut readonly weight mutated')
        context.copy_(before);record('restored_context',context)
        restored=F.linear(context,weights['proj.weight'],weights['proj.bias']);state['restore_projection_calls']=1;record('restore_projection',restored)
        if not _same(raw['cuts.restore_projection'],native_reference.detach().cpu()):raise ValueError('Projection cut post-restore native control differs')
        if len(raw)!=10 or state['projection_calls']!=8:raise ValueError('Projection cut raw/call coverage incomplete')
        state['complete']=True
    except BaseException as error:
        primary=error;error.bridge_partial=(raw,state);raise
    finally:
        errors=[]
        try:
            context.copy_(before);state['restored_context_exact']=_same(context,before) and _layout(context)==initial_layout
            if not state['restored_context_exact']:raise ValueError('actual live context not restored')
            if 'cuts.restored_context' not in raw:record('restored_context',context)
        except BaseException as error:errors.append('context restore failed: '+str(error))
        try:
            if any(not _same(v,initial_weights[k]) for k,v in weights.items()) or not _same(old_context,old_before):raise ValueError('readonly weight/old-context mutation')
        except BaseException as error:errors.append('readonly cleanup failed: '+str(error))
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False
            if primary is not None:primary.add_note('; '.join(errors))
            else:
                error=RuntimeError('Projection cut cleanup failed: '+'; '.join(errors));error.bridge_partial=(raw,state);raise error
    return raw,state


def _difference(a,b):
    import torch
    if a.dtype!=b.dtype or a.shape!=b.shape:raise ValueError('Projection comparison shape/dtype differs')
    x,y=a.detach().cpu().double(),b.detach().cpu().double();d=x-y;den=float(x.norm())*float(y.norm())
    return {'exact':_same(a,b),'unequal_elements':int(torch.count_nonzero(a!=b)),'max_abs':float(d.abs().max()),
        'rmse':float(d.square().mean().sqrt()),'relative_l2':float(d.norm()/max(float(y.norm()),1e-12)),
        'cosine':float((x*y).sum())/den if den else float(_same(a,b))}


def qualify_cuts(raw,state,*,native_reference,old_reference):
    import torch
    if (state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True or state.get('restored_context_exact') is not True
            or state.get('projection_calls')!=8 or state.get('restore_projection_calls')!=1):raise ValueError('Projection cut coverage/cleanup incomplete')
    expected={f'cuts.{cell}.{field}' for cell in ('N18','N6','O18','O6') for field in ('input','output')}|{'cuts.restored_context','cuts.restore_projection'}
    if set(raw)!=expected:raise ValueError('Projection cut raw role set incomplete')
    if any(v.dtype!=torch.bfloat16 or v.shape!=native_reference.shape or not torch.isfinite(v).all() for v in raw.values()):raise ValueError('Projection cut raw dtype/shape/finite differs')
    if set(state.get('cells',{}))!={'N18','N6','O18','O6'}:raise ValueError('Projection cut cell coverage differs')
    layout=state['original_context_layout']
    for cell in state['cells']:
        count=3 if cell.endswith('6') else 1;views=state['cells'][cell].get('views',[])
        if len(views)!=count or any(v.get('shares_live_storage') is not True for v in views):raise ValueError('Projection cut view/storage proof differs')
        for i,v in enumerate(views):
            shape=[6 if count==3 else 18,196,native_reference.shape[-1]]
            if (v['shape']!=shape or v['live_stride']!=layout['live_stride'] or v['dtype']!='torch.bfloat16'
                    or v['storage_offset']!=layout['storage_offset']+(i*6*layout['live_stride'][0] if count==3 else 0)):
                raise ValueError('Projection cut actual input view geometry/offset differs')
    if not _same(raw['cuts.N6.input'],raw['cuts.N18.input']) or not _same(raw['cuts.O6.input'],raw['cuts.O18.input']):raise ValueError('Projection cut fixed-context input pair differs')
    if not _same(raw['cuts.N18.output'],native_reference) or not _same(raw['cuts.restore_projection'],native_reference) or not _same(raw['cuts.restored_context'],raw['cuts.N18.input']):
        raise ValueError('Projection native control/restoration failed')
    comparisons={cell:{'native':_difference(raw[f'cuts.{cell}.output'],native_reference),'old':_difference(raw[f'cuts.{cell}.output'],old_reference)} for cell in ('N18','N6','O18','O6')}
    return {'native_control_and_restoration_exact':True,'old_corner_exact':comparisons['O6']['old']['exact'],'comparisons':comparisons,
        'scope':'Negativeoldcornerisavalidoutcome,currentlayoutnewruntimeisnotisolatedruntimecause.'}


def interaction(cuts):
    import torch
    keys={f'cuts.{cell}.output' for cell in ('N18','N6','O18','O6')}
    if not keys<=set(cuts) or any(cuts[k].dtype!=torch.bfloat16 for k in keys):raise ValueError('Projection interaction requires four actual BF16 outputs')
    values={cell:cuts[f'cuts.{cell}.output'].detach().cpu().double() for cell in ('N18','N6','O18','O6')}
    if any(v.shape!=values['N18'].shape or not torch.isfinite(v).all() for v in values.values()):raise ValueError('Projection interaction shape/finite differs')
    value=(values['O6']-values['O18'])-(values['N6']-values['N18'])
    return value,{'shape':list(value.shape),'dtype':str(value.dtype),'nonzero_elements':int(torch.count_nonzero(value)),
        'max_abs':float(value.abs().max()),'l2_norm':float(value.norm()),'exactly_zero':bool(torch.count_nonzero(value)==0)}


def validate_capture(raw,state,*,engine,references):
    import torch
    if engine not in ('official','native'):raise ValueError('Unknown actual capture engine')
    count=3 if engine=='official' else 1;B=6 if engine=='official' else 18
    expected={((f'camera{i}.' if engine=='official' else '')+field):(B,196,3456 if field=='qkv' else 1152) for i in range(count) for field in ACTUAL_FIELDS}
    if (set(raw)!=set(expected) or any(v.device.type!='cpu' or v.dtype!=torch.bfloat16 or v.shape!=expected[k] or not torch.isfinite(v).all() for k,v in raw.items())):
        raise ValueError('Actual attention capture raw role/dtype/shape/finite differs')
    if (state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True or state.get('api_calls')!={'control':count,'observed':count}
            or state.get('observed_qkv_calls')!=count or state.get('observed_projection_calls')!=count
            or state.get('context_aliases_fixed_inputs') is not False or set(state.get('layouts',{}))!=set(raw)):
        raise ValueError('Actual attention capture calls/cleanup/alias proof differs')
    for i in range(count):
        base=f'camera{i}.' if engine=='official' else ''
        for field in ACTUAL_FIELDS:
            layout=state['layouts'][base+field]
            if layout['shape']!=list(expected[base+field]) or layout['dtype']!='torch.bfloat16' or len(layout['live_stride'])!=3 or layout['storage_offset']<0:
                raise ValueError('Actual attention capture raw layout role differs')
        for key,target in (('norm1','norm1'),('qkv','qkv'),('projection','post'),('api','post'),('control','post'),('self_projection','post')):
            ref=references[target][i] if engine=='official' else references[target]
            if not _same(raw[base+key],ref):raise ValueError('Actual attention capture differs from R38 reference anchor: '+key)
    return {'tensors':len(raw),'R38_norm1_qkv_exact':True,'API_control_projection_self_exact_R38':True}


def validate_bundle(raw,*,engine,actual_state,references,cut_state=None,old_context=None,old_reference=None,interaction_array=None):
    import torch
    if engine=='official':return validate_capture(raw,actual_state,engine=engine,references=references)
    if engine!='native' or old_context is None or old_reference is None:raise ValueError('Native bridge bundle requires actual old context/API references')
    actual={k:v for k,v in raw.items() if not k.startswith('cuts.')};cuts={k:v for k,v in raw.items() if k.startswith('cuts.')}
    proof=validate_capture(actual,actual_state,engine='native',references=references)
    if len(raw)!=17:raise ValueError('Native bridge bundle requires exact17 raw roles')
    qualification=qualify_cuts(cuts,cut_state or {},native_reference=references['post'],old_reference=old_reference)
    if (not _same(cuts['cuts.N18.input'],actual['context']) or not _same(cuts['cuts.N6.input'],actual['context'])
            or not _same(cuts['cuts.O18.input'],old_context) or not _same(cuts['cuts.O6.input'],old_context)):
        raise ValueError('Projection cut input values differ from their actual source contexts')
    expected,stats=interaction(cuts)
    if (interaction_array is None or interaction_array.dtype!=torch.float64 or interaction_array.device.type!='cpu'
            or interaction_array.shape!=(18,196,1152) or not _same(interaction_array,expected)):
        raise ValueError('Native bridge bundle actual interaction array differs')
    return {'tensors':len(raw),'actual':proof,'cuts':qualification,'interaction_exact':True,'interaction_stats':stats}
