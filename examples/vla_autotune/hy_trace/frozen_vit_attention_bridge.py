"""Actual block0 attention context and local projection cuts, not a policy."""
from __future__ import annotations

import ast
import copy
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType,SimpleNamespace

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
            handles.append(owner.qkv.register_forward_hook(qkv_hook))
            handles.append(owner.proj.register_forward_pre_hook(context_hook))
            handles.append(owner.proj.register_forward_hook(projection_hook))
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


def _validate_weights(weights):
    import torch
    shapes=attention_keys()
    if set(weights)!=set(shapes):raise ValueError('Attention weight key set must contain exactly four tensors')
    device=weights['qkv.weight'].device
    if any(v.dtype!=torch.bfloat16 or v.shape!=shapes[k] or v.device!=device or not torch.isfinite(v).all() for k,v in weights.items()):
        raise ValueError('Attention weight finite/device/shape/BF16 contract differs')


def make_native_owner(weights,*,source):
    _validate_weights(weights)
    owner=SimpleNamespace(_vit_f8=False,vit_heads=16,vit_hd=72,vit_scale=72**-.5,_bridge_source=Path(source),
        _vit_qkv_w_cur=weights['qkv.weight'],_vit_qkv_b_cur=weights['qkv.bias'],_vit_proj_w_cur=weights['proj.weight'],_vit_proj_b_cur=weights['proj.bias'])
    for name,fn in load_native_methods(source).items():setattr(owner,name,MethodType(fn,owner))
    return owner


def make_official_attention(vision,weights,*,device):
    import torch
    _validate_weights(weights)
    owner=vision._ViTAttention(dim=1152,num_heads=16,qkv_bias=True,qk_norm=False,attn_drop=0.,proj_drop=0.,seperate_qv_bias=False).to(device=device,dtype=torch.bfloat16)
    if (set(owner.state_dict())!=set(weights) or owner.q_bias is not None or not isinstance(owner.q_norm,torch.nn.Identity)
            or not isinstance(owner.k_norm,torch.nn.Identity) or (owner.num_heads,owner.head_dim,owner.scale)!=(16,72,72**-.5)
            or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in owner.modules())):raise ValueError('Actual official attention parameter/head/scale/drop/bias routing differs')
    owner.load_state_dict(weights,strict=True);owner.eval()
    for p in owner.parameters():p.requires_grad_(False)
    if any(not _same(value,weights[key]) for key,value in owner.state_dict().items()):raise ValueError('Actual official attention loaded parameter values differ')
    return owner


def file_record(path):
    p=Path(path).resolve();s=p.stat();return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def admit_before_weights(manifest,*,engine,runtime,torch_git,reduction_flag,current_native,current_prefix,factory):
    if manifest.get('kind')!='hy_actual_vit_attention_bridge_inputs' or engine not in ('official','native'):raise ValueError('R39 input kind/engine differs')
    task=Path(manifest['task_root']).resolve();root=task/'results/hy-vit-prefix-318174'
    roles={'SOURCE_GIT_COMMIT','inputs.json','official/result.json','official/operator_outputs.safetensors','native/result.json','native/operator_outputs.safetensors'}
    records=manifest.get('R38_records',{})
    if set(records)!=roles:raise ValueError('R38 actual record roles differ')
    for role,record in records.items():
        expected=root/role
        if Path(record['path']).resolve()!=expected or file_record(expected)!=record:raise ValueError('R38 record path/stat changed')
    if (root/'SOURCE_GIT_COMMIT').read_text().strip()!='311c66d06c20224dd0a11ce6108fb4cb2ac76c62':raise ValueError('R38 actual source marker differs')
    prefix_record=manifest['frozen_prefix'];frozen=task/'snapshots/round038/code/examples/vla_autotune/hy_trace/frozen_vit_prefix.py'
    if Path(prefix_record['path']).resolve()!=frozen or file_record(frozen)!=prefix_record:raise ValueError('R38 frozen prefix namespace/path/stat differs')
    if frozen.read_bytes()!=Path(current_prefix).read_bytes():raise ValueError('R38 frozen prefix driver source changed')
    current=task/'snapshots/round039/code/flash_rt/models/hyvla/pipeline_thor.py'
    if Path(current_native).resolve()!=current or Path(manifest['current_native']['path']).resolve()!=current or file_record(current)!=manifest['current_native']:
        raise ValueError('Current native source path/stat differs')
    reports={e:json.loads((root/e/'result.json').read_text()) for e in ('official','native')};inputs=json.loads((root/'inputs.json').read_text())
    prefix=_module(frozen,'hy_bridge_frozen_R38_prefix')
    native38=task/'snapshots/round038/code/flash_rt/models/hyvla/pipeline_thor.py'
    historical=prefix.admit_before_weights(inputs,native_source=native38,factory=lambda old:old)
    expected_sources=prefix._source_records(inputs)
    if current.read_bytes()!=native38.read_bytes():raise ValueError('Current native source differs from R38 actual implementation')
    for e,count in (('official',124),('native',42)):
        r=reports[e];state=r.get('observations',{})
        if (r.get('kind')!='hy_actual_frozen_vit_prefix' or r.get('status')!='prefix_capture_complete' or r.get('engine')!=e
                or r.get('job_id')!='318174' or r.get('source_revision')!='311c66d06c20224dd0a11ce6108fb4cb2ac76c62'
                or r.get('input_manifest')!=inputs or r.get('source_records')!=expected_sources or r.get('checkpoint')!=historical['checkpoint']
                or r.get('full_model_constructed') is not False or r.get('blocks_constructed')!=4 or r.get('weight_read_events')!=51
                or r.get('raw_proof')!={'tensors':count,'block_chain_exact':True,'residual_chains_exact':True,'actual_endpoint_bridge_exact':True}
                or state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True):raise ValueError('R38 prefix source/header/checkpoint/cleanup differs')
        if e=='official' and state.get('camera_block_indices')!=[[0,1,2,3]]*3:raise ValueError('R38 actual official call coverage differs')
        if e=='native' and (state.get('control_block_indices')!=[0,1,2,3] or state.get('observed_block_indices')!=[0,1,2,3]
                or state.get('block4_called') is not False or state.get('normal_vit_forward_return_reached') is not False
                or r.get('comparison',{}).get('first_different_stage')!='block0.attention'):raise ValueError('R38 actual native boundary/call proof differs')
    if reports[engine].get('runtime')!=runtime or reports[engine].get('torch_git_version')!=torch_git or reports[engine].get('bf16_reduced_precision_reduction')!=reduction_flag:
        raise ValueError('R38/current attention runtime/Git/flags differ')
    return factory(prefix,reports)


def admit_legacy_header(report,*,task,job,engine,checkpoint,runtime,reduction_flag):
    if (job not in ('309182','309409') or engine not in ('official','native')
            or report.get('kind')!='diagnostic_frozen_block0_operators_not_recorded_block0_or_action_acceptance'
            or report.get('job_id')!=job or report.get('engine')!=engine or report.get('torch')!=runtime['torch'] or report.get('cuda')!=runtime['cuda']
            or report.get('gpu')!='NVIDIA A800-SXM4-80GB' or report.get('model_constructed') is not False or report.get('performance_measurement') is not False
            or report.get('bf16_reduced_precision_reduction')!=reduction_flag or Path(report['checkpoint']).resolve()!=Path(checkpoint['path']).resolve()
            or report.get('checkpoint_stat')!={k:checkpoint[k] for k in ('bytes','mtime_ns')}
            or Path(report['official_trace']).resolve()!=Path(task)/'results/hy-trace-308690/official/baseline'):
        raise ValueError('Legacy context source/input/runtime/checkpoint binding differs')


def capture_and_cut(engine,owner,norm1,weights,*,references,old_context=None,old_reference=None,synchronize):
    raw={};observations={'actual':None,'cuts':None};phase='actual';I=None
    try:
        actual,state,live=capture_actual(engine,owner,norm1,synchronize=synchronize)
        raw.update(actual);observations['actual']=state;phase='actual_anchor'
        validate_capture(actual,state,engine=engine,references=references)
        if engine=='native':
            phase='cuts'
            cuts,cut_state=projection_cuts(live,old_context,weights,native_reference=references['post'],synchronize=synchronize)
            raw.update(cuts);observations['cuts']=cut_state;phase='analysis'
            I,_=interaction(cuts)
        proof=validate_bundle(raw,engine=engine,actual_state=state,references=references,cut_state=observations['cuts'],
            old_context=old_context,old_reference=old_reference,interaction_array=I)
        return raw,observations,I,proof
    except BaseException as error:
        partial=getattr(error,'bridge_partial',None)
        if partial is not None:
            raw.update(partial[0])
            if phase=='actual':observations['actual']=partial[1]
            elif phase=='cuts':observations['cuts']=partial[1]
        observations['failure_phase']=phase;error.bridge_partial=(raw,observations);raise


def write_failure(output,error,*,raw=None,observations=None,interaction_array=None):
    from safetensors.torch import save_file
    retained=dict(raw or {});state=dict(observations or {});partial=getattr(error,'bridge_partial',None)
    if partial is not None:retained.update(partial[0]);state.update(partial[1])
    failures=[]
    if retained:
        try:save_file({k:v.contiguous() for k,v in retained.items()},str(Path(output)/'unqualified_partial.safetensors'))
        except BaseException as secondary:failures.append('raw evidence write failed: '+str(secondary));error.add_note(failures[-1])
    if interaction_array is not None:
        try:save_file({'projection_interaction':interaction_array.contiguous()},str(Path(output)/'unqualified_interaction.safetensors'))
        except BaseException as secondary:failures.append('interaction evidence write failed: '+str(secondary));error.add_note(failures[-1])
    record={'status':'unqualified_vit_attention_bridge_failure','qualified':False,'error_type':type(error).__name__,'error':str(error),
        'observations':state,'captured_raw_roles':list(retained),'evidence_write_errors':failures}
    try:(Path(output)/'failure.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    except BaseException as secondary:error.add_note('failure metadata write failed: '+str(secondary))


def admit_official_bridge(report,raw,*,job_id,source_revision,runtime,torch_git,reduction_flag,checkpoint,inputs,source_records,references,factory):
    obs=report.get('observations',{})
    if (report.get('kind')!='hy_actual_vit_attention_context_projection_bridge' or report.get('status')!='bridge_complete'
            or report.get('engine')!='official' or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision
            or report.get('runtime')!=runtime or report.get('torch_git_version')!=torch_git or report.get('bf16_reduced_precision_reduction')!=reduction_flag
            or report.get('checkpoint')!=checkpoint or report.get('input_manifest')!=inputs or report.get('source_records')!=source_records
            or report.get('weight_read_events')!=4 or len(report.get('selected_checkpoint_keys',[]))!=4
            or set(report['selected_checkpoint_keys'])!={PREFIX+k for k in attention_keys()} or report.get('full_model_constructed') is not False
            or report.get('attention_module_constructed') is not True or obs.get('cuts') is not None):
        raise ValueError('Official actual bridge predecessor source/runtime/header differs')
    proof=validate_bundle(raw,engine='official',actual_state=obs.get('actual',{}),references=references)
    if report.get('raw_proof')!=proof:raise ValueError('Official actual bridge predecessor raw proof differs')
    return factory()


def publish_capture(output,raw,report,*,references,old_context=None,old_reference=None,interaction_array=None):
    from safetensors.torch import save_file
    obs=report['observations'];proof=validate_bundle(raw,engine=report['engine'],actual_state=obs['actual'],references=references,
        cut_state=obs.get('cuts'),old_context=old_context,old_reference=old_reference,interaction_array=interaction_array)
    if report.get('raw_proof',proof)!=proof:raise ValueError('Bridge raw proof differs before publication')
    save_file({k:v.contiguous() for k,v in raw.items()},str(Path(output)/'operator_outputs.safetensors'))
    if report['engine']=='native':save_file({'projection_interaction':interaction_array.contiguous()},str(Path(output)/'projection_interaction.safetensors'))
    temp=Path(output)/'result.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(Path(output)/'result.json')


LEGACY={'309182':('hy-frozen-attention-309182','ddbb5faa2ce76ee4fe1cf78016bd60f210f88c0c'),
    '309409':('hy-sdpa-backends-309409','d78ecfb07fc91fadff427d01ab44a8682639628b')}


def _load_legacy(manifest,*,task,reports,checkpoint,qkv):
    import torch
    from safetensors import safe_open
    records=manifest.get('legacy_records',{});expected={}
    for job,(folder,_) in LEGACY.items():
        expected[job+'.source']=task/'results'/folder/'SOURCE_GIT_COMMIT'
        for engine in ('official','native'):
            for role,name in (('report','result.json'),('raw','operator_outputs.safetensors')):expected[f'{job}.{engine}.{role}']=task/'results'/folder/engine/name
    if set(records)!=set(expected):raise ValueError('Legacy context record roles incomplete')
    for role,path in expected.items():
        if Path(records[role]['path']).resolve()!=path or file_record(path)!=records[role]:raise ValueError('Legacy context record path/stat differs')
    result={}
    for job,(_,source) in LEGACY.items():
        if expected[job+'.source'].read_text().strip()!=source:raise ValueError('Legacy context actual source marker differs')
        for engine in ('official','native'):
            report=json.loads(expected[f'{job}.{engine}.report'].read_text())
            admit_legacy_header(report,task=task,job=job,engine=engine,checkpoint=checkpoint,runtime=reports[engine]['runtime'],
                reduction_flag=reports[engine]['bf16_reduced_precision_reduction'])
            role='attention_flash_per_camera' if engine=='official' else 'attention_sdpa_combined'
            with safe_open(str(expected[f'{job}.{engine}.raw']),framework='pt',device='cpu') as sf:
                previous=sf.get_tensor('qkv_per_camera');context=sf.get_tensor(role)
            if (not _same(previous,qkv) or context.dtype!=torch.bfloat16 or context.shape!=(18,196,1152)
                    or not torch.isfinite(context).all()):raise ValueError('Legacy actual-QKV/context dtype/shape/finite bridge differs')
            result[job+'.'+engine]=context
    return result


def load_inputs(manifest,*,engine,runtime,torch_git,reduction_flag):
    import torch
    from safetensors.torch import load_file
    task=Path(manifest['task_root']).resolve();root=task/'results/hy-vit-prefix-318174'
    current_native=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
    def consume(prefix,reports):
        old_inputs=json.loads((root/'inputs.json').read_text())
        base=prefix.load_inputs(old_inputs,engine=engine,runtime=runtime,torch_git=torch_git)
        historical=base['history_manifest'];raws={e:load_file(str(root/e/'operator_outputs.safetensors')) for e in ('official','native')}
        for e in ('official','native'):
            ref=json.loads(Path(historical['records']['runtime.'+e]['path']).read_text())
            if (reports[e]['runtime']!=ref['runtime'] or reports[e]['torch_git_version']!=ref['torch_git_version']
                    or reports[e]['bf16_reduced_precision_reduction']!=ref['bf16_reduced_precision_reduction']):raise ValueError('R38 prefix historical runtime/flags binding differs')
        old_ref=json.loads(Path(historical['records']['runtime.official']['path']).read_text())
        prefix.admit_prior_official(reports['official'],raws['official'],base['gold'],job_id='318174',source_revision='311c66d06c20224dd0a11ce6108fb4cb2ac76c62',
            runtime=old_ref['runtime'],torch_git=old_ref['torch_git_version'],checkpoint=base['checkpoint'],input_manifest=old_inputs,
            source_records=prefix._source_records(old_inputs),factory=lambda:None)
        prefix.validate_raw(raws['native'],engine='native')
        imports=reports['official'].get('official_imports',{})
        if imports!={k:old_inputs['source_files']['official_'+k] for k in ('modeling','video')}:raise ValueError('R38 actual official imports differ')
        refs={};layouts={}
        for e in ('official','native'):
            values={};info={}
            for name,key in (('norm1','norm1'),('qkv','qkv'),('post','attention')):
                if e=='official':values[name]=tuple(raws[e][f'camera{i}.block0.'+key] for i in range(3))
                else:values[name]=raws[e]['block0.'+key]
            names=[f'camera{i}.block0.norm1' for i in range(3)] if e=='official' else ['block0.norm1']
            observed=[reports[e]['observations']['layouts'][key] for key in names]
            B=6 if e=='official' else 18
            if any(v!={'shape':[B,196,1152],'dtype':'torch.bfloat16','live_stride':[225792,1152,1],'storage_offset':0} for v in observed):
                raise ValueError('R38 actual observed norm1 layout differs from fixed context')
            refs[e]=values;layouts[e]=tuple(observed) if e=='official' else observed[0]
        for key in ('norm1','qkv'):
            if not _same(torch.cat(refs['official'][key],0),refs['native'][key]):raise ValueError('R38 common actual norm1/QKV values differ')
        legacy=_load_legacy(manifest,task=task,reports=reports,checkpoint=base['checkpoint'],qkv=refs['native']['qkv'])
        return {'prefix':prefix,'R38_reports':reports,'R38_inputs':old_inputs,'checkpoint':base['checkpoint'],'directory_checkpoint':base['directory_checkpoint'],
            'native_source':current_native,'references':refs,'norm1_layouts':layouts,'legacy_contexts':legacy}
    return admit_before_weights(manifest,engine=engine,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag,
        current_native=current_native,current_prefix=Path(__file__).with_name('frozen_vit_prefix.py'),factory=consume)


def restore_norm1(values,layouts,*,engine,device):
    tensors=values if engine=='official' else (values,);metadata=layouts if engine=='official' else (layouts,);out=[]
    if engine not in ('official','native') or len(tensors)!=(3 if engine=='official' else 1) or len(metadata)!=len(tensors):
        raise ValueError('Actual norm1 camera/layout coverage differs')
    for value,layout in zip(tensors,metadata):
        restored=value.to(device=device)
        if _layout(restored)!=layout:raise ValueError('Actual R38 norm1 layout/offset restoration failed; no artifact-layout fallback')
        out.append(restored)
    return tuple(out) if engine=='official' else out[0]


def _source_records(manifest,admitted):
    return {'driver':file_record(__file__),'native':manifest['current_native'],'frozen_R38_prefix':manifest['frozen_prefix'],
        'R38_components':admitted['R38_reports']['official']['source_records']}


def run(args,runtime,torch_git,reduction_flag):
    import sys
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file
    manifest=json.loads(args.inputs.read_text());task=Path(manifest['task_root']).resolve();root=task/'results'/('hy-vit-attention-bridge-'+os.environ['SLURM_JOB_ID'])
    if (args.inputs.resolve()!=root/'inputs.json' or args.output.resolve()!=root/args.engine
            or (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.source_revision):raise ValueError('R39 job/source/input/output root differs')
    admitted=load_inputs(manifest,engine=args.engine,runtime=runtime,torch_git=torch_git,reduction_flag=reduction_flag)
    sources=_source_records(manifest,admitted);prefix=admitted['prefix'];refs=admitted['references'][args.engine];old_context=old_reference=None
    if args.engine=='native':
        if args.official_bridge is None or args.official_bridge.resolve()!=root/'official':raise ValueError('Native bridge requires same-job actual official predecessor')
        old_report=json.loads((args.official_bridge/'result.json').read_text());old_raw=load_file(str(args.official_bridge/'operator_outputs.safetensors'))
        ref=admitted['R38_reports']['official']
        admit_official_bridge(old_report,old_raw,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,runtime=ref['runtime'],torch_git=ref['torch_git_version'],
            reduction_flag=ref['bf16_reduced_precision_reduction'],checkpoint=admitted['checkpoint'],inputs=manifest,source_records=sources,
            references=admitted['references']['official'],factory=lambda:None)
        old_context=torch.cat([old_raw[f'camera{i}.context'] for i in range(3)],0);old_reference=torch.cat([old_raw[f'camera{i}.api'] for i in range(3)],0)
        if any(n=='hy_vla' or n.startswith('hy_vla.') for n in sys.modules):raise ValueError('Native bridge must not import old Hy/FA2 package')
    args.output.mkdir(parents=True,exist_ok=False);raw={};observations={};I=None
    try:
        vision,video=prefix._official_modules(admitted['R38_inputs']) if args.engine=='official' else (None,None)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            with safe_open(admitted['checkpoint']['path'],framework='pt',device='cpu') as sf:weights=read_attention_weights(sf,device='cuda')
            norm1=restore_norm1(refs['norm1'],admitted['norm1_layouts'][args.engine],engine=args.engine,device='cuda')
            owner=make_official_attention(vision,weights,device='cuda') if args.engine=='official' else make_native_owner(weights,source=admitted['native_source'])
            raw,observations,I,proof=capture_and_cut(args.engine,owner,norm1,weights,references=refs,old_context=old_context,old_reference=old_reference,synchronize=torch.cuda.synchronize)
        context=torch.cat([raw[f'camera{i}.context'] for i in range(3)],0) if args.engine=='official' else raw['context']
        bridges={job:_difference(context,value) for name,value in admitted['legacy_contexts'].items() for job,e in (name.split('.'),) if e==args.engine}
        report={'kind':'hy_actual_vit_attention_context_projection_bridge','status':'bridge_complete','engine':args.engine,'job_id':os.environ['SLURM_JOB_ID'],
            'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch_git,'bf16_reduced_precision_reduction':reduction_flag,
            'input_manifest':manifest,'source_records':sources,'checkpoint':admitted['checkpoint'],'directory_checkpoint':admitted['directory_checkpoint'],
            'weight_read_events':4,'selected_checkpoint_keys':[PREFIX+k for k in weights],'full_model_constructed':False,'full_prefix_constructed':False,
            'attention_module_constructed':args.engine=='official','performance_measurement':False,'observations':observations,'raw_proof':proof,
            'legacy_context_bridges':bridges,'input_layout_scope':'ActualR38recordednorm1strideandoffsetrestored;NEWcontextlayoutsobservedhere,notretroactiveR38captures',
            'scope':'Oneblock0norm1handoff/4weights/localprojectioncuts,notpolicycombination/newbackend/kernel/default/robot/speed/generalizationqualification.'}
        if args.engine=='native':
            report.update(context_difference=_difference(raw['context'],old_context),interaction_formula='(O6-O18)-(N6-N18)',
                old_corner_is_outcome_not_hard_gate=True,official_predecessor={n:file_record(args.official_bridge/n) for n in ('result.json','operator_outputs.safetensors')})
        publish_capture(args.output,raw,report,references=refs,old_context=old_context,old_reference=old_reference,interaction_array=I)
        print(json.dumps({'engine':args.engine,'raw_tensors':len(raw),'weight_read_events':4,'old_corner_exact':proof.get('cuts',{}).get('old_corner_exact'),
            'interaction_nonzero':proof.get('interaction_stats',{}).get('nonzero_elements')}),flush=True)
    except BaseException as error:
        write_failure(args.output,error,raw=raw,observations=observations,interaction_array=I);raise
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--official-bridge',type=Path);parser.add_argument('--source-revision',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);_module(Path(__file__).with_name('diagnose.py'),'hy_bridge_step_guard').require_step()
    import platform,sys
    import numpy as np
    import torch
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact R39 source commit required')
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Actual attention bridge requires admitted A800')
    torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    return run(args,runtime,torch.version.git_version,torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)


if __name__=='__main__':raise SystemExit(main())
