"""Bounded actual ViT block0–3 observers; never a policy or kernel change."""
from __future__ import annotations

import ast
import copy
import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import MethodType,SimpleNamespace

WEIGHT_PREFIX='model.dual_tower.vlm.model.visual.vision_tower.'
FIELDS=('input','norm1_input','norm1','qkv','attention','residual','norm2','mlp','output')
METHODS={'_vit_qkv':'self, h','_vit_spatial_attn':'self, q, k, v','_vit_time_pe':'self, kf, device, dtype',
    '_vit_time_mix':'self, q, k, v, b, kf','_vit_mlp':'self, x','_vit_block':'self, x, li, num_frames',
    '_vit_pos_embed_rescale':'self, h, w, dtype','vit_forward':'self, imgs'}


def prefix_keys():
    C,F=1152,4304
    keys={'patch_embed.proj.weight':(C,3,16,16),'patch_embed.proj.bias':(C,),'pos_embed':(1,16384,C)}
    block={'norm1.weight':(C,),'norm1.bias':(C,),'norm2.weight':(C,),'norm2.bias':(C,),
        'attn.qkv.weight':(3*C,C),'attn.qkv.bias':(3*C,),'attn.proj.weight':(C,C),'attn.proj.bias':(C,),
        'mlp.fc1.weight':(F,C),'mlp.fc1.bias':(F,),'mlp.fc2.weight':(C,F),'mlp.fc2.bias':(C,)}
    keys.update({f'blocks.{i}.{k}':shape for i in range(4) for k,shape in block.items()})
    return keys


def read_prefix_weights(reader,*,device):
    import torch
    weights={}
    for name,shape in prefix_keys().items():
        value=reader.get_tensor(WEIGHT_PREFIX+name)
        if value.dtype!=torch.bfloat16 or tuple(value.shape)!=shape or not torch.isfinite(value).all():
            raise ValueError('Prefix weight must have exactly admitted finite BF16 shape: '+name)
        weights[name]=value.to(device=device)
    return weights


def _validate_weights(weights):
    import torch
    spec=prefix_keys()
    if set(weights)!=set(spec):raise ValueError('Prefix weight keys must be exactly51 selected tensors')
    device=weights['pos_embed'].device
    if any(v.dtype!=torch.bfloat16 or v.shape!=spec[k] or v.device!=device or not torch.isfinite(v).all() for k,v in weights.items()):
        raise ValueError('Prefix weight shape/device/finite BF16 contract differs')


def make_native_owner(weights,*,source):
    _validate_weights(weights)
    W=SimpleNamespace(_vit_patch_w=weights['patch_embed.proj.weight'],_vit_patch_b=weights['patch_embed.proj.bias'],_vit_pos_embed=weights['pos_embed'])
    for attr,stem in {'ln1':'norm1','ln2':'norm2','qkv':'attn.qkv','proj':'attn.proj','fc1':'mlp.fc1','fc2':'mlp.fc2'}.items():
        for suffix,field in (('w','weight'),('b','bias')):setattr(W,'_vit_'+attr+'_'+suffix,[weights[f'blocks.{i}.{stem}.{field}'] for i in range(4)])
    owner=SimpleNamespace(W=W,_fp8=False,vit_heads=16,vit_hd=72,vit_scale=72**-.5,vit_eps=1e-6,vit_time_base=100.,
        vit_spacetime_ids=set(range(3,27,4)),_prefix_source=Path(source))
    for name,method in load_native_methods(source).items():setattr(owner,name,MethodType(method,owner))
    return owner


def make_official_prefix(vision,video,weights,*,device):
    import torch
    _validate_weights(weights)
    tower=vision._create_hyvit2(select_layer=4,ckpt_path='',weight_init='skip').to(device=device,dtype=torch.bfloat16)
    if len(tower.blocks)!=4 or not isinstance(tower.patch_drop,torch.nn.Identity) or not isinstance(tower.norm_pre,torch.nn.Identity):
        raise ValueError('Official prefix constructor is not original four-block identity-preprocessing form')
    for block in tower.blocks:
        if (block.attn.q_bias is not None or not isinstance(block.attn.q_norm,torch.nn.Identity) or not isinstance(block.attn.k_norm,torch.nn.Identity)
                or (block.attn.num_heads,block.attn.head_dim)!=(16,72)
                or any(not isinstance(getattr(block,k),torch.nn.Identity) for k in ('ls1','ls2','drop_path1','drop_path2'))):
            raise ValueError('Official prefix block has unsupported qk-norm/scale/drop/head routing')
    if any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in tower.modules()):raise ValueError('Official prefix requires original zero-dropout source defaults')
    tower.blocks[3]=video.SpaceTimeBlock(tower.blocks[3],max_num_frames=18,learnable_time_embed=False,time_embed_base=100.)
    tower.past_drop_layer=None;tower.spacetime_layer_stride=4;tower.max_num_frames=18
    if set(tower.state_dict())!=set(weights):raise ValueError('Official prefix parameter keys differ from selected51 weights')
    tower.load_state_dict(weights,strict=True)
    for parameter in tower.parameters():parameter.requires_grad_(False)
    tower.eval()
    if any(not _same(value,weights[key]) for key,value in tower.state_dict().items()):raise ValueError('Official prefix selected parameter values changed during loading')
    return tower


def _layout(x):
    return {'shape':list(x.shape),'dtype':str(x.dtype),'live_stride':list(x.stride()),'storage_offset':x.storage_offset()}


def _same(a,b):
    import torch
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)


def _record(raw,metadata,key,value):
    import torch
    if key in raw or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():raise ValueError('Duplicate or invalid actual prefix raw field: '+key)
    metadata[key]=_layout(value);raw[key]=value.detach().cpu().contiguous().clone()


def load_native_methods(path,*,observer=None):
    import torch
    import torch.nn.functional as F
    tree=ast.parse(Path(path).read_text());classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HyVLAThorBF16Pipeline']
    nodes=[n for c in classes for n in c.body if isinstance(n,ast.FunctionDef) and n.name in METHODS]
    if len(classes)!=1 or len(nodes)!=8 or {n.name for n in nodes}!=set(METHODS):raise ValueError('Native prefix source requires eight unique actual methods')
    for node in nodes:
        args=ast.parse(f'def f({METHODS[node.name]}): pass').body[0].args
        decorators=['torch.no_grad()'] if node.name=='vit_forward' else []
        if ast.dump(node.args)!=ast.dump(args) or [ast.unparse(d) for d in node.decorator_list]!=decorators:
            raise ValueError('Native prefix source signature/decorator drift')
    forward=next(n for n in nodes if n.name=='vit_forward')
    loops=[n for n in ast.walk(forward) if isinstance(n,ast.For)]
    expected=ast.parse('for li in range(27):\n x = self._vit_block(x, li, K)').body[0]
    if (len(loops)!=1 or ast.dump(loops[0].target)!=ast.dump(expected.target) or ast.dump(loops[0].iter)!=ast.dump(expected.iter)
            or ast.dump(loops[0].body[0])!=ast.dump(expected.body[0])):raise ValueError('Native prefix actual loop/source drift')
    selected=copy.deepcopy(nodes);replacements={'_vit_block':('layer_norm','_prefix_layer_norm',3),
        '_vit_qkv':('linear','_prefix_linear',1),'vit_forward':('conv2d','_prefix_conv2d',1)}
    if observer is not None:
        for name,(op,replacement,count) in replacements.items():
            node=next(n for n in selected if n.name==name);calls=[n for n in ast.walk(node) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='F' and n.func.attr==op]
            if len(calls)!=count:raise ValueError('Native prefix observation AST call count drift')
            for call in calls:call.func=ast.Name(id=replacement,ctx=ast.Load())
        inverse=copy.deepcopy(selected)
        for node in inverse:
            if node.name not in replacements:continue
            op,replacement,_=replacements[node.name]
            for call in ast.walk(node):
                if isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id==replacement:
                    call.func=ast.Attribute(value=ast.Name(id='F',ctx=ast.Load()),attr=op,ctx=ast.Load())
        if ast.dump(ast.Module(body=inverse,type_ignores=[]))!=ast.dump(ast.Module(body=nodes,type_ignores=[])):
            raise ValueError('Native prefix observation inverse AST changed math')
    def observed(kind,fn):
        def run(*a,**kw):
            out=fn(*a,**kw);observer(kind,a,kw,out);return out
        return run
    scope={'torch':torch,'F':F}
    if observer is not None:scope.update(_prefix_layer_norm=observed('norm',F.layer_norm),
        _prefix_linear=observed('qkv',F.linear),_prefix_conv2d=observed('patch',F.conv2d))
    module=ast.fix_missing_locations(ast.Module(body=selected,type_ignores=[]));exec(compile(module,str(path),'exec'),scope)
    return {name:scope[name] for name in METHODS}


class _PrefixStop(Exception):
    def __init__(self,value):self.value=value


def _four_blocks(owner,images,*,on_input=None,on_output=None):
    calls=[];original=owner._vit_block;stop=None
    def block(x,li,num_frames):
        if li!=len(calls) or li not in (0,1,2,3) or num_frames!=6:raise ValueError('Unexpected actual ViT block order/frame count; block4 prohibited')
        calls.append(li)
        if on_input is not None:on_input(li,x)
        value=original(x,li,num_frames)
        if on_output is not None:on_output(li,value)
        if li==3:raise _PrefixStop(value)
        return value
    owner._vit_block=block
    try:
        try:owner.vit_forward(images)
        except _PrefixStop as signal:stop=signal
        if stop is None or calls!=[0,1,2,3]:raise ValueError('Native vit_forward did not take exact post-block3 stop')
        return stop.value,calls
    finally:owner._vit_block=original


def capture_native(owner,images,*,synchronize):
    import torch
    if getattr(owner,'_frozen_vit_prefix_active',False):raise ValueError('Nested native prefix capture is active')
    if images.dtype!=torch.bfloat16 or images.shape!=(3,6,3,224,224) or not torch.isfinite(images).all() or getattr(owner,'_fp8',None) is not False:
        raise ValueError('Native prefix requires admitted BF16 three-camera six-frame input, no quantization')
    initial=dict(owner.__dict__);raw={};meta={};state={'complete':False,'cleanup':{'passed':False},'layouts':meta,
        'normal_vit_forward_return_reached':False,'block4_called':False};primary=None
    weights=[]
    for value in vars(owner.W).values():
        if isinstance(value,torch.Tensor):weights.append(value)
        elif isinstance(value,list):weights.extend(v for v in value if isinstance(v,torch.Tensor))
    old_weights=[v.clone() for v in weights];before=images.clone();hold={'block':None,'norm':0}
    def record(key,value):_record(raw,meta,key,value)
    def event(kind,a,kw,out):
        li=hold['block']
        if kind=='patch':
            if li is not None:raise ValueError('Unexpected native patch call inside block')
            record('patch',out.flatten(2).transpose(1,2));state['actual_conv_layout']=_layout(out);return
        if li not in (0,1,2,3):raise ValueError('Native prefix observer escaped block scope')
        if kind=='norm':
            index=hold['norm'];hold['norm']+=1
            if index not in (0,1):raise ValueError('Native block has unexpected normalization count')
            record(f'block{li}.'+('norm1_input' if index==0 else 'residual'),a[0])
            record(f'block{li}.'+('norm1' if index==0 else 'norm2'),out)
        elif kind=='qkv':record(f'block{li}.qkv',out)
    def method_wrapper(name,label):
        original=getattr(owner,name)
        def run(*a,**kw):
            out=original(*a,**kw)
            if label=='position':record(label,out)
            else:
                li=hold['block']
                if li not in (0,1,2,3) or (label in ('time_pe','temporal_v') and li!=3):raise ValueError('Native prefix method observer phase drift')
                record(f'block{li}.'+label,out)
            return out
        setattr(owner,name,run)
    try:
        owner._frozen_vit_prefix_active=True
        control,control_calls=_four_blocks(owner,images)
        record('control.block3.output',control)
        # Discard only temporary owner attributes created by its own actual
        # block calls; preserve original method identity and parameter objects.
        owner.__dict__.clear();owner.__dict__.update(initial);owner._frozen_vit_prefix_active=True
        for name,method in load_native_methods(owner._prefix_source,observer=event).items():setattr(owner,name,MethodType(method,owner))
        for name,label in (('_vit_pos_embed_rescale','position'),('_vit_time_pe','time_pe'),('_vit_time_mix','temporal_v'),
                ('_vit_spatial_attn','attention'),('_vit_mlp','mlp')):method_wrapper(name,label)
        def start(li,x):hold.update(block=li,norm=0);record(f'block{li}.input',x)
        def finish(li,x):
            if hold['norm']!=2:raise ValueError('Native prefix normalization phase incomplete')
            record(f'block{li}.output',x);hold['block']=None
        observed,observed_calls=_four_blocks(owner,images,on_input=start,on_output=finish)
        record('prepared.images',images)
        if not _same(control,observed) or not _same(images,before) or any(not _same(v,b) for v,b in zip(weights,old_weights)):
            raise ValueError('Native prefix actual control or readonly inputs/weights differ')
        if len(raw)!=42:raise ValueError('Native prefix actual raw coverage incomplete')
        state.update(complete=True,control_block_indices=control_calls,observed_block_indices=observed_calls)
    except BaseException as error:
        primary=error;error.prefix_partial=(raw,state);raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        try:owner.__dict__.clear();owner.__dict__.update(initial)
        except BaseException as error:errors.append('owner restore failed: '+str(error))
        state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False
            if primary is not None:primary.add_note('; '.join(errors))
            else:
                error=RuntimeError('Native prefix cleanup failed: '+'; '.join(errors));error.prefix_partial=(raw,state);raise error
    return raw,state


def observe_official(tower,video_function,images,*,synchronize):
    import torch
    if getattr(tower,'_frozen_vit_prefix_active',False):raise ValueError('Nested official prefix observer is active')
    if (images.shape!=(3,6,3,224,224) or images.dtype!=torch.bfloat16 or not torch.isfinite(images).all()
            or len(tower.blocks)!=4 or any(m.training for m in tower.modules())):
        raise ValueError('Official prefix requires eval four-block module and admitted BF16 images')
    raw={};layouts={};state={'complete':False,'cleanup':{'passed':False},'layouts':layouts,'camera_block_indices':[]}
    hold={'camera':None,'block':None,'indices':[]};handles=[];methods=[];missing=object();primary=None
    active=tower.__dict__.get('_frozen_vit_prefix_active',missing);before=images.clone()
    weights=list(tower.parameters());old_weights=[v.clone() for v in weights]
    def record(key,value):
        camera=hold['camera']
        if camera not in (0,1,2):raise ValueError('Official actual stage escaped camera scope')
        _record(raw,layouts,f'camera{camera}.'+key,value)
    def stage(li,key):
        def post(module,a,out):
            if hold['block']!=li:raise ValueError('Official actual stage escaped block scope')
            record(f'block{li}.'+key,out)
        return post
    def stage_input(li,key):
        def pre(module,a):
            if hold['block']!=li or len(a)!=1:raise ValueError('Official actual input stage drift')
            record(f'block{li}.'+key,a[0])
        return pre
    def block_start(li):
        def pre(module,a,kw):
            if hold['block'] is not None or li!=len(hold['indices']) or len(a)!=1:raise ValueError('Official actual block order/reentry differs')
            if (li==3 and (set(kw)!={'cu_slens','num_frames'} or kw['cu_slens'] is not None or kw['num_frames']!=6)) or (li!=3 and kw):
                raise ValueError('Official actual block3 video frame parameters differ')
            hold['block']=li;hold['indices'].append(li);record(f'block{li}.input',a[0])
        return pre
    def block_finish(li):
        def post(module,a,out):
            if hold['block']!=li:raise ValueError('Official actual block return order differs')
            record(f'block{li}.output',out);hold['block']=None
        return post
    def wrap(owner,name,label):
        original=getattr(owner,name);old=owner.__dict__.get(name,missing);methods.append((owner,name,old))
        def run(*a,**kw):
            out=original(*a,**kw)
            if label=='position':
                if hold['block'] is not None:raise ValueError('Official positional embedding inside block')
            elif hold['block']!=3:raise ValueError('Official temporal stage outside block3')
            record(label if label=='position' else 'block3.'+label,out);return out
        setattr(owner,name,run)
    try:
        tower._frozen_vit_prefix_active=True
        handles.append(tower.patch_embed.register_forward_hook(lambda m,a,out:record('patch',out)))
        wrap(tower,'rescale_positional_embedding','position')
        wrap(tower.blocks[3],'_build_time_pe','time_pe');wrap(tower.blocks[3],'_time_softmax_on_v','temporal_v')
        for li,block in enumerate(tower.blocks):
            handles.append(block.register_forward_pre_hook(block_start(li),with_kwargs=True))
            handles.append(block.register_forward_hook(block_finish(li)))
            for module,key in ((block.norm1,'norm1'),(block.norm2,'norm2'),(block.attn.qkv,'qkv'),(block.attn.proj,'attention'),(block.mlp,'mlp')):
                handles.append(module.register_forward_hook(stage(li,key)))
            handles.append(block.norm1.register_forward_pre_hook(stage_input(li,'norm1_input')))
            handles.append(block.norm2.register_forward_pre_hook(stage_input(li,'residual')))
        _record(raw,layouts,'prepared.images',images)
        for camera in range(3):
            hold.update(camera=camera,block=None,indices=[])
            result=video_function(tower,images[camera:camera+1])
            if hold['block'] is not None or hold['indices']!=[0,1,2,3]:raise ValueError('Official prefix actual block coverage incomplete')
            if not isinstance(result,tuple) or len(result)!=2 or tuple(result[1])!=(14,14):raise ValueError('Official video API result geometry differs')
            record('api_current_frame',result[0]);state['camera_block_indices'].append(list(hold['indices']))
            endpoint=raw[f'camera{camera}.block3.output'].view(1,6,196,-1)[:,-1]
            if not _same(endpoint,raw[f'camera{camera}.api_current_frame']):raise ValueError('Official actual block3/current-frame API bridge differs')
        if len(raw)!=124:raise ValueError('Official prefix actual stage coverage incomplete')
        if not _same(images,before) or any(not _same(v,b) for v,b in zip(weights,old_weights)):raise ValueError('Official prefix readonly input/weight mutation')
        state['complete']=True
    except BaseException as error:
        primary=error;error.prefix_partial=(raw,state);raise
    finally:
        errors=[]
        try:synchronize()
        except BaseException as error:errors.append('sync failed: '+str(error))
        for handle in reversed(handles):
            try:handle.remove()
            except BaseException as error:errors.append('hook restore failed: '+str(error))
        for owner,name,value in reversed(methods):
            try:
                if value is missing:owner.__dict__.pop(name,None)
                else:setattr(owner,name,value)
            except BaseException as error:errors.append('method restore failed: '+str(error))
        if active is missing:tower.__dict__.pop('_frozen_vit_prefix_active',None)
        else:tower._frozen_vit_prefix_active=active
        state['cleanup']={'passed':not errors,'errors':errors}
        if errors:
            state['complete']=False
            if primary is not None:primary.add_note('; '.join(errors))
            else:
                error=RuntimeError('Official prefix cleanup failed: '+'; '.join(errors));error.prefix_partial=(raw,state);raise error
    return raw,state


def _raw_shapes(engine):
    C=1152;spec={'prepared.images':(3,6,3,224,224)}
    if engine not in ('official','native'):raise ValueError('Unknown ViT prefix engine')
    for camera in range(3 if engine=='official' else 1):
        base=f'camera{camera}.' if engine=='official' else '';B=1 if engine=='official' else 3
        spec[base+'patch']=(B*6,196,C);spec[base+'position']=(1,196,C)
        spec[base+'block3.time_pe']=(6,C);spec[base+'block3.temporal_v']=(B*6,16,196,72)
        for li in range(4):
            for field in FIELDS:
                shape=(B,6,196,C) if li==3 and field in ('norm1_input','norm1') else (B*6,196,3*C if field=='qkv' else C)
                spec[base+f'block{li}.'+field]=shape
        if engine=='official':spec[base+'api_current_frame']=(1,196,C)
    if engine=='native':spec['control.block3.output']=(18,196,C)
    return spec


def validate_raw(raw,*,engine):
    import torch
    spec=_raw_shapes(engine)
    if set(raw)!=set(spec) or any(v.device.type!='cpu' or v.dtype!=torch.bfloat16 or tuple(v.shape)!=spec[k] or not torch.isfinite(v).all() for k,v in raw.items()):
        raise ValueError('ViT prefix raw role/shape/dtype/finite contract differs')
    for camera in range(3 if engine=='official' else 1):
        base=f'camera{camera}.' if engine=='official' else '';B=1 if engine=='official' else 3
        if not _same(raw[base+'patch']+raw[base+'position'],raw[base+'block0.input']):raise ValueError('ViT prefix patch/position chain differs')
        for li in range(4):
            stem=base+f'block{li}.';x=raw[stem+'input']
            if li and not _same(x,raw[base+f'block{li-1}.output']):raise ValueError('ViT prefix actual block chain differs')
            norm_input=x if li!=3 else x.view(B,6,196,1152)+raw[base+'block3.time_pe'].view(1,6,1,1152)
            if not _same(norm_input,raw[stem+'norm1_input']):raise ValueError('ViT prefix norm1 input/actual timePE chain differs')
            if not _same(x+raw[stem+'attention'],raw[stem+'residual']) or not _same(raw[stem+'residual']+raw[stem+'mlp'],raw[stem+'output']):
                raise ValueError('ViT prefix actual residual chain differs')
        if engine=='official' and not _same(raw[base+'api_current_frame'],raw[base+'block3.output'].view(1,6,196,1152)[:,-1]):
            raise ValueError('ViT prefix current-frame API bridge differs')
    if engine=='native' and not _same(raw['control.block3.output'],raw['block3.output']):raise ValueError('Native prefix observed/control endpoint differs')
    return {'tensors':len(raw),'block_chain_exact':True,'residual_chains_exact':True,'actual_endpoint_bridge_exact':True}


def admit_official_capture(raw,gold,*,factory):
    proof=validate_raw(raw,engine='official')
    if not _same(raw['prepared.images'],gold['images']):raise ValueError('Actual ViT input differs from admitted Gold images')
    for camera in range(3):
        if (not _same(raw[f'camera{camera}.patch'],gold['patches'][camera]) or not _same(raw[f'camera{camera}.position'],gold['position'])
                or not _same(raw[f'camera{camera}.block3.output'],gold['block3'][camera])):
            raise ValueError('Actual prefix Patch/position/block3 Gold anchor differs')
    return factory()


def _metric(a,b):
    import torch
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Prefix comparison counterpart shape/dtype differs')
    x,y=a.detach().cpu().double(),b.detach().cpu().double();d=x-y;den=float(x.norm())*float(y.norm())
    return {'exact':_same(a,b),'unequal_elements':int(torch.count_nonzero(a!=b)),'max_abs':float(d.abs().max()),
        'rmse':float(d.square().mean().sqrt()),'relative_l2':float(d.norm()/max(float(y.norm()),1e-12)),
        'cosine':float((x*y).sum())/den if den else float(_same(a,b))}


def compare_prefix(official,native):
    """Compare already-admitted captures; camera/frame assembly is CPU analysis."""
    import torch
    if set(official)!=set(_raw_shapes('official')) or set(native)!=set(_raw_shapes('native')):raise ValueError('Prefix comparison raw roles incomplete')
    stages={}
    def compare(key,a,b,*,frames=True):
        row=_metric(a,b);aa=a.reshape(3,6,-1) if frames else a.reshape(3,-1);bb=b.reshape(aa.shape)
        row['per_camera']=[]
        for i in range(3):
            camera=_metric(aa[i],bb[i]);camera['per_frame']=[_metric(aa[i,j],bb[i,j]) for j in range(6)] if frames else []
            row['per_camera'].append(camera)
        stages[key]=row
    compare('prepared.images',official['prepared.images'],native['prepared.images'])
    compare('patch',torch.cat([official[f'camera{i}.patch'] for i in range(3)]),native['patch'])
    compare('position',torch.cat([official[f'camera{i}.position'] for i in range(3)]),native['position'].expand(3,-1,-1),frames=False)
    for li in range(4):
        for field in FIELDS:
            if li==3 and field=='norm1_input':
                compare('block3.time_pe',torch.stack([official[f'camera{i}.block3.time_pe'] for i in range(3)]),native['block3.time_pe'][None].expand(3,-1,-1))
            if li==3 and field=='attention':
                compare('block3.temporal_v',torch.cat([official[f'camera{i}.block3.temporal_v'] for i in range(3)]),native['block3.temporal_v'])
            name=f'block{li}.'+field
            compare(name,torch.cat([official[f'camera{i}.'+name] for i in range(3)]),native[name])
    first=next((name for name,row in stages.items() if not row['exact']),None)
    return {'first_different_stage':first,'first_different_block':int(first[5]) if first is not None and first.startswith('block') else None,
        'stages':stages,'comparison_representation':'camera-major/frame-major CPU concatenation; native position/timePE reused only for comparison',
        'scope':'Actualprefixboundarylocalization,notisolatedruntimecauseorpropagatederrorfraction;physicalschemaadmissionseparate.'}


def _sibling(name):
    spec=importlib.util.spec_from_file_location('hy_vit_prefix_'+name,Path(__file__).with_name(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def file_record(path):
    p=Path(path).resolve();stat=p.stat();return {'path':str(p),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}


def admit_before_weights(manifest,*,native_source,factory):
    if manifest.get('kind')!='hy_frozen_actual_vit_prefix_inputs':raise ValueError('Actual ViT input manifest kind differs')
    task=Path(manifest['task_root']).resolve();record=manifest['history_manifest'];expected=task/'results/hy-merger-stages-317360/inputs.json'
    if Path(record['path']).resolve()!=expected or file_record(expected)!=record:raise ValueError('Actual ViT historical manifest path/stat differs')
    history=json.loads(expected.read_text());_sibling('merger_stages').admit_file_records(history,factory=lambda:None)
    gold=json.loads(Path(history['records']['gold.baseline.report']['path']).read_text())
    package=Path(gold['hy_vla_source']).resolve();root=task/'snapshots/round007/diagnostic/official/hy_vla'
    if package!=root/'__init__.py' or not package.is_file() or Path(history['official_source']['path']).resolve()!=root/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py':
        raise ValueError('Actual ViT official source must belong to accepted Gold round007')
    policy=root/'modeling_hy_vla.py'
    paths={'official_policy':policy,'official_modeling':Path(history['official_source']['path']).resolve(),
        'official_video':root/'space_time_attention.py','official_configuration':root/'configuration_hy_vla.py',
        'native':task/'snapshots/round038/code/flash_rt/models/hyvla/pipeline_thor.py','config':Path(history['checkpoint']['path']).parent/'config.json'}
    if Path(native_source).resolve()!=paths['native'] or set(manifest.get('source_files',{}))!=set(paths):raise ValueError('Actual ViT new source/input role paths differ')
    for role,path in paths.items():
        rec=manifest['source_files'][role]
        if Path(rec['path']).resolve()!=path or file_record(path)!=rec:raise ValueError('Actual ViT source/config path/stat differs: '+role)
    return factory(history)


def admit_prior_official(report,raw,gold,*,job_id,source_revision,runtime,torch_git,checkpoint,input_manifest,source_records,factory):
    state=report.get('observations',{})
    if (report.get('kind')!='hy_actual_frozen_vit_prefix' or report.get('engine')!='official' or report.get('status')!='prefix_capture_complete'
            or report.get('job_id')!=job_id or report.get('source_revision')!=source_revision or report.get('runtime')!=runtime
            or report.get('torch_git_version')!=torch_git or report.get('checkpoint')!=checkpoint or report.get('input_manifest')!=input_manifest
            or report.get('source_records')!=source_records or report.get('weight_read_events')!=51
            or len(report.get('selected_checkpoint_keys',[]))!=51 or set(report['selected_checkpoint_keys'])!={WEIGHT_PREFIX+k for k in prefix_keys()}
            or report.get('full_model_constructed') is not False or report.get('blocks_constructed')!=4
            or state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True
            or state.get('camera_block_indices')!=[[0,1,2,3]]*3):raise ValueError('Prior official prefix header/source/runtime/cleanup differs')
    if report.get('raw_proof')!=validate_raw(raw,engine='official'):raise ValueError('Prior official raw proof differs')
    return admit_official_capture(raw,gold,factory=factory)


def admit_gold_inputs(samples):
    import torch
    if set(samples)!={'baseline','state'}:raise ValueError('Gold shared-context roles differ')
    for sample in samples.values():
        if (len(sample.get('patches',()))!=3 or len(sample.get('block3',()))!=3
                or sample['images'].shape!=(3,6,3,224,224) or sample['position'].shape!=(1,196,1152)
                or any(v.shape!=(6,196,1152) for v in (*sample['patches'],*sample['block3']))):raise ValueError('Gold images/Patch/block3 role geometry differs')
        if any(v.dtype!=torch.bfloat16 or not torch.isfinite(v).all() for v in (sample['images'],sample['position'],*sample['patches'],*sample['block3'])):
            raise ValueError('Gold prefix inputs require finite actual BF16 captures')
        meta=sample['image_metadata']
        if meta.get('shape')!=[3,6,3,224,224] or meta.get('dtype')!='torch.bfloat16' or meta.get('live_stride')!=[903168,150528,50176,224,1]:
            raise ValueError('Gold actual prepared-image layout metadata differs')
    a,b=samples['baseline'],samples['state']
    if (not _same(a['images'],b['images']) or not _same(a['position'],b['position'])
            or any(not _same(x,y) for x,y in zip((*a['patches'],*a['block3']),(*b['patches'],*b['block3'])))):
        raise ValueError('Gold baseline/state are not one shared visual context')
    return a


def load_inputs(manifest,*,engine,runtime,torch_git):
    from safetensors import safe_open
    native=Path(__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py';merger=_sibling('merger_stages')
    historical=admit_before_weights(manifest,native_source=native,factory=lambda old:old)
    history,proof=merger.load_history(historical,engine=engine,runtime=runtime,torch_git=torch_git)
    samples={}
    for case in ('baseline','state'):
        rec=historical['records'];doc=json.loads(Path(rec['gold.'+case+'.report']['path']).read_text())
        with safe_open(rec['gold.'+case+'.raw']['path'],framework='pt',device='cpu') as sf:
            samples[case]={'images':sf.get_tensor('prepared.images'),'position':sf.get_tensor('vision.position_embedding'),
                'patches':tuple(sf.get_tensor(f'vision.patch_embed.{i}') for i in range(3)),
                'block3':tuple(sf.get_tensor(f'vision.block3.{i}') for i in range(3)),'image_metadata':doc['tensor_metadata']['prepared.images']}
    gold=admit_gold_inputs(samples)
    checkpoint=historical['checkpoint'];checkpoint_dir=Path(checkpoint['path']).parent
    validator=_sibling('native_policy_tree').validator;_,directory=validator.checkpoint_record(checkpoint_dir)
    old_native=json.loads(Path(historical['records']['native.baseline.report']['path']).read_text())
    if directory!=old_native['checkpoint']:raise ValueError('Actual prefix checkpoint/config directory identity changed')
    cfg=json.loads(Path(manifest['source_files']['config']['path']).read_text())
    if (cfg.get('use_video_encoder') is not True or cfg.get('spacetime_layer_stride',4)!=4 or cfg.get('past_drop_layer') is not None
            or cfg.get('max_num_frames',18)!=18 or cfg.get('learnable_time_embed',False) is not False or cfg.get('time_embed_base',100.)!=100.):
        raise ValueError('Actual prefix video configuration differs from fixed temporal source path')
    if file_record(checkpoint_dir/'config.json')!=manifest['source_files']['config']:raise ValueError('Actual prefix config file stat changed during admission')
    return {'gold':gold,'checkpoint':checkpoint,'directory_checkpoint':directory,'native_source':native,'history_manifest':historical,
        'history_proof':proof,'image_layout':{'live_stride':gold['image_metadata']['live_stride'],'storage_offset_assumption':0,'historical_offset_observed':False}}


def _source_records(manifest):
    return {'driver':file_record(__file__),**manifest['source_files']}


def publish_capture(output,raw,report):
    from safetensors.torch import save_file
    state=report.get('observations',{})
    if state.get('complete') is not True or state.get('cleanup',{}).get('passed') is not True:raise ValueError('Actual prefix capture/cleanup is not complete')
    proof=validate_raw(raw,engine=report['engine'])
    if report.get('raw_proof',proof)!=proof:raise ValueError('Actual prefix raw proof changed before publication')
    save_file({k:v.contiguous() for k,v in raw.items()},str(Path(output)/'operator_outputs.safetensors'))
    temp=Path(output)/'result.json.tmp';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(Path(output)/'result.json')


def _official_modules(manifest):
    import importlib,sys
    sources=manifest['source_files'];root=Path(sources['official_policy']['path']).parent.parent
    before=list(sys.path)
    try:
        sys.path.insert(0,str(root))
        vision=importlib.import_module('hy_vla.hunyuan_vl_mot.modeling_hunyuan_vl_mot')
        video=importlib.import_module('hy_vla.space_time_attention')
    finally:sys.path[:]=before
    for module,role in ((vision,'official_modeling'),(video,'official_video')):
        if file_record(module.__file__)!=sources[role]:raise ValueError('Imported official prefix module belongs to another source tree')
    return vision,video


def qualified_capture(owner,images,*,engine,video_function,gold,official_raw,synchronize):
    raw={};observations={}
    try:
        if engine=='official':
            raw,observations=observe_official(owner,video_function,images,synchronize=synchronize)
            admit_official_capture(raw,gold,factory=lambda:None);comparison=None
        elif engine=='native':
            raw,observations=capture_native(owner,images,synchronize=synchronize)
            validate_raw(raw,engine='native');comparison=compare_prefix(official_raw,raw)
        else:raise ValueError('Unknown actual prefix capture engine')
        return raw,observations,comparison
    except BaseException as error:
        # An observer can finish successfully and only THEN fail the external
        # Gold anchor, typed schema or comparison. Retain that new evidence too.
        if raw:error.prefix_partial=(raw,observations)
        raise


def write_failure(output,error,*,raw=None,observations=None):
    from safetensors.torch import save_file
    partial=getattr(error,'prefix_partial',None)
    if partial is not None:raw,observations=partial
    raw=raw or {};observations=observations or {};failures=[]
    if raw:
        try:save_file({k:v.contiguous() for k,v in raw.items()},str(Path(output)/'unqualified_partial.safetensors'))
        except BaseException as secondary:
            failures.append('raw evidence write failed: '+str(secondary));error.add_note(failures[-1])
    report={'status':'unqualified_frozen_vit_prefix_failure','qualified':False,'error_type':type(error).__name__,'error':str(error),
        'observations':observations,'captured_raw_roles':list(raw),'evidence_write_errors':failures}
    try:(Path(output)/'failure.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    except BaseException as secondary:error.add_note('failure metadata write failed: '+str(secondary))


def run(args,runtime,torch_git):
    import sys
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    manifest=json.loads(args.inputs.read_text());task=Path(manifest['task_root']).resolve();root=task/'results'/('hy-vit-prefix-'+os.environ['SLURM_JOB_ID'])
    if (args.inputs.resolve()!=root/'inputs.json' or args.output.resolve()!=root/args.engine
            or (root/'SOURCE_GIT_COMMIT').read_text().strip()!=args.source_revision):raise ValueError('Actual prefix current source/job/input/output identity differs')
    admitted=load_inputs(manifest,engine=args.engine,runtime=runtime,torch_git=torch_git)
    reference=admitted['history_proof']['runtime_reference'];flag=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    if flag!=reference['bf16_reduced_precision_reduction']:raise ValueError('Actual prefix BF16 reduction flag differs from admitted runtime')
    sources=_source_records(manifest);gold=admitted['gold'];prior=None;old_raw=None
    if args.engine=='native':
        if args.official_prefix is None or args.official_prefix.resolve()!=root/'official':raise ValueError('Native prefix needs same-job actual official prefix predecessor')
        prior=json.loads((args.official_prefix/'result.json').read_text());old_raw=load_file(str(args.official_prefix/'operator_outputs.safetensors'))
        old_runtime=json.loads(Path(admitted['history_manifest']['records']['runtime.official']['path']).read_text())
        admit_prior_official(prior,old_raw,gold,job_id=os.environ['SLURM_JOB_ID'],source_revision=args.source_revision,runtime=old_runtime['runtime'],
            torch_git=old_runtime['torch_git_version'],checkpoint=admitted['checkpoint'],input_manifest=manifest,source_records=sources,factory=lambda:None)
        if any(n=='hy_vla' or n.startswith('hy_vla.') for n in sys.modules):raise ValueError('Native runtime must not import official FA2/Hy package')
    args.output.mkdir(parents=True,exist_ok=False);raw={};observations={}
    try:
        vision,video=_official_modules(manifest) if args.engine=='official' else (None,None)
        with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
            with safe_open(admitted['checkpoint']['path'],framework='pt',device='cpu') as sf:weights=read_prefix_weights(sf,device='cuda')
            # Both historical runtimes observed this contiguous live image
            # stride. Zero offset is an explicitly labelled reconstruction.
            images=gold['images'].to('cuda')
            if list(images.stride())!=admitted['image_layout']['live_stride'] or images.storage_offset()!=0:raise ValueError('Actual frozen image layout reconstruction differs')
            if args.engine=='official':
                owner=make_official_prefix(vision,video,weights,device='cuda')
            else:
                owner=make_native_owner(weights,source=admitted['native_source'])
            raw,observations,comparison=qualified_capture(owner,images,engine=args.engine,
                video_function=video._patched_forward_video_features if video is not None else None,gold=gold,official_raw=old_raw,synchronize=torch.cuda.synchronize)
        proof=validate_raw(raw,engine=args.engine)
        report={'kind':'hy_actual_frozen_vit_prefix','status':'prefix_capture_complete','engine':args.engine,'job_id':os.environ['SLURM_JOB_ID'],
            'source_revision':args.source_revision,'runtime':runtime,'torch_git_version':torch_git,'bf16_reduced_precision_reduction':flag,
            'input_manifest':manifest,'checkpoint':admitted['checkpoint'],'directory_checkpoint':admitted['directory_checkpoint'],
            'history_proof':admitted['history_proof'],'source_records':sources,'weight_read_events':51,'selected_checkpoint_keys':[WEIGHT_PREFIX+k for k in weights],
            'full_model_constructed':False,'vision_prefix_module_constructed':args.engine=='official','blocks_constructed':4,'performance_measurement':False,
            'image_layout':admitted['image_layout'],'observations':observations,'raw_proof':proof,'comparison':comparison,
            'official_imports':{n:file_record(m.__file__) for n,m in (('modeling',vision),('video',video))} if args.engine=='official' else {},
            'actual_endpoint_scope':'Official4-blockvideoAPIand3Goldblock3returns' if args.engine=='official' else 'Actualnativeblock3returnafteroriginalvit_forwardprefixSTOP,NOTfull27-blockAPIreturn',
            'scope':'Onefrozenvisualcontext/4blocks;runtimeANDcallformconfounded,nopolicyerrorfraction/robot/speed/defaultclaim. Noautomaticblock4–26extension.'}
        if args.engine=='native':report['official_predecessor']={n:file_record(args.official_prefix/n) for n in ('result.json','operator_outputs.safetensors')}
        publish_capture(args.output,raw,report)
        print(json.dumps({'engine':args.engine,'raw_proof':proof,'first_different_stage':comparison['first_different_stage'] if comparison is not None else None,
            'full_model_constructed':False}),flush=True)
    except BaseException as error:
        write_failure(args.output,error,raw=raw,observations=observations)
        raise
    return 0


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--engine',choices=('official','native'),required=True);parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--official-prefix',type=Path);parser.add_argument('--source-revision',required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);_sibling('diagnose').require_step()
    import platform,sys
    import numpy as np
    import torch
    if len(args.source_revision)!=40 or any(c not in '0123456789abcdef' for c in args.source_revision):raise ValueError('Exact R38 source revision required')
    if tuple(torch.cuda.get_device_capability())!=(8,0) or 'A800' not in torch.cuda.get_device_name():raise RuntimeError('Actual ViT prefix requires admitted A800')
    torch.set_num_threads(2);torch.manual_seed(17);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    runtime={'python':platform.python_version(),'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}
    return run(args,runtime,torch.version.git_version)


if __name__=='__main__':raise SystemExit(main())
