"""Bounded actual ViT block0–3 observers; never a policy or kernel change."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import MethodType

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
            else:raise RuntimeError('Native prefix cleanup failed: '+'; '.join(errors))
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
