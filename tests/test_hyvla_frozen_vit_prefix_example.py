import importlib.util
from pathlib import Path
from types import MethodType,SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_prefix.py'
NATIVE=ROOT/'flash_rt/models/hyvla/pipeline_thor.py'


@pytest.fixture
def prefix():
    s=importlib.util.spec_from_file_location('hy_frozen_vit_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def tensor_schema():
    c,ff=1152,4304
    shapes={'patch_embed.proj.weight':(c,3,16,16),'patch_embed.proj.bias':(c,),'pos_embed':(1,16384,c)}
    one={'norm1.weight':(c,),'norm1.bias':(c,),'norm2.weight':(c,),'norm2.bias':(c,),
        'attn.qkv.weight':(3*c,c),'attn.qkv.bias':(3*c,),'attn.proj.weight':(c,c),'attn.proj.bias':(c,),
        'mlp.fc1.weight':(ff,c),'mlp.fc1.bias':(ff,),'mlp.fc2.weight':(c,ff),'mlp.fc2.bias':(c,)}
    shapes.update({f'blocks.{i}.{k}':v for i in range(4) for k,v in one.items()})
    return shapes


@pytest.mark.parametrize('bad',['none','dtype','shape','finite'])
def test_weight_reader_reads_only_the_51_prefix_tensors_and_rejects_other_precision(prefix,bad):
    shapes=tensor_schema();seen=[]
    class Reader:
        def get_tensor(self,key):
            stem='model.dual_tower.vlm.model.visual.vision_tower.'
            assert key.startswith(stem) and key[len(stem):] in shapes
            name=key[len(stem):];seen.append(key)
            value=torch.zeros((),dtype=torch.bfloat16).expand(shapes[name])
            if name=='pos_embed':
                if bad=='dtype':value=value.float()
                if bad=='shape':value=torch.zeros(1,4,1152,dtype=torch.bfloat16)
                if bad=='finite':value=torch.full((),float('nan'),dtype=torch.bfloat16).expand(shapes[name])
            return value
    if bad=='none':
        values=prefix.read_prefix_weights(Reader(),device='cpu')
        assert len(seen)==len(set(seen))==len(values)==51
        assert sum(v.numel()*v.element_size() for v in values.values())==161436544
        assert all('blocks.4.' not in k for k in seen)
        assert values['blocks.0.mlp.fc1.weight'].shape==(4304,1152)
    else:
        with pytest.raises(ValueError,match='weight|BF16|shape|finite'):prefix.read_prefix_weights(Reader(),device='cpu')


def tiny_native(prefix):
    c=2;w=SimpleNamespace(_vit_patch_w=torch.full((c,3,16,16),1/768,dtype=torch.bfloat16),
        _vit_patch_b=torch.zeros(c,dtype=torch.bfloat16),_vit_pos_embed=torch.zeros(1,4,c,dtype=torch.bfloat16))
    for name,shape in {'ln1_w':(c,),'ln1_b':(c,),'ln2_w':(c,),'ln2_b':(c,),
            'qkv_w':(3*c,c),'qkv_b':(3*c,),'proj_w':(c,c),'proj_b':(c,),
            'fc1_w':(4,c),'fc1_b':(4,),'fc2_w':(c,4),'fc2_b':(c,)}.items():
        setattr(w,'_vit_'+name,[torch.full(shape,.125 if name.endswith('_w') else 0.,dtype=torch.bfloat16) for _ in range(4)])
    owner=SimpleNamespace(W=w,_fp8=False,vit_heads=1,vit_hd=2,vit_scale=2**-.5,vit_eps=1e-6,vit_time_base=100.,
        vit_spacetime_ids={3,7,11,15,19,23},_prefix_source=NATIVE)
    for name,method in prefix.load_native_methods(NATIVE).items():setattr(owner,name,MethodType(method,owner))
    images=torch.arange(18,dtype=torch.bfloat16).reshape(3,6,1,1,1).expand(3,6,3,224,224).clone()/16
    return owner,images


def test_actual_native_vit_forward_stops_after_four_real_blocks_and_records_live_stages(prefix):
    owner,images=tiny_native(prefix);before=images.clone();old=dict(owner.__dict__)
    raw,state=prefix.capture_native(owner,images,synchronize=lambda:None)
    assert len(raw)==42 and state['complete'] and state['cleanup']['passed']
    assert state['control_block_indices']==state['observed_block_indices']==[0,1,2,3]
    assert state['normal_vit_forward_return_reached'] is False and state['block4_called'] is False
    assert torch.equal(raw['control.block3.output'],raw['block3.output'])
    assert raw['block0.input'].shape==(18,196,2) and raw['block3.norm1_input'].shape==(3,6,196,2)
    assert raw['block3.norm1'].shape==(3,6,196,2) and raw['block0.qkv'].shape==(18,196,6)
    assert raw['block3.temporal_v'].shape==(18,1,196,2) and raw['block3.time_pe'].shape==(6,2)
    for i in range(1,4):assert torch.equal(raw[f'block{i}.input'],raw[f'block{i-1}.output'])
    assert torch.equal(images,before) and set(owner.__dict__)==set(old) and all(owner.__dict__[k] is v for k,v in old.items())


@pytest.mark.parametrize('bad',['signature','loop','duplicate'])
def test_changed_native_signature_loop_or_duplicate_selected_method_rejects(prefix,tmp_path,bad):
    source=NATIVE.read_text()
    if bad=='signature':source=source.replace('def _vit_block(self, x, li, num_frames):','def _vit_block(self, x, li):')
    if bad=='loop':source=source.replace('for li in range(27):','for li in range(4):')
    if bad=='duplicate':source=source.replace('    def _vit_mlp(self, x):','    def _vit_mlp(self, x):\n        return x\n\n    def _vit_mlp(self, x):')
    path=tmp_path/'native.py';path.write_text(source)
    with pytest.raises(ValueError,match='source|signature|AST|method|loop'):prefix.load_native_methods(path)


@pytest.mark.parametrize('failure',['body','sync','nested'])
def test_native_capture_restores_owner_on_body_sync_or_reentry_failure(prefix,failure):
    owner,images=tiny_native(prefix);original=owner._vit_block
    def block(x,li,frames):
        if li==1:
            if failure=='body':raise RuntimeError('block body failed')
            if failure=='nested':prefix.capture_native(owner,images,synchronize=lambda:None)
        return original(x,li,frames)
    owner._vit_block=block;old=dict(owner.__dict__)
    def sync():
        if failure=='sync':raise RuntimeError('sync failed')
    with pytest.raises((ValueError,RuntimeError),match='body|sync|[Nn]ested|active'):
        prefix.capture_native(owner,images,synchronize=sync)
    assert set(owner.__dict__)==set(old) and all(owner.__dict__[k] is v for k,v in old.items())


def tiny_official():
    import torch.nn as nn
    class Patch(nn.Module):
        patch_size=(16,16)
        def forward(self,x):return x[:,0,::16,::16].reshape(6,196,1).expand(6,196,2).contiguous()
    class MLP(nn.Module):
        def __init__(self):super().__init__();self.fc1=nn.Linear(2,4);self.fc2=nn.Linear(4,2)
        def forward(self,x):return self.fc2(torch.nn.functional.gelu(self.fc1(x)))
    class Block(nn.Module):
        def __init__(self):
            super().__init__();self.norm1=nn.LayerNorm(2);self.norm2=nn.LayerNorm(2);self.mlp=MLP()
            self.attn=nn.Module();self.attn.qkv=nn.Linear(2,6);self.attn.proj=nn.Linear(2,2)
        def attend(self,h):
            q,k,v=self.attn.qkv(h).reshape(6,196,3,1,2).permute(2,0,3,1,4).unbind(0)
            if isinstance(self,SpaceTimeBlock):v=self._time_softmax_on_v(q,k,v,1,6)
            return self.attn.proj(v.transpose(1,2).reshape(6,196,2))
        def forward(self,x):
            x=x+self.attend(self.norm1(x));return x+self.mlp(self.norm2(x))
    class SpaceTimeBlock(Block):
        def _build_time_pe(self,kf,device,dtype):return torch.arange(kf,device=device,dtype=dtype)[:,None].expand(kf,2)
        def _time_softmax_on_v(self,q,k,v,b,kf):return v
        def forward(self,x,cu_slens=None,num_frames=1):
            assert cu_slens is None and num_frames==6
            h=self.norm1(x.view(1,6,196,2)+self._build_time_pe(6,x.device,x.dtype).view(1,6,1,2)).view(6,196,2)
            x=x+self.attend(h);return x+self.mlp(self.norm2(x))
    class Tower(nn.Module):
        def __init__(self):
            super().__init__();self.patch_embed=Patch();self.blocks=nn.Sequential(Block(),Block(),Block(),SpaceTimeBlock())
        def rescale_positional_embedding(self,out_size):return torch.zeros(1,196,2,dtype=torch.bfloat16)
    tower=Tower().to(torch.bfloat16).eval()
    with torch.no_grad():
        for p in tower.parameters():p.zero_()
    def video(tower,images):
        x=tower.patch_embed(images.reshape(6,3,224,224))+tower.rescale_positional_embedding((14,14))
        for i,blk in enumerate(tower.blocks):x=blk(x,cu_slens=None,num_frames=6) if i==3 else blk(x)
        return x.view(1,6,196,2)[:,-1],(14,14)
    images=torch.arange(18,dtype=torch.bfloat16).reshape(3,6,1,1,1).expand(3,6,3,224,224).clone()/16
    return tower,video,images


def test_official_actual_module_hooks_keep_camera_order_four_blocks_and_full_video_norm_shapes(prefix):
    tower,video,images=tiny_official();before=images.clone();old_hook_ids={id(m):(set(m._forward_hooks),set(m._forward_pre_hooks)) for m in tower.modules()}
    raw,state=prefix.observe_official(tower,video,images,synchronize=lambda:None)
    assert len(raw)==124 and state['complete'] and state['cleanup']['passed']
    assert state['camera_block_indices']==[[0,1,2,3]]*3
    for camera,value in ((0,.3125),(1,.6875),(2,1.0625)):
        assert torch.equal(raw[f'camera{camera}.api_current_frame'],torch.full((1,196,2),value,dtype=torch.bfloat16))
        assert raw[f'camera{camera}.block3.norm1_input'].shape==raw[f'camera{camera}.block3.norm1'].shape==(1,6,196,2)
        for i in range(1,4):assert torch.equal(raw[f'camera{camera}.block{i}.input'],raw[f'camera{camera}.block{i-1}.output'])
    assert torch.equal(images,before) and '_frozen_vit_prefix_active' not in tower.__dict__
    assert all((set(m._forward_hooks),set(m._forward_pre_hooks))==old_hook_ids[id(m)] for m in tower.modules())
    assert 'rescale_positional_embedding' not in tower.__dict__ and '_build_time_pe' not in tower.blocks[3].__dict__


@pytest.mark.parametrize('bad',['body','nested','sync','readonly','missing_stage'])
def test_official_observer_never_leaks_hooks_or_complete_on_failures(prefix,bad):
    tower,video,images=tiny_official();hooks={id(m):(set(m._forward_hooks),set(m._forward_pre_hooks)) for m in tower.modules()}
    def run(tower,x):
        if bad=='body':raise RuntimeError('official body failed')
        if bad=='nested':return prefix.observe_official(tower,video,images,synchronize=lambda:None)
        if bad=='missing_stage':return torch.zeros(1,196,2,dtype=torch.bfloat16),(14,14)
        value=video(tower,x)
        if bad=='readonly':x.add_(1)
        return value
    def sync():
        if bad=='sync':raise RuntimeError('official sync failed')
    with pytest.raises((ValueError,RuntimeError),match='body|[Nn]ested|active|sync|readonly|coverage|stage|block'):
        prefix.observe_official(tower,run,images,synchronize=sync)
    assert all((set(m._forward_hooks),set(m._forward_pre_hooks))==hooks[id(m)] for m in tower.modules())
    assert '_frozen_vit_prefix_active' not in tower.__dict__ and 'rescale_positional_embedding' not in tower.__dict__


def prefix_raw(width=1152):
    def zero(shape):return torch.zeros((),dtype=torch.bfloat16).expand(shape)
    official={'prepared.images':zero((3,6,3,224,224))};native=dict(official)
    for camera in range(3):
        base=f'camera{camera}.'
        official[base+'patch']=zero((6,196,width));official[base+'position']=zero((1,196,width))
        official[base+'api_current_frame']=zero((1,196,width))
        official[base+'block3.time_pe']=zero((6,width));official[base+'block3.temporal_v']=zero((6,16,196,width//16))
        for i in range(4):
            for field in ('input','norm1_input','norm1','qkv','attention','residual','norm2','mlp','output'):
                shape=(1,6,196,width) if i==3 and field in ('norm1_input','norm1') else (6,196,3*width if field=='qkv' else width)
                official[base+f'block{i}.'+field]=zero(shape)
    native.update(patch=zero((18,196,width)),position=zero((1,196,width)))
    native['control.block3.output']=zero((18,196,width));native['block3.time_pe']=zero((6,width))
    native['block3.temporal_v']=zero((18,16,196,width//16))
    for i in range(4):
        for field in ('input','norm1_input','norm1','qkv','attention','residual','norm2','mlp','output'):
            shape=(3,6,196,width) if i==3 and field in ('norm1_input','norm1') else (18,196,3*width if field=='qkv' else width)
            native[f'block{i}.'+field]=zero(shape)
    return official,native


@pytest.mark.parametrize('bad',['none','dummy','dtype','norm_shape','control','chain'])
def test_physical_124_and42_schema_rejects_missing_fields_shapes_and_control_drift(prefix,bad):
    official,native=prefix_raw()
    if bad=='dummy':del official['camera0.block0.qkv'];official['dummy']=torch.zeros(1,dtype=torch.bfloat16)
    if bad=='dtype':native['block2.norm1']=native['block2.norm1'].float()
    if bad=='norm_shape':official['camera1.block3.norm1']=torch.zeros(6,196,1152,dtype=torch.bfloat16)
    if bad=='control':native['control.block3.output']=native['control.block3.output']+1
    if bad=='chain':native['block1.input']=native['block1.input']+1
    def check():
        assert prefix.validate_raw(official,engine='official')['tensors']==124
        assert prefix.validate_raw(native,engine='native')['tensors']==42
    if bad=='none':check()
    else:
        with pytest.raises(ValueError,match='raw|role|shape|dtype|control|chain|residual'):check()


def test_gold_actual_block3_endpoint_is_external_anchor_not_self_consistent_saved_answer(prefix):
    official,_=prefix_raw();C=1152;called=[]
    gold={'images':official['prepared.images'],'patches':[official[f'camera{i}.patch'] for i in range(3)],
        'position':official['camera0.position'],'block3':[torch.ones((),dtype=torch.bfloat16).expand(6,196,C) for _ in range(3)]}
    with pytest.raises(ValueError,match='Gold|anchor|block3'):
        prefix.admit_official_capture(official,gold,factory=lambda:called.append('native process consumer'))
    assert not called


@pytest.mark.parametrize('changed',[None,'block1.qkv','block3.time_pe'])
def test_comparison_finds_first_actual_stage_and_reports_camera_frame_differences(prefix,changed):
    official,native=prefix_raw(width=16)
    if changed is not None:native[changed]=torch.ones_like(native[changed])
    result=prefix.compare_prefix(official,native)
    assert result['first_different_stage']==changed
    if changed is not None:
        row=result['stages'][changed]
        assert row['max_abs']==1. and row['unequal_elements']==native[changed].numel()*(3 if changed=='block3.time_pe' else 1)
        assert len(row['per_camera'])==3 and all(len(c['per_frame'])==6 for c in row['per_camera'])
    else:assert all(row['exact'] for row in result['stages'].values())
