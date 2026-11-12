import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
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


def input_manifest(prefix,tmp_path):
    task=tmp_path;roles={}
    for origin,folder,engine in (('gold','hy-trace-308690','official'),('native','hy-folded-state-trace-313766','native'),
            ('current','hy-native-policy-tree-316975','native/control')):
        for case in ('baseline','state'):
            for kind,name in (('report','result.json'),('raw','trace.safetensors')):
                p=task/'results'/folder/engine/case
                if origin=='current':p=p/'replay0'
                roles[f'{origin}.{case}.{kind}']=p/name
    roles.update({'r33':task/'results/hy-native-policy-tree-316975/native/result.json',
        **{'runtime.'+e:task/'results/hy-rms-stages-315256'/e/'result.json' for e in ('official','native')}})
    for p in roles.values():p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
    off=task/'snapshots/round007/diagnostic/official/hy_vla';source=off/'hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    native=task/'snapshots/round038/code/flash_rt/models/hyvla/pipeline_thor.py'
    records={}
    for role,p in {'official_modeling':source,'official_video':off/'space_time_attention.py','official_policy':off/'modeling_hy_vla.py',
            'official_configuration':off/'configuration_hy_vla.py','native':native,'config':task/'checkpoint/config.json'}.items():
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}' if role=='config' else '# source')
        records[role]=prefix.file_record(p)
    (off/'__init__.py').write_text('# historical package source')
    roles['gold.baseline.report'].write_text(json.dumps({'hy_vla_source':str(off/'__init__.py')}))
    ck=task/'checkpoint/model.safetensors';ck.write_bytes(b'no tensor data')
    history={'kind':'hy_frozen_complete_merger_inputs','task_root':str(task),'checkpoint':prefix.file_record(ck),
        'official_source':prefix.file_record(source),'records':{k:prefix.file_record(p) for k,p in roles.items()}}
    path=task/'results/hy-merger-stages-317360/inputs.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(history))
    manifest={'kind':'hy_frozen_actual_vit_prefix_inputs','task_root':str(task),'history_manifest':prefix.file_record(path),'source_files':records}
    return manifest,native


@pytest.mark.parametrize('bad',['none','old_raw_stat','config_stat','new_source_path','old_source_path','missing_role','history_path','package_file','package_round'])
def test_input_source_config_history_paths_and_stats_precede_any_weight_consumer(prefix,tmp_path,bad):
    manifest,native=input_manifest(prefix,tmp_path);called=[]
    if bad=='old_raw_stat':
        path=tmp_path/'results/hy-trace-308690/official/baseline/trace.safetensors';path.write_bytes(b'changed old actual raw')
    if bad=='config_stat':Path(manifest['source_files']['config']['path']).write_text('{"changed":true}')
    if bad=='new_source_path':manifest['source_files']['native']=manifest['source_files']['official_video']
    if bad=='old_source_path':manifest['source_files']['official_video']=manifest['source_files']['official_policy']
    if bad=='missing_role':manifest['source_files'].pop('official_configuration')
    if bad=='history_path':manifest['history_manifest']['path']=str(tmp_path/'other.json')
    if bad in ('package_file','package_round'):
        historical=Path(manifest['history_manifest']['path']);history=json.loads(historical.read_text())
        report=Path(history['records']['gold.baseline.report']['path']);root=tmp_path/'snapshots/round007/diagnostic/official/hy_vla'
        report.write_text(json.dumps({'hy_vla_source':str(root/'modeling_hy_vla.py' if bad=='package_file' else tmp_path/'snapshots/other/official/hy_vla/__init__.py')}))
        history['records']['gold.baseline.report']=prefix.file_record(report);historical.write_text(json.dumps(history));manifest['history_manifest']=prefix.file_record(historical)
    call=lambda:prefix.admit_before_weights(manifest,native_source=native,factory=lambda m:called.append('weight consumer') or m)
    if bad=='none':assert call()['kind']=='hy_frozen_complete_merger_inputs' and called==['weight consumer']
    else:
        with pytest.raises((ValueError,FileNotFoundError),match='source|record|path|stat|role|manifest|inputs|No such'):call()
        assert not called


@pytest.mark.parametrize('bad',['none','job','source','runtime','weight_keys','full_model','anchor','cleanup'])
def test_native_predecessor_official_header_and_actual_Gold_raw_precede_51_weight_reads(prefix,bad):
    raw,_=prefix_raw();gold={'images':raw['prepared.images'],'patches':[raw[f'camera{i}.patch'] for i in range(3)],
        'position':raw['camera0.position'],'block3':[raw[f'camera{i}.block3.output'] for i in range(3)]}
    report={'kind':'hy_actual_frozen_vit_prefix','engine':'official','status':'prefix_capture_complete','job_id':'9','source_revision':'a'*40,
        'runtime':{'torch':'old'},'torch_git_version':'b'*40,'checkpoint':{'bytes':17},'input_manifest':{'identity':'fixture'},'source_records':{'driver':{}},
        'weight_read_events':51,'selected_checkpoint_keys':['model.dual_tower.vlm.model.visual.vision_tower.'+k for k in tensor_schema()],
        'full_model_constructed':False,'blocks_constructed':4,
        'raw_proof':{'tensors':124,'block_chain_exact':True,'residual_chains_exact':True,'actual_endpoint_bridge_exact':True},
        'observations':{'complete':True,'cleanup':{'passed':True},'camera_block_indices':[[0,1,2,3]]*3}}
    if bad=='job':report['job_id']='8'
    if bad=='source':report['source_revision']='c'*40
    if bad=='runtime':report['runtime']={'torch':'new'}
    if bad=='weight_keys':report['selected_checkpoint_keys'][-1]='blocks.4.norm1.weight'
    if bad=='full_model':report['full_model_constructed']=True
    if bad=='anchor':gold['block3'][0]=torch.ones_like(gold['block3'][0])
    if bad=='cleanup':report['observations']['cleanup']['passed']=False
    called=[]
    call=lambda:prefix.admit_prior_official(report,raw,gold,job_id='9',source_revision='a'*40,runtime={'torch':'old'},torch_git='b'*40,
        checkpoint={'bytes':17},input_manifest={'identity':'fixture'},source_records={'driver':{}},factory=lambda:called.append('51-key reader'))
    if bad=='none':call();assert called==['51-key reader']
    else:
        with pytest.raises(ValueError,match='prior|official|source|runtime|anchor|Gold|weight|header|cleanup'):call()
        assert not called


def test_help_and_allocation_guard_before_any_source_or_weight_read(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--inputs' in r.stdout and '--official-prefix' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--inputs','/missing','--source-revision','a'*40,
        '--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()


def physical_weights():
    return {name:torch.full((),(i+1)/64,dtype=torch.bfloat16).expand(shape) for i,(name,shape) in enumerate(tensor_schema().items())}


def test_native_standalone_owner_uses_exact_selected_objects_without_constructing_full_pipeline(prefix):
    weights=physical_weights();owner=prefix.make_native_owner(weights,source=NATIVE)
    assert owner._fp8 is False and owner.vit_heads==16 and owner.vit_hd==72 and owner.vit_spacetime_ids=={3,7,11,15,19,23}
    assert owner.W._vit_patch_w is weights['patch_embed.proj.weight']
    assert len(owner.W._vit_qkv_w)==4 and owner.W._vit_fc1_w[3] is weights['blocks.3.mlp.fc1.weight']
    assert owner._prefix_source==NATIVE and all(getattr(owner,k).__self__ is owner for k in prefix.METHODS)


@pytest.mark.parametrize('bad',['missing','dtype','shape'])
def test_invalid_prefix_weight_set_rejects_before_official_module_constructor(prefix,bad):
    weights=physical_weights();called=[]
    if bad=='missing':weights.pop('blocks.0.norm1.bias')
    if bad=='dtype':weights['blocks.0.norm1.weight']=weights['blocks.0.norm1.weight'].float()
    if bad=='shape':weights['blocks.0.mlp.fc1.bias']=torch.ones(4303,dtype=torch.bfloat16)
    vision=SimpleNamespace(_create_hyvit2=lambda **kw:called.append(kw))
    with pytest.raises(ValueError,match='weight|shape|BF16|keys'):
        prefix.make_official_prefix(vision,SimpleNamespace(),weights,device='cpu')
    assert not called


def test_official_factory_constructs_only_four_blocks_and_loads_every_selected_parameter(prefix):
    import torch.nn as nn
    def factory(*,select_layer,ckpt_path,weight_init):
        assert select_layer==4 and ckpt_path=='' and weight_init=='skip'
        tower=nn.Module();tower.pos_embed=nn.Parameter(torch.empty(1,16384,1152));tower.patch_embed=nn.Module()
        tower.patch_embed.proj=nn.Conv2d(3,1152,16,bias=True);tower.blocks=nn.Sequential()
        tower.patch_drop=nn.Identity();tower.norm_pre=nn.Identity()
        for i in range(4):
            b=nn.Module();b.norm1=nn.LayerNorm(1152);b.norm2=nn.LayerNorm(1152);b.attn=nn.Module()
            b.attn.qkv=nn.Linear(1152,3456);b.attn.proj=nn.Linear(1152,1152);b.attn.q_bias=None
            b.attn.q_norm=nn.Identity();b.attn.k_norm=nn.Identity();b.attn.num_heads=16;b.attn.head_dim=72
            b.mlp=nn.Module();b.mlp.fc1=nn.Linear(1152,4304);b.mlp.fc2=nn.Linear(4304,1152)
            b.ls1=nn.Identity();b.ls2=nn.Identity();b.drop_path1=nn.Identity();b.drop_path2=nn.Identity();tower.blocks.add_module(str(i),b)
        return tower
    class SpaceTimeBlock(nn.Module):
        def __init__(self,base_block,*,max_num_frames,learnable_time_embed,time_embed_base):
            super().__init__();assert (max_num_frames,learnable_time_embed,time_embed_base)==(18,False,100.)
            for k,v in base_block._modules.items():setattr(self,k,v)
    values=physical_weights();tower=prefix.make_official_prefix(SimpleNamespace(_create_hyvit2=factory),SimpleNamespace(SpaceTimeBlock=SpaceTimeBlock),values,device='cpu')
    assert len(tower.blocks)==4 and isinstance(tower.blocks[3],SpaceTimeBlock) and tower.past_drop_layer is None
    assert set(tower.state_dict())==set(values) and all(not m.training for m in tower.modules())
    assert all(torch.equal(v,values[k]) and v.dtype==torch.bfloat16 for k,v in tower.state_dict().items())


@pytest.mark.parametrize('bad',['none','state_images','patch_dtype','block3_missing','layout'])
def test_actual_Gold_images_patch_block3_roles_and_shared_context_admit_before_forward(prefix,bad):
    def zero(shape):return torch.zeros((),dtype=torch.bfloat16).expand(shape)
    samples={c:{'images':zero((3,6,3,224,224)),'patches':tuple(zero((6,196,1152)) for _ in range(3)),
        'position':zero((1,196,1152)),'block3':tuple(zero((6,196,1152)) for _ in range(3)),
        'image_metadata':{'shape':[3,6,3,224,224],'dtype':'torch.bfloat16','live_stride':[903168,150528,50176,224,1]}} for c in ('baseline','state')}
    if bad=='state_images':samples['state']['images']=torch.ones_like(samples['state']['images'])
    if bad=='patch_dtype':samples['baseline']['patches']=(samples['baseline']['patches'][0].float(),*samples['baseline']['patches'][1:])
    if bad=='block3_missing':samples['baseline']['block3']=samples['baseline']['block3'][:2]
    if bad=='layout':samples['baseline']['image_metadata']['live_stride'][0]+=1
    called=[]
    def consume():
        result=prefix.admit_gold_inputs(samples);called.append('actual four-block owner');return result
    if bad=='none':assert consume() is samples['baseline'] and called==['actual four-block owner']
    else:
        with pytest.raises(ValueError,match='Gold|image|shared|block3|BF16|layout|role'):consume()
        assert not called


@pytest.mark.parametrize('failure',['raw_write','incomplete'])
def test_prefix_publication_never_marks_partial_or_raw_write_failure_complete(prefix,tmp_path,monkeypatch,failure):
    import safetensors.torch
    raw,_=prefix_raw();report={'engine':'official','observations':{'complete':failure!='incomplete','cleanup':{'passed':True}}}
    if failure=='raw_write':
        def fail(*a,**kw):raise OSError('prefix raw write failed')
        monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises((OSError,ValueError),match='raw write|complete|cleanup'):
        prefix.publish_capture(tmp_path,raw,report)
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('evidence_failure',[False,True])
def test_post_observation_Gold_rejection_retains_unqualified_raw_state_and_original_error(prefix,tmp_path,monkeypatch,evidence_failure):
    from safetensors.torch import load_file
    raw,_=prefix_raw();state={'complete':True,'cleanup':{'passed':True},'layouts':{'evidence':'captured before anchor check'}}
    def observed(*a,**kw):return raw,state
    monkeypatch.setattr(prefix,'observe_official',observed)
    gold={'images':raw['prepared.images'],'patches':[raw[f'camera{i}.patch'] for i in range(3)],'position':raw['camera0.position'],
        'block3':[torch.ones_like(raw['camera0.block3.output']),raw['camera1.block3.output'],raw['camera2.block3.output']]}
    with pytest.raises(ValueError,match='Gold anchor') as caught:
        prefix.qualified_capture(None,None,engine='official',video_function=None,gold=gold,official_raw=None,synchronize=lambda:None)
    original=caught.value
    if evidence_failure:
        import safetensors.torch
        def broken(*a,**kw):raise OSError('evidence disk failed')
        monkeypatch.setattr(safetensors.torch,'save_file',broken)
    prefix.write_failure(tmp_path,original)
    failed=json.loads((tmp_path/'failure.json').read_text())
    assert failed['qualified'] is False and failed['observations']['complete'] is True and failed['observations']['cleanup']['passed'] is True
    assert failed['error_type']=='ValueError' and 'Gold anchor' in failed['error'] and not (tmp_path/'result.json').exists()
    if evidence_failure:
        assert failed['evidence_write_errors'] and any('evidence disk failed' in note for note in original.__notes__)
    else:
        saved=load_file(str(tmp_path/'unqualified_partial.safetensors'));assert len(saved)==124
        assert torch.equal(saved['camera0.block3.output'],raw['camera0.block3.output'])


def test_native_cleanup_only_failure_carries_full_CPU_partial_to_failure_audit(prefix,tmp_path):
    from safetensors.torch import load_file
    owner,images=tiny_native(prefix);old=dict(owner.__dict__)
    def sync():raise RuntimeError('final sync failed')
    with pytest.raises(RuntimeError,match='cleanup') as caught:prefix.capture_native(owner,images,synchronize=sync)
    raw,state=caught.value.prefix_partial
    assert len(raw)==42 and state['complete'] is False and state['cleanup']['passed'] is False
    prefix.write_failure(tmp_path,caught.value)
    assert len(load_file(str(tmp_path/'unqualified_partial.safetensors')))==42
    failed=json.loads((tmp_path/'failure.json').read_text());assert failed['qualified'] is False and failed['observations']['cleanup']['passed'] is False
    assert not (tmp_path/'result.json').exists() and set(owner.__dict__)==set(old)
