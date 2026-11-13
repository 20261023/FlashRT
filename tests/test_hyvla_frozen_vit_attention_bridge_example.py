import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import MethodType,SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_attention_bridge.py'
NATIVE=ROOT/'flash_rt/models/hyvla/pipeline_thor.py'


@pytest.fixture
def bridge():
    spec=importlib.util.spec_from_file_location('hy_actual_attention_bridge_test',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('bad',['none','dtype','shape','finite'])
def test_exact_four_weight_reader_never_reads_prefix_or_policy_keys(bridge,bad):
    shapes={'qkv.weight':(3456,1152),'qkv.bias':(3456,),'proj.weight':(1152,1152),'proj.bias':(1152,)};seen=[]
    class Reader:
        def get_tensor(self,key):
            base='model.dual_tower.vlm.model.visual.vision_tower.blocks.0.attn.'
            assert key.startswith(base);name=key[len(base):];assert name in shapes;seen.append(name)
            t=torch.zeros((),dtype=torch.bfloat16).expand(shapes[name])
            if name=='proj.bias':
                if bad=='dtype':t=t.float()
                if bad=='shape':t=t[:1151]
                if bad=='finite':t=torch.full((1152,),float('nan'),dtype=torch.bfloat16)
            return t
    if bad=='none':
        w=bridge.read_attention_weights(Reader(),device='cpu')
        assert set(w)==set(shapes) and len(seen)==4 and len(set(seen))==4
        assert sum(v.numel()*v.element_size() for v in w.values())==10626048
    else:
        with pytest.raises(ValueError,match='weight|BF16|shape|finite'):bridge.read_attention_weights(Reader(),device='cpu')


def tiny_weights():
    eye=torch.eye(16,dtype=torch.bfloat16)
    return {'qkv.weight':torch.cat([eye]*3),'qkv.bias':torch.zeros(48,dtype=torch.bfloat16),
        'proj.weight':eye.clone(),'proj.bias':torch.full((16,),.125,dtype=torch.bfloat16)}


def tiny_owner(bridge,engine):
    import torch.nn as nn
    w=tiny_weights()
    if engine=='official':
        class ActualAttention(nn.Module):
            def __init__(self):
                super().__init__();self.qkv=nn.Linear(16,48,dtype=torch.bfloat16);self.proj=nn.Linear(16,16,dtype=torch.bfloat16)
            def forward(self,x):
                B,N,C=x.shape;q,k,v=self.qkv(x).reshape(B,N,3,2,8).permute(2,0,3,1,4).unbind(0)
                c=torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=8**-.5)
                return self.proj(c.transpose(1,2).reshape(B,N,C))
        owner=ActualAttention();owner.load_state_dict(w);owner.eval()
    else:
        owner=SimpleNamespace(_vit_f8=False,vit_heads=2,vit_hd=8,vit_scale=8**-.5,_bridge_source=NATIVE,
            _vit_qkv_w_cur=w['qkv.weight'],_vit_qkv_b_cur=w['qkv.bias'],_vit_proj_w_cur=w['proj.weight'],_vit_proj_b_cur=w['proj.bias'])
        for name,fn in bridge.load_native_methods(NATIVE).items():setattr(owner,name,MethodType(fn,owner))
    value=(torch.arange(196*16,dtype=torch.bfloat16).remainder(16)/16).reshape(1,196,16)
    inputs=value.expand(18,196,16).clone()+torch.arange(18,dtype=torch.bfloat16)[:,None,None]/16
    return owner,tuple(x.clone() for x in inputs.split(6)) if engine=='official' else inputs


@pytest.mark.parametrize('engine',['official','native'])
def test_actual_context_is_captured_before_projection_and_self_replays_preserve_API(bridge,engine):
    owner,inputs=tiny_owner(bridge,engine);before=[x.clone() for x in inputs] if engine=='official' else inputs.clone()
    old=dict(owner.__dict__);raw,state,live=bridge.capture_actual(engine,owner,inputs,synchronize=lambda:None)
    assert state['complete'] and state['cleanup']['passed'] and len(raw)==(21 if engine=='official' else 7)
    assert state['api_calls']=={'control':3 if engine=='official' else 1,'observed':3 if engine=='official' else 1}
    assert state['observed_qkv_calls']==state['observed_projection_calls']==(3 if engine=='official' else 1)
    for i in range(3 if engine=='official' else 1):
        p=f'camera{i}.' if engine=='official' else ''
        assert torch.equal(raw[p+'api'],raw[p+'projection']) and torch.equal(raw[p+'api'],raw[p+'control'])
        assert torch.equal(raw[p+'api'],raw[p+'self_projection'])
        assert torch.equal(raw[p+'context']+.125,raw[p+'projection'])
        assert not torch.equal(raw[p+'context'],raw[p+'projection'])
        assert state['layouts'][p+'context']['shape']==[6 if engine=='official' else 18,196,16]
    assert len(live)==3 if engine=='official' else live.shape==(18,196,16)
    if engine=='official':
        assert all(torch.equal(x,y) for x,y in zip(inputs,before))
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in owner.modules())
    else:
        assert torch.equal(inputs,before) and all(owner.__dict__[k] is v for k,v in old.items())
    assert set(owner.__dict__)==set(old)


@pytest.mark.parametrize('engine',['official','native'])
@pytest.mark.parametrize('failure',['body','nested','sync'])
def test_actual_capture_cleanup_preserves_original_owner_and_failed_evidence(bridge,engine,failure):
    owner,inputs=tiny_owner(bridge,engine)
    name='forward' if engine=='official' else '_vit_qkv';original=getattr(owner,name)
    def broken(*a,**kw):
        if failure=='body':raise RuntimeError('actual API body failed')
        if failure=='nested':return bridge.capture_actual(engine,owner,inputs,synchronize=lambda:None)
        return original(*a,**kw)
    setattr(owner,name,broken);old=dict(owner.__dict__)
    def sync():
        if failure=='sync':raise RuntimeError('actual API sync failed')
    with pytest.raises((ValueError,RuntimeError),match='body|[Nn]ested|active|sync|cleanup') as caught:
        bridge.capture_actual(engine,owner,inputs,synchronize=sync)
    raw,state=caught.value.bridge_partial
    assert state['complete'] is False and raw and set(owner.__dict__)==set(old)
    if engine=='official':assert all(not m._forward_hooks and not m._forward_pre_hooks for m in owner.modules())
    assert getattr(owner,name) is broken


@pytest.mark.parametrize('bad',['signature','duplicate'])
def test_native_observation_source_drift_is_rejected_not_reimplemented(bridge,tmp_path,bad):
    source=NATIVE.read_text()
    if bad=='signature':source=source.replace('def _vit_spatial_attn(self, q, k, v):','def _vit_spatial_attn(self, q, k):')
    else:source=source.replace('    def _vit_qkv(self, h):','    def _vit_qkv(self, h):\n        return h\n\n    def _vit_qkv(self, h):')
    p=tmp_path/'native.py';p.write_text(source)
    with pytest.raises(ValueError,match='source|signature|AST|method'):bridge.load_native_methods(p,observer=lambda *a:None)


def live_context():
    data=torch.arange(18*196*16+16,dtype=torch.bfloat16).remainder(16)/16
    return data[16:].reshape(18,196,16)


def test_four_local_projection_cuts_preserve_live_storage_offsets_bias_and_restore(bridge):
    x=live_context();before=x.clone();old=before-.25;w=tiny_weights();ptr=x.untyped_storage().data_ptr();layout=(x.stride(),x.storage_offset())
    raw,state=bridge.projection_cuts(x,old,w,native_reference=before+.125,synchronize=lambda:None)
    assert len(raw)==10 and state['complete'] and state['cleanup']['passed'] and state['restored_context_exact']
    assert state['projection_calls']==8 and state['restore_projection_calls']==1
    assert x.untyped_storage().data_ptr()==ptr and (x.stride(),x.storage_offset())==layout and torch.equal(x,before)
    for cell in ('N18','N6','O18','O6'):
        want=old if cell.startswith('O') else before
        assert torch.equal(raw['cuts.'+cell+'.input'],want) and torch.equal(raw['cuts.'+cell+'.output'],want+.125)
        views=state['cells'][cell]['views'];assert len(views)==(3 if cell.endswith('6') else 1)
        assert all(v['shares_live_storage'] for v in views)
        if cell.endswith('6'):
            assert [v['shape'] for v in views]==[[6,196,16]]*3
            assert [v['storage_offset'] for v in views]==[16+i*6*x.stride(0) for i in range(3)]
    assert torch.equal(raw['cuts.restored_context'],before) and torch.equal(raw['cuts.restore_projection'],before+.125)
    qualification=bridge.qualify_cuts(raw,state,native_reference=before+.125,old_reference=torch.full_like(before,99.))
    assert qualification['native_control_and_restoration_exact'] and qualification['old_corner_exact'] is False
    assert qualification['comparisons']['O6']['old']['exact'] is False


@pytest.mark.parametrize('bad',['dtype','shape','overlap','weight_alias','old_finite'])
def test_unsupported_projection_cut_operands_reject_before_live_value_substitution(bridge,bad):
    x=live_context();w=tiny_weights();old=x.clone()-.25
    if bad=='dtype':old=old.float()
    if bad=='shape':x=x[:,:195]
    if bad=='overlap':x=x[:1].expand(18,-1,-1)
    if bad=='weight_alias':w['proj.weight']=x.reshape(-1)[:256].view(16,16)
    if bad=='old_finite':old[0,0,0]=float('nan')
    before=x.clone()
    with pytest.raises(ValueError,match='cut|context|BF16|layout|alias|overlap|geometry'):
        bridge.projection_cuts(x,old,w,native_reference=torch.zeros_like(x),synchronize=lambda:None)
    assert torch.equal(x,before)


@pytest.mark.parametrize('failure',['body','sync','weights'])
def test_projection_cuts_restore_live_values_and_keep_partial_on_failures(bridge,monkeypatch,failure):
    import torch.nn.functional as F
    x=live_context();before=x.clone();w=tiny_weights();original=F.linear;calls=[]
    def broken(*a,**kw):
        calls.append(True)
        if failure=='body' and len(calls)==5:raise RuntimeError('old context cut body failed')
        if failure=='weights' and len(calls)==2:w['proj.bias'].add_(.25)
        return original(*a,**kw)
    monkeypatch.setattr(F,'linear',broken)
    def sync():
        if failure=='sync':raise RuntimeError('cut final sync failed')
    with pytest.raises((ValueError,RuntimeError),match='body|sync|cleanup|readonly|weight') as caught:
        bridge.projection_cuts(x,before-.25,w,native_reference=before+.125,synchronize=sync)
    raw,state=caught.value.bridge_partial
    assert raw and state['complete'] is False and torch.equal(x,before)
    assert state['restored_context_exact'] and 'cuts.restored_context' in raw


@pytest.mark.parametrize('joint,expected',[(6.,0.),(8.,2.)])
def test_literal_projection_interaction_is_actual_FP64_finite_difference(bridge,joint,expected):
    cuts={f'cuts.{g}.output':torch.full((18,196,16),v,dtype=torch.bfloat16) for g,v in {'N18':1.,'N6':3.,'O18':4.,'O6':joint}.items()}
    value,stats=bridge.interaction(cuts)
    assert value.dtype==torch.float64 and value.device.type=='cpu' and torch.equal(value,torch.full(value.shape,expected,dtype=torch.float64))
    assert stats['nonzero_elements']==(0 if expected==0 else value.numel()) and stats['max_abs']==expected


def test_native_self_value_control_mismatch_stops_before_any_other_cut(bridge):
    x=live_context();before=x.clone()
    with pytest.raises(ValueError,match='native|N18|control') as caught:
        bridge.projection_cuts(x,before-.25,tiny_weights(),native_reference=before+1,synchronize=lambda:None)
    raw,state=caught.value.bridge_partial
    assert state['projection_calls']==1 and not state['complete'] and torch.equal(x,before)
    assert 'cuts.N18.output' in raw and 'cuts.N6.input' not in raw and 'cuts.O18.input' not in raw


@pytest.mark.parametrize('bad',['none','counter','finite','dtype','input_pair','view_alias'])
def test_complete_cut_qualification_rejects_false_counterproofs_or_tampered_raw(bridge,bad):
    x=live_context();before=x.clone();raw,state=bridge.projection_cuts(x,before-.25,tiny_weights(),native_reference=before+.125,synchronize=lambda:None)
    if bad=='counter':state['projection_calls']=7
    if bad=='finite':raw['cuts.O6.input'][0,0,0]=float('nan')
    if bad=='dtype':raw['cuts.O18.output']=raw['cuts.O18.output'].float()
    if bad=='input_pair':raw['cuts.O6.input']+=1
    if bad=='view_alias':state['cells']['O6']['views'][1]['shares_live_storage']=False
    if bad=='none':assert bridge.qualify_cuts(raw,state,native_reference=before+.125,old_reference=before)['native_control_and_restoration_exact']
    else:
        with pytest.raises(ValueError,match='cut|Projection|raw|role|dtype|finite|view|input|coverage'):
            bridge.qualify_cuts(raw,state,native_reference=before+.125,old_reference=before)


def actual_schema(bridge,engine):
    def zero(shape):return torch.zeros((),dtype=torch.bfloat16).expand(shape)
    raw={};refs={'norm1':[],'qkv':[],'post':[]};state={'complete':True,'cleanup':{'passed':True},'layouts':{},'context_aliases_fixed_inputs':False,
        'api_calls':{'control':3 if engine=='official' else 1,'observed':3 if engine=='official' else 1},
        'observed_qkv_calls':3 if engine=='official' else 1,'observed_projection_calls':3 if engine=='official' else 1}
    for i in range(3 if engine=='official' else 1):
        base=f'camera{i}.' if engine=='official' else '';B=6 if engine=='official' else 18
        for field in ('norm1','qkv','context','projection','api','control','self_projection'):
            shape=(B,196,3456 if field=='qkv' else 1152);raw[base+field]=zero(shape)
            state['layouts'][base+field]={'shape':list(shape),'dtype':'torch.bfloat16','live_stride':[196*shape[-1],shape[-1],1],'storage_offset':0}
        for name,key in (('norm1','norm1'),('qkv','qkv'),('post','api')):refs[name].append(raw[base+key])
    return raw,state,{k:tuple(v) if engine=='official' else v[0] for k,v in refs.items()}


@pytest.mark.parametrize('engine',['official','native'])
@pytest.mark.parametrize('bad',['none','dummy','dtype','reference','calls','cleanup'])
def test_physical_21_or7_API_capture_and_R38_references_are_mandatory(bridge,engine,bad):
    raw,state,refs=actual_schema(bridge,engine);base='camera0.' if engine=='official' else ''
    if bad=='dummy':del raw[base+'qkv'];raw['dummy']=torch.ones(1,dtype=torch.bfloat16)
    if bad=='dtype':raw[base+'context']=raw[base+'context'].float()
    if bad=='reference':
        if engine=='official':refs['post']=(torch.ones_like(refs['post'][0]),*refs['post'][1:])
        else:refs['post']=torch.ones_like(refs['post'])
    if bad=='calls':state['observed_qkv_calls']=0
    if bad=='cleanup':state['cleanup']['passed']=False
    call=lambda:bridge.validate_capture(raw,state,engine=engine,references=refs)
    if bad=='none':assert call()['tensors']==(21 if engine=='official' else 7)
    else:
        with pytest.raises(ValueError,match='raw|role|dtype|reference|anchor|calls|cleanup|capture'):call()


def test_interaction_does_not_silently_accept_another_output_dtype(bridge):
    cuts={f'cuts.{g}.output':torch.ones(18,196,16,dtype=torch.bfloat16) for g in ('N18','N6','O18','O6')}
    cuts['cuts.O6.output']=cuts['cuts.O6.output'].float()
    with pytest.raises(ValueError,match='dtype|BF16|interaction'):bridge.interaction(cuts)


@pytest.mark.parametrize('engine',['official','native'])
def test_unsupported_context_alias_is_recorded_in_failure_proof_before_cuts(bridge,monkeypatch,engine):
    owner,inputs=tiny_owner(bridge,engine)
    monkeypatch.setattr(torch.nn.functional,'scaled_dot_product_attention',lambda q,k,v,**kw:v)
    with pytest.raises(ValueError,match='aliases') as caught:bridge.capture_actual(engine,owner,inputs,synchronize=lambda:None)
    raw,state=caught.value.bridge_partial
    assert raw and not state['complete'] and state['context_aliases_fixed_inputs'] is True


@pytest.mark.parametrize('bad',['none','missing_cut','forged_context_pair','missing_I','wrong_I'])
def test_native_17_raw_and_one_interaction_bind_actual_contexts_before_completion(bridge,bad):
    raw,actual,refs=actual_schema(bridge,'native');C=1152
    context=torch.zeros((),dtype=torch.bfloat16).expand(18,196,C)
    for cell in ('N18','N6','O18','O6'):
        for field in ('input','output'):raw[f'cuts.{cell}.{field}']=context
    raw.update({'cuts.restored_context':context,'cuts.restore_projection':context})
    cut={'complete':True,'cleanup':{'passed':True},'restored_context_exact':True,'projection_calls':8,'restore_projection_calls':1,
        'original_context_layout':{'shape':[18,196,C],'dtype':'torch.bfloat16','live_stride':[196*C,C,1],'storage_offset':0},'cells':{}}
    for cell in ('N18','N6','O18','O6'):
        cut['cells'][cell]={'views':[{'shape':[6 if cell.endswith('6') else 18,196,C],'dtype':'torch.bfloat16',
            'live_stride':[196*C,C,1],'storage_offset':i*6*196*C if cell.endswith('6') else 0,'shares_live_storage':True} for i in range(3 if cell.endswith('6') else 1)]}
    I=torch.zeros((),dtype=torch.float64).expand(18,196,C)
    if bad=='missing_cut':raw.pop('cuts.O6.output')
    if bad=='forged_context_pair':
        raw['cuts.O18.input']=context+1;raw['cuts.O6.input']=context+1
    if bad=='missing_I':I=None
    if bad=='wrong_I':I=I+1
    call=lambda:bridge.validate_bundle(raw,engine='native',actual_state=actual,references=refs,cut_state=cut,old_context=context,
        old_reference=context,interaction_array=I)
    if bad=='none':
        proof=call();assert proof['tensors']==17 and proof['interaction_exact'] and proof['cuts']['native_control_and_restoration_exact']
    else:
        with pytest.raises(ValueError,match='bundle|raw|context|interaction|coverage|role'):call()


def r39_manifest(bridge,tmp_path):
    from test_hyvla_frozen_vit_prefix_example import input_manifest
    old_spec=importlib.util.spec_from_file_location('fixture_R38_prefix',ROOT/'examples/vla_autotune/hy_trace/frozen_vit_prefix.py')
    old=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old)
    inputs,native38=input_manifest(old,tmp_path)
    folder=tmp_path/'snapshots/round038/code/examples/vla_autotune/hy_trace';folder.mkdir(parents=True)
    for name in ('frozen_vit_prefix.py','merger_stages.py','proj1_camera_ablation.py','frozen_eager_attention.py'):
        shutil.copyfile(ROOT/'examples/vla_autotune/hy_trace'/name,folder/name)
    current=tmp_path/'snapshots/round039/code/flash_rt/models/hyvla/pipeline_thor.py';current.parent.mkdir(parents=True);shutil.copyfile(native38,current)
    prefix_file=folder/'frozen_vit_prefix.py';r38=tmp_path/'results/hy-vit-prefix-318174';r38.mkdir(parents=True)
    (r38/'SOURCE_GIT_COMMIT').write_text('311c66d06c20224dd0a11ce6108fb4cb2ac76c62\n');(r38/'inputs.json').write_text(json.dumps(inputs))
    sources={'driver':bridge.file_record(prefix_file),**inputs['source_files']}
    checkpoint=json.loads(Path(inputs['history_manifest']['path']).read_text())['checkpoint']
    for engine,runtime,git,count in (('official',{'torch':'old'},'a'*40,124),('native',{'torch':'new'},'b'*40,42)):
        f=r38/engine;f.mkdir();(f/'operator_outputs.safetensors').write_bytes(b'actual raw not read in metadata-only fixture')
        state={'complete':True,'cleanup':{'passed':True},'camera_block_indices':[[0,1,2,3]]*3} if engine=='official' else {
            'complete':True,'cleanup':{'passed':True},'control_block_indices':[0,1,2,3],'observed_block_indices':[0,1,2,3],
            'normal_vit_forward_return_reached':False,'block4_called':False}
        report={'kind':'hy_actual_frozen_vit_prefix','status':'prefix_capture_complete','engine':engine,'job_id':'318174',
            'source_revision':'311c66d06c20224dd0a11ce6108fb4cb2ac76c62','runtime':runtime,'torch_git_version':git,
            'bf16_reduced_precision_reduction':True,'input_manifest':inputs,'checkpoint':checkpoint,'source_records':sources,
            'full_model_constructed':False,'blocks_constructed':4,'weight_read_events':51,'observations':state,
            'raw_proof':{'tensors':count,'block_chain_exact':True,'residual_chains_exact':True,'actual_endpoint_bridge_exact':True},
            'comparison':{'first_different_stage':'block0.attention'} if engine=='native' else None}
        (f/'result.json').write_text(json.dumps(report))
    records={str(p.relative_to(r38)):bridge.file_record(p) for p in r38.rglob('*') if p.is_file()}
    return {'kind':'hy_actual_vit_attention_bridge_inputs','task_root':str(tmp_path),'R38_records':records,
        'frozen_prefix':bridge.file_record(prefix_file),'current_native':bridge.file_record(current)},current


@pytest.mark.parametrize('bad',['none','old_raw_stat','copied_namespace','helper_source','current_source','runtime','checkpoint','source_marker'])
def test_real_disk_R38_namespace_source_runtime_and_stats_precede_attention_weight_consumer(bridge,tmp_path,bad):
    manifest,current=r39_manifest(bridge,tmp_path);r38=tmp_path/'results/hy-vit-prefix-318174'
    if bad=='old_raw_stat':(r38/'official/operator_outputs.safetensors').write_bytes(b'changed')
    if bad=='copied_namespace':
        copied=tmp_path/'snapshots/round039/code/examples/vla_autotune/hy_trace/frozen_vit_prefix.py';copied.parent.mkdir(parents=True)
        shutil.copyfile(manifest['frozen_prefix']['path'],copied);manifest['frozen_prefix']=bridge.file_record(copied)
    if bad=='helper_source':
        p=Path(manifest['frozen_prefix']['path']);p.write_text('# changed protected driver');manifest['frozen_prefix']=bridge.file_record(p)
    if bad=='current_source':current.write_text('# different math');manifest['current_native']=bridge.file_record(current)
    if bad in ('runtime','checkpoint','source_marker'):
        if bad=='source_marker':(r38/'SOURCE_GIT_COMMIT').write_text('wrong source');manifest['R38_records']['SOURCE_GIT_COMMIT']=bridge.file_record(r38/'SOURCE_GIT_COMMIT')
        else:
            f=r38/'native/result.json';r=json.loads(f.read_text())
            if bad=='runtime':r['runtime']={'torch':'other'}
            else:r['checkpoint']['bytes']=18
            f.write_text(json.dumps(r));manifest['R38_records']['native/result.json']=bridge.file_record(f)
    consumed=[]
    def consumer(prefix,reports):
        # This call actually consumes __file__ from the loaded module's
        # namespace: an R39 copy would fail the unmodified R38 source guard.
        inputs=json.loads((r38/'inputs.json').read_text())
        derived=Path(prefix.__file__).resolve().parents[3]/'flash_rt/models/hyvla/pipeline_thor.py'
        prefix.admit_before_weights(inputs,native_source=derived,factory=lambda m:None)
        consumed.append('four-weight consumer');return prefix,reports
    call=lambda:bridge.admit_before_weights(manifest,engine='native',runtime={'torch':'new'},torch_git='b'*40,reduction_flag=True,
        current_native=current,current_prefix=ROOT/'examples/vla_autotune/hy_trace/frozen_vit_prefix.py',factory=consumer)
    if bad=='none':
        prefix,reports=call();assert consumed==['four-weight consumer'] and 'round038' in str(prefix.__file__)
        assert reports['official']['engine']=='official' and reports['native']['raw_proof']['tensors']==42
    else:
        with pytest.raises(ValueError,match='R38|source|record|path|namespace|runtime|checkpoint|prefix'):call()
        assert not consumed


def test_real_attention_factories_load_all_four_physical_parameters_without_other_modules(bridge):
    import torch.nn as nn
    class VendorAttention(nn.Module):
        def __init__(self,*,dim,num_heads,qkv_bias,qk_norm,attn_drop,proj_drop,seperate_qv_bias):
            super().__init__();assert (dim,num_heads,qkv_bias,qk_norm,attn_drop,proj_drop,seperate_qv_bias)==(1152,16,True,False,0.,0.,False)
            self.num_heads=num_heads;self.head_dim=72;self.scale=72**-.5;self.q_bias=None;self.q_norm=nn.Identity();self.k_norm=nn.Identity()
            self.qkv=nn.Linear(1152,3456);self.proj=nn.Linear(1152,1152)
    shapes={'qkv.weight':(3456,1152),'qkv.bias':(3456,),'proj.weight':(1152,1152),'proj.bias':(1152,)}
    weights={k:torch.full((),(i+1)/8,dtype=torch.bfloat16).expand(s) for i,(k,s) in enumerate(shapes.items())}
    old=bridge.make_official_attention(SimpleNamespace(_ViTAttention=VendorAttention),weights,device='cpu')
    native=bridge.make_native_owner(weights,source=NATIVE)
    assert set(old.state_dict())==set(weights) and all(torch.equal(v,weights[k]) and v.dtype==torch.bfloat16 for k,v in old.state_dict().items())
    assert all(not m.training for m in old.modules()) and native._vit_qkv_w_cur is weights['qkv.weight']
    assert native._vit_proj_b_cur is weights['proj.bias'] and native._vit_f8 is False


def test_help_and_allocation_guard_before_R38_or_weight_reads(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--official-bridge' in r.stdout and '--inputs' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--inputs','/missing','--source-revision','a'*40,'--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()


@pytest.mark.parametrize('bad',['none','job','trace_path','runtime','checkpoint','flag'])
def test_legacy_context_header_has_exact_source_input_and_runtime_binding(bridge,tmp_path,bad):
    checkpoint={'path':str(tmp_path/'weights/model.safetensors'),'bytes':17,'mtime_ns':29};runtime={'torch':'2.11.0+cu128','cuda':'12.8'}
    report={'kind':'diagnostic_frozen_block0_operators_not_recorded_block0_or_action_acceptance','engine':'native','job_id':'309409',
        'torch':'2.11.0+cu128','cuda':'12.8','gpu':'NVIDIA A800-SXM4-80GB','model_constructed':False,'performance_measurement':False,
        'official_trace':str(tmp_path/'results/hy-trace-308690/official/baseline'),'checkpoint':checkpoint['path'],
        'checkpoint_stat':{'bytes':17,'mtime_ns':29},'bf16_reduced_precision_reduction':True}
    if bad=='job':report['job_id']='309182'
    if bad=='trace_path':report['official_trace']=str(tmp_path/'results/other')
    if bad=='runtime':report['torch']='other'
    if bad=='checkpoint':report['checkpoint_stat']['bytes']+=1
    if bad=='flag':report['bf16_reduced_precision_reduction']=False
    call=lambda:bridge.admit_legacy_header(report,task=tmp_path,job='309409',engine='native',checkpoint=checkpoint,runtime=runtime,reduction_flag=True)
    if bad=='none':call()
    else:
        with pytest.raises(ValueError,match='legacy|Legacy|context|input|runtime|checkpoint'):call()


def test_cut_failure_retains_already_captured_API_fields_and_both_observation_states(bridge,tmp_path,monkeypatch):
    from safetensors.torch import load_file
    raw,actual,refs=actual_schema(bridge,'native');live=torch.zeros((),dtype=torch.bfloat16).expand(18,196,1152)
    monkeypatch.setattr(bridge,'capture_actual',lambda *a,**kw:(raw,actual,live))
    def broken(*a,**kw):
        error=RuntimeError('cut failed after API succeeded');error.bridge_partial=({'cuts.N18.input':live},{'complete':False,'cleanup':{'passed':True},'captured':'cut phase'})
        raise error
    monkeypatch.setattr(bridge,'projection_cuts',broken)
    with pytest.raises(RuntimeError,match='cut failed') as caught:
        bridge.capture_and_cut('native',None,None,{},references=refs,old_context=live,old_reference=refs['post'],synchronize=lambda:None)
    partial,state=caught.value.bridge_partial
    assert len(partial)==8 and 'api' in partial and 'cuts.N18.input' in partial
    assert state['actual']['complete'] and state['cuts']['complete'] is False
    bridge.write_failure(tmp_path,caught.value)
    assert len(load_file(str(tmp_path/'unqualified_partial.safetensors')))==8
    failed=json.loads((tmp_path/'failure.json').read_text());assert failed['qualified'] is False and failed['observations']['actual']['complete']
    assert not (tmp_path/'result.json').exists()


def test_external_actual_API_anchor_failure_blocks_all_cuts_and_retains_capture(bridge,monkeypatch):
    raw,actual,refs=actual_schema(bridge,'native');live=raw['context'];refs['post']=torch.ones_like(refs['post'])
    monkeypatch.setattr(bridge,'capture_actual',lambda *a,**kw:(raw,actual,live))
    called=[]
    def forbidden(*a,**kw):called.append(True);raise AssertionError('cuts must not run')
    monkeypatch.setattr(bridge,'projection_cuts',forbidden)
    with pytest.raises(ValueError,match='reference|anchor') as caught:
        bridge.capture_and_cut('native',None,None,{},references=refs,old_context=live,old_reference=refs['post'],synchronize=lambda:None)
    assert not called and len(caught.value.bridge_partial[0])==7


def test_failure_evidence_IO_does_not_replace_the_primary_error(bridge,tmp_path,monkeypatch):
    import safetensors.torch
    error=ValueError('primary anchor failure');error.bridge_partial=({'api':torch.zeros(1,dtype=torch.bfloat16)},{'actual':{'complete':False}})
    def fail(*a,**kw):raise OSError('evidence write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    bridge.write_failure(tmp_path,error)
    result=json.loads((tmp_path/'failure.json').read_text())
    assert result['qualified'] is False and result['error_type']=='ValueError' and result['error']=='primary anchor failure'
    assert result['evidence_write_errors'] and any('evidence write failed' in note for note in error.__notes__)


@pytest.mark.parametrize('bad',['none','job','source','runtime','keys','cleanup','anchor'])
def test_same_job_actual_old_attention_predecessor_is_required_before_native_weights(bridge,bad):
    raw,state,refs=actual_schema(bridge,'official');checkpoint={'path':'fixture','bytes':17}
    report={'kind':'hy_actual_vit_attention_context_projection_bridge','status':'bridge_complete','engine':'official','job_id':'9','source_revision':'c'*40,
        'runtime':{'torch':'old'},'torch_git_version':'a'*40,'bf16_reduced_precision_reduction':True,'checkpoint':checkpoint,
        'input_manifest':{'identity':'fixture'},'source_records':{'driver':{}},'weight_read_events':4,
        'selected_checkpoint_keys':['model.dual_tower.vlm.model.visual.vision_tower.blocks.0.attn.'+k for k in ('qkv.weight','qkv.bias','proj.weight','proj.bias')],
        'full_model_constructed':False,'attention_module_constructed':True,'observations':{'actual':state,'cuts':None},
        'raw_proof':{'tensors':21,'R38_norm1_qkv_exact':True,'API_control_projection_self_exact_R38':True}}
    if bad=='job':report['job_id']='8'
    if bad=='source':report['source_revision']='d'*40
    if bad=='runtime':report['runtime']={'torch':'other'}
    if bad=='keys':report['selected_checkpoint_keys'][-1]='other.weight'
    if bad=='cleanup':state['cleanup']['passed']=False
    if bad=='anchor':refs['post']=(torch.ones_like(refs['post'][0]),*refs['post'][1:])
    called=[]
    call=lambda:bridge.admit_official_bridge(report,raw,job_id='9',source_revision='c'*40,runtime={'torch':'old'},torch_git='a'*40,
        reduction_flag=True,checkpoint=checkpoint,inputs={'identity':'fixture'},source_records={'driver':{}},references=refs,factory=lambda:called.append('native four-weight reader'))
    if bad=='none':call();assert called==['native four-weight reader']
    else:
        with pytest.raises(ValueError,match='bridge|capture|reference|anchor|cleanup|source|runtime|predecessor'):call()
        assert not called


def test_raw_publication_failure_never_creates_completed_bridge_report(bridge,tmp_path,monkeypatch):
    import safetensors.torch
    raw,state,refs=actual_schema(bridge,'official');report={'engine':'official','observations':{'actual':state,'cuts':None}}
    def broken(*a,**kw):raise OSError('bridge raw IO failed')
    monkeypatch.setattr(safetensors.torch,'save_file',broken)
    with pytest.raises(OSError,match='raw IO'):bridge.publish_capture(tmp_path,raw,report,references=refs)
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('bad',['none','missing_layout','offset'])
def test_norm1_view_restoration_requires_all_three_actual_camera_layouts(bridge,bad):
    values=tuple(torch.full((6,196,1152),i/8,dtype=torch.bfloat16) for i in range(3))
    layouts=tuple({'shape':[6,196,1152],'dtype':'torch.bfloat16','live_stride':[225792,1152,1],'storage_offset':0} for _ in range(3))
    if bad=='missing_layout':layouts=layouts[:1]
    if bad=='offset':layouts[1]['storage_offset']=4
    call=lambda:bridge.restore_norm1(values,layouts,engine='official',device='cpu')
    if bad=='none':assert all(torch.equal(x,y) for x,y in zip(call(),values))
    else:
        with pytest.raises(ValueError,match='layout|offset|camera|coverage'):call()


@pytest.mark.parametrize('registration',['context_pre','projection_post'])
def test_partial_official_hook_installation_is_transactional_and_keeps_original_error(bridge,monkeypatch,registration):
    owner,inputs=tiny_owner(bridge,'official');owner._attention_bridge_active=False
    existing_qkv=owner.qkv.register_forward_hook(lambda m,a,out:None)
    existing_proj=owner.proj.register_forward_pre_hook(lambda m,a:None)
    before={id(m):(set(m._forward_hooks),set(m._forward_pre_hooks)) for m in owner.modules()}
    expected=owner(inputs[0]).detach().clone();injected=RuntimeError('partial hook installation failed')
    method='register_forward_pre_hook' if registration=='context_pre' else 'register_forward_hook'
    def fail(*a,**kw):raise injected
    monkeypatch.setattr(owner.proj,method,fail)
    with pytest.raises(RuntimeError,match='partial hook') as caught:bridge.capture_actual('official',owner,inputs,synchronize=lambda:None)
    assert caught.value is injected
    raw,state=caught.value.bridge_partial;assert not state['complete'] and state['cleanup']['passed'] is True
    assert owner._attention_bridge_active is False
    assert all((set(m._forward_hooks),set(m._forward_pre_hooks))==before[id(m)] for m in owner.modules())
    keys=set(raw)
    for _ in range(2):assert torch.equal(owner(inputs[0]),expected)
    assert set(raw)==keys  # No old observer closure may consume later forwards.
    existing_qkv.remove();existing_proj.remove()
