import importlib.util
from pathlib import Path
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
