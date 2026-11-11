import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def layers():
    p=ROOT/'examples/vla_autotune/hy_trace/expert_layers.py'
    s=importlib.util.spec_from_file_location('hy_core_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def tiny_data():
    bf=torch.bfloat16;z=lambda *shape:torch.zeros(*shape,dtype=bf)
    w={'model.state_proj.weight':torch.eye(2,dtype=bf),'model.state_proj.bias':z(2),
       'model.action_in_proj.weight':z(2,1),'model.action_in_proj.bias':z(2),
       'model.action_time_mlp_in.weight':z(2,4),'model.action_time_mlp_in.bias':z(2),
       'model.action_time_mlp_out.weight':z(2,2),'model.action_time_mlp_out.bias':torch.ones(2,dtype=bf),
       'model.dual_tower.expert.model.norm.weight':torch.ones(2,dtype=bf),
       'model.action_out_proj.weight':torch.tensor([[1.,2.]],dtype=bf),'model.action_out_proj.bias':torch.tensor([.5],dtype=bf)}
    for i in range(2):
        e=f'model.dual_tower.expert.model.layers.{i}';v=f'model.dual_tower.vlm.model.language_model.model.layers.{i}.self_attn'
        for name,shape in (('q',(4,2)),('k',(2,2)),('v',(2,2)),('o',(2,4))):w[e+f'.self_attn.{name}_proj_v.weight']=z(*shape)
        for name in ('gate','up','down'):w[e+f'.mlp_v.{name}_proj.weight']=torch.eye(2,dtype=bf)
        for name in ('input_layernorm_v','post_attention_layernorm_v'):w[e+'.'+name+'.weight']=torch.ones(2,dtype=bf)
        for name in ('query','key'):w[v+'.'+name+'_layernorm.weight']=torch.ones(2,dtype=bf)
    gold={'prefix.k':torch.zeros(2,1,1,1,2),'prefix.v':z(2,1,1,1,2)}
    trace={'prepared.state':torch.ones(1,2,dtype=bf),'suffix.state_embedding':torch.ones(1,1,2,dtype=bf),
           'suffix.mask_sorted':torch.ones(1,1,3,4,dtype=torch.bool),'suffix.cos':torch.ones(1,1,3,2),'suffix.sin':torch.zeros(1,1,3,2)}
    frozen={}
    for step in range(2):
        latent=torch.zeros(1,2,1);time=z(1,2)
        gold[f'step{step}.latent_before']=latent;gold[f'step{step}.velocity']=torch.full((1,2,1),3.5,dtype=bf)
        gold[f'expert.step{step}.final_input']=torch.full((1,3,2),2.46875,dtype=bf)
        for i,value in ((0,1.),(1,1.734375)):
            base=f'expert.step{step}.layer{i}.'
            gold[base+'input']=torch.full((3,2),value,dtype=bf)
            gold[base+'residual']=torch.full((3,2),value,dtype=bf)
            gold[base+'mlp_output']=torch.full((3,2),.73046875,dtype=bf)
        trace[f'step{step}.time_embedding']=time
        frozen[f'frozen_inputs.step{step}.latent']=latent
        frozen[f'frozen_inputs.step{step}.time_embedding']=time
        frozen[f'official_kv_sorted.step{step}.velocity']=torch.full((1,2,1),3.5,dtype=bf)
    return w,{'tiny':{'gold':gold,'trace':trace,'frozen':frozen,'perm':np.array([0],dtype=np.int64)}}


def test_shared_core_actual_two_layer_math_and_fresh_state(layers):
    weights,cases=tiny_data();owner=layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)
    first=layers.replay_core(owner,cases,weights.__getitem__,num_layers=2,num_steps=2,device='cpu')
    second=layers.replay_core(owner,cases,weights.__getitem__,num_layers=2,num_steps=2,device='cpu')
    assert len(first['selected_checkpoint_keys'])==33 and set(first['selected_checkpoint_keys'])==set(weights)
    a,b=first['cases']['tiny'],second['cases']['tiny']
    assert a['h0_exact']==b['h0_exact']==2 and a['attention_calls']==b['attention_calls']==8
    assert len(a['outputs'])==20 and 'outputs' not in cases['tiny'] and 'propagated' not in cases['tiny']
    torch.testing.assert_close(a['outputs']['propagated.step1.layer1.output'],torch.full((1,3,2),2.46875,dtype=torch.bfloat16),atol=0,rtol=0)
    torch.testing.assert_close(a['outputs']['propagated.step1.velocity'],torch.full((1,2,1),3.5,dtype=torch.bfloat16),atol=0,rtol=0)
    assert all(r['exact'] for r in a['metrics'].values())
    for key in a['outputs']:torch.testing.assert_close(a['outputs'][key],b['outputs'][key],atol=0,rtol=0)


def test_shared_core_rejects_weight_dtype_without_changing_admitted_inputs(layers):
    weights,cases=tiny_data();weights['model.action_in_proj.weight']=weights['model.action_in_proj.weight'].float()
    owner=layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)
    with pytest.raises(ValueError,match='weight|BF16'):layers.replay_core(owner,cases,weights.__getitem__,num_layers=2,num_steps=2,device='cpu')
    assert 'outputs' not in cases['tiny']


def test_original_order_scope_reorders_actual_nonzero_attention_without_changing_q(layers):
    owner=layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)
    q=torch.zeros(1,2,2,2);q[:,0,:,0]=math.sqrt(2)
    k=torch.tensor([[[[0.,0.],[math.log(3),0.],[10.,0.],[0.,0.],[-10.,0.]]]]).repeat(1,2,1,1)
    v=torch.tensor([[[[2.,0.],[6.,0.],[100.,0.],[2.,0.],[-100.,0.]]]],dtype=torch.bfloat16).repeat(1,2,1,1)
    mask=torch.tensor([[[[True,True,False,False,False],[False,True,False,True,False]]]])
    columns=torch.tensor([2,0,1,3,4]);seen=[];real=owner._attn
    def observe(q,k,v,mask):seen.append((k.clone(),mask.clone()));return real(q,k,v,mask)
    owner._attn=observe;state={};before=q.clone()
    with layers.attention_order_scope(owner,np.array([2,0,1]),suffix_rows=2,order='original',device='cpu',state=state):
        actual=owner._attn(q,k.index_select(2,columns),v.index_select(2,columns),mask.index_select(-1,columns))
    expected=torch.tensor([[[[5.,0.],[5.,0.]],[[4.,0.],[4.,0.]]]],dtype=torch.bfloat16)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(seen[0][0],k,atol=0,rtol=0);assert torch.equal(seen[0][1],mask)
    assert owner._attn is observe and torch.equal(q,before) and state['cleanup']['passed']
    assert state['attention_calls']==1 and state['query_layouts'][0]['live_stride']==list(q.stride())


def test_order_scope_sync_failure_preserves_primary_and_restores_binding(layers):
    owner=layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)
    original=owner._attn;state={}
    def fail():raise RuntimeError('sync failed')
    with pytest.raises(RuntimeError,match='primary failed') as err:
        with layers.attention_order_scope(owner,np.array([0]),suffix_rows=2,order='original',device='cpu',state=state,synchronize=fail):
            raise RuntimeError('primary failed')
    assert owner._attn is original and state['_failed_owners'] and not state['cleanup']['passed']
    assert any('sync failed' in note for note in err.value.__notes__)
