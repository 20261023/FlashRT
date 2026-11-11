import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def controls():
    path=ROOT/'examples/vla_autotune/hy_trace/native_controls.py'
    if not path.exists():pytest.fail('native frozen-control helpers are missing')
    spec=importlib.util.spec_from_file_location('hy_native_controls',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_reference_replay_characterizes_bf16_scalar_before_fp32_latent_update(controls):
    x=torch.tensor([[[1.,-1.]]],dtype=torch.float32);v=torch.tensor([[[1.0078125,3.]]],dtype=torch.bfloat16)
    result=controls.reference_euler_step(x,v,num_steps=10)
    torch.testing.assert_close(result,torch.tensor([[[.89892578125,-1.30078125]]]),atol=0,rtol=0)
    assert not torch.equal(result,x+v*(-.1))
    torch.testing.assert_close(x,torch.tensor([[[1.,-1.]]]),atol=0,rtol=0)


def reference():
    return {'step0.latent_before':torch.zeros(1,2,1),'step1.latent_before':torch.full((1,2,1),-1.5),
        'step0.velocity':torch.full((1,2,1),3.,dtype=torch.bfloat16),
        'step1.velocity':torch.full((1,2,1),3.,dtype=torch.bfloat16),
        'step0.time':torch.tensor([1.]),'step1.time':torch.tensor([.5]),
        'actions':torch.full((1,2,1),-3.),'prefix.k':torch.full((1,1,1,1,1),3.),
        'prefix.v':torch.ones(1,1,1,1,1,dtype=torch.bfloat16)}


def test_recorded_euler_admission_requires_every_observed_step_and_final_action(controls):
    assert controls.verify_reference_integration(reference(),num_steps=2)['exact'] is True
    for field in ('step1.latent_before','actions','step1.time'):
        wrong=reference();wrong[field]=wrong[field]+.01
        with pytest.raises(ValueError,match='recorded|time|integration'):
            controls.verify_reference_integration(wrong,num_steps=2)


@pytest.mark.parametrize('bad',['latent_dtype','velocity_dtype','nonfinite','shape','steps'])
def test_replay_refuses_implicit_dtype_or_geometry_changes(controls,bad):
    x=torch.zeros(1,2,1);v=torch.ones(1,2,1,dtype=torch.bfloat16);steps=10
    if bad=='latent_dtype':x=x.bfloat16()
    if bad=='velocity_dtype':v=v.float()
    if bad=='nonfinite':v.fill_(float('nan'))
    if bad=='shape':v=v[:,:1]
    if bad=='steps':steps=0
    with pytest.raises(ValueError):controls.reference_euler_step(x,v,num_steps=steps)


def live_args():
    return (torch.zeros(1,2,1),torch.ones(1,1,dtype=torch.bfloat16),
        torch.zeros(1,1,1,dtype=torch.bfloat16),torch.ones(1,1,3,4,dtype=torch.bool),
        torch.ones(1,3,1),torch.zeros(1,3,1),torch.full((1,1,1,4,1),2.),
        torch.ones(1,1,1,4,1,dtype=torch.bfloat16),1)


def test_snapshots_do_not_alias_live_state_rope_or_prefix_and_zero_suffix(controls):
    args=live_args();snapshot=controls.snapshot_velocity_inputs(args)
    args[2].fill_(8);args[4].fill_(8);args[6].fill_(8)
    assert torch.equal(snapshot['state_emb'],torch.zeros(1,1,1,dtype=torch.bfloat16))
    assert torch.equal(snapshot['scos'],torch.ones(1,3,1))
    assert torch.equal(snapshot['kbuf'][...,:1,:],torch.full((1,1,1,1,1),2.))
    assert torch.count_nonzero(snapshot['kbuf'][...,1:,:])==0


def native_trace():
    return {**reference(),'step0.latent_before':torch.zeros(1,2,1),
            'step1.latent_before':torch.full((1,2,1),-1.),
            'step0.time_embedding':torch.ones(1,1,dtype=torch.bfloat16),
            'step1.time_embedding':torch.full((1,1),.5,dtype=torch.bfloat16),
            'step0.velocity':torch.full((1,2,1),2.,dtype=torch.bfloat16),
            'step1.velocity':torch.full((1,2,1),2.,dtype=torch.bfloat16)}


def common():
    path=ROOT/'examples/vla_autotune/hy_trace/diagnose.py'
    spec=importlib.util.spec_from_file_location('hy_controls_common',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_native_replay_then_teacher_forced_kv_arms_use_independent_scratch(controls):
    snapshot=controls.snapshot_velocity_inputs(live_args());snapshot['perm']=np.array([0])
    calls=[]
    def evaluate(x,t,state,mask,cos,sin,k,v,prefix):
        calls.append((float(k[...,0,:].item()),k.data_ptr(),v.data_ptr(),x.clone(),t.clone()))
        assert torch.count_nonzero(k[...,prefix:,:])==0 or torch.all(k[...,prefix:,:]==1)
        k[...,prefix:,:].fill_(1)
        return torch.full_like(x,float(k[...,0,:].item()),dtype=torch.bfloat16)
    frontend=SimpleNamespace(device=torch.device('cpu'),n_heads=1,
        _sinusoidal_time=lambda time:torch.tensor([[time]],dtype=torch.bfloat16))
    outputs,rows,evidence=controls.evaluate_controls(frontend,SimpleNamespace(evaluate_velocity=evaluate),
        reference(),native_trace(),snapshot,common=common(),num_steps=2)
    assert evidence['native_replay_exact'] and evidence['prefix_preserved']
    assert [x[0] for x in calls]==[2.,2.,2.,3.,2.,3.]
    assert len({calls[i][1] for i in (0,2,3)})==3
    assert torch.equal(calls[2][3],calls[3][3]) and torch.equal(calls[2][4],calls[3][4])
    assert all(row['exact'] for row in rows if row['condition']=='official_kv_sorted')
    assert torch.equal(outputs['native_kv.step1.velocity'],torch.full((1,2,1),2.,dtype=torch.bfloat16))
    assert torch.equal(snapshot['kbuf'][...,1:,:],torch.zeros(1,1,1,3,1))


def test_native_replay_mismatch_stops_before_interpreting_official_counterfactual(controls):
    snapshot=controls.snapshot_velocity_inputs(live_args());snapshot['perm']=np.array([0]);calls=[]
    def wrong(*args):calls.append(1);return torch.full((1,2,1),4.,dtype=torch.bfloat16)
    frontend=SimpleNamespace(device=torch.device('cpu'),n_heads=1,_sinusoidal_time=lambda t:torch.ones(1,1))
    with pytest.raises(ValueError,match='replay'):
        controls.evaluate_controls(frontend,SimpleNamespace(evaluate_velocity=wrong),reference(),native_trace(),
            snapshot,common=common(),num_steps=2)
    assert len(calls)==1


def test_evaluator_prefix_mutation_is_fatal_not_a_small_numeric_difference(controls):
    snapshot=controls.snapshot_velocity_inputs(live_args());snapshot['perm']=np.array([0])
    def wrong(x,t,state,mask,cos,sin,k,v,prefix):
        k[...,0,:].fill_(9);return torch.full_like(x,2.,dtype=torch.bfloat16)
    frontend=SimpleNamespace(device=torch.device('cpu'),n_heads=1,_sinusoidal_time=lambda t:torch.ones(1,1))
    with pytest.raises(ValueError,match='prefix'):
        controls.evaluate_controls(frontend,SimpleNamespace(evaluate_velocity=wrong),reference(),native_trace(),
            snapshot,common=common(),num_steps=2)


@pytest.mark.parametrize('position',[0,1,2,3,4,5])
def test_counterfactual_evaluator_cannot_mutate_shared_latent_time_or_conditions(controls,position):
    snapshot=controls.snapshot_velocity_inputs(live_args());snapshot['perm']=np.array([0])
    def wrong(*args):
        value=args[position]
        if value.dtype==torch.bool:value.logical_not_()
        else:value.add_(1)
        return torch.full((1,2,1),2.,dtype=torch.bfloat16)
    frontend=SimpleNamespace(device=torch.device('cpu'),n_heads=1,_sinusoidal_time=lambda t:torch.ones(1,1))
    with pytest.raises(ValueError,match='mutat|read.only|condition'):
        controls.evaluate_controls(frontend,SimpleNamespace(evaluate_velocity=wrong),reference(),native_trace(),
            snapshot,common=common(),num_steps=2)


@pytest.mark.parametrize('bad',['none','job','fold','checkpoint','runtime','noise','oracle','replay','graph'])
def test_candidate_prior_admission_binds_actual_folded_case_and_input_evidence(controls,bad):
    from test_hyvla_proj1_ablation_example import completed_record,saved_arrays
    record=completed_record(True,'camera_proj1')
    record['experimental_merger_proj1'].update(fold_spatial=True,effective_projection='per_camera_hwd')
    record['experimental_merger_proj1']['precondition']['fold_spatial']=True
    saved=saved_arrays();saved.update(replay_actions32=np.zeros((2,40,32),np.float32),
        graph_actions32=np.zeros((40,32),np.float32),eager_actions32=np.zeros((40,32),np.float32))
    data={'noise':saved['noise'].copy(),'state':saved['input_state'].copy(),
        'prompt':saved['input_prompt'].copy(),'expected_actions32':saved['oracle_actions32'].copy()}
    checkpoint=record['checkpoint'].copy();runtime=record['runtime'].copy();fixture=record['fixture'].copy()
    if bad=='job':record['job_id']='8'
    if bad=='fold':record['experimental_merger_proj1']['fold_spatial']=False
    if bad=='checkpoint':checkpoint={**checkpoint,'weights_mtime_ns':999}
    if bad=='runtime':runtime={**runtime,'torch':'wrong'}
    if bad=='noise':saved['noise'][0,0]=7
    if bad=='oracle':saved['oracle_actions32'][0,0]=7
    if bad=='replay':saved['replay_actions32'][1,0,0]=7
    if bad=='graph':saved['eager_actions32'][0,0]=7
    call=lambda:controls.admit_candidate_prior(record,saved,data,variant='baseline',expected_job='7',
        checkpoint=checkpoint,runtime=runtime,fixture=fixture)
    if bad=='none':assert call()['admitted'] is True
    else:
        with pytest.raises(ValueError,match='prior|input|fold|identity'):call()


@pytest.mark.parametrize('field,value',[('engine','native'),('job_id','wrong'),('variant','noise')])
def test_historical_reference_identity_cannot_be_forged_as_current_or_other_case(controls,field,value):
    report={'engine':'official','job_id':'308690','variant':'state','oracle':{'exact':True}}
    controls.admit_historical_reference(report,expected_job='308690',variant='state')
    report[field]=value
    with pytest.raises(ValueError,match='historical|reference'):
        controls.admit_historical_reference(report,expected_job='308690',variant='state')


def test_final_trace_publication_requires_successful_eager_cleanup(controls):
    pending=[{'report':{'variant':'state'}}];published=[]
    def save(item):published.append(item['report'].copy())
    with pytest.raises(ValueError,match='cleanup'):
        controls.publish_after_cleanup(pending,{'cleanup':{'passed':False}},save)
    assert published==[]
    state={'execution':'diagnostic_eager','cleanup':{'passed':True,'owned_graphs_reset':0}}
    controls.publish_after_cleanup(pending,state,save)
    assert published[0]['diagnostic_cleanup']['passed'] is True
    assert published[0]['execution_mode']=='diagnostic_eager'
    assert published[0]['performance_measurement'] is False
