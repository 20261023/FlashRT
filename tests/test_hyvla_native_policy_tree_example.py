import importlib.util
import copy
from pathlib import Path
import subprocess
import os
import sys

import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/native_policy_tree.py'


@pytest.fixture
def driver():
    s=importlib.util.spec_from_file_location('hy_native_policy_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def request_fixture(driver):
    tensors={name:torch.zeros(1) for name in driver.CONDITIONS}
    tensors['actions']=torch.ones(1,40,32)
    for i in range(10):
        tensors[f'step{i}.latent_before']=torch.zeros(1,40,32)
        tensors[f'step{i}.time_embedding']=torch.ones(1,1024,dtype=torch.bfloat16)
        tensors[f'step{i}.velocity']=torch.zeros(1,40,32,dtype=torch.bfloat16)
    return {'tensors':tensors,'sampler_metadata':{'sampler':'euler','scheduled_steps':10,'nfe':10,
        'timesteps':[1.+i*(-1./10) for i in range(10)],'numerical_profile':'official_local_fp32'},
        'backend_metadata':{'model':'hyvla','precision':'bf16','action_domain':'normalized-padded-model-actions'},
        'scope':{'complete':True,'cleanup':{'passed':True},'nfe':10}}


def clone_request(r):
    return {**r,'tensors':{k:v.clone() for k,v in r['tensors'].items()},'sampler_metadata':dict(r['sampler_metadata'])}


@pytest.mark.parametrize('bad',['none','value','dtype','metadata'])
def test_current_control_two_replays_require_all_step_arrays_and_metadata_equal(driver,bad):
    a=request_fixture(driver);b=clone_request(a)
    if bad=='value':b['tensors']['step9.velocity']+=1
    if bad=='dtype':b['tensors']['step0.velocity']=b['tensors']['step0.velocity'].float()
    if bad=='metadata':b['sampler_metadata']['nfe']=9
    if bad=='none':assert driver.admit_replay(a,b)['all_current_raw_exact'] is True
    else:
        with pytest.raises(ValueError,match='replay|metadata'):driver.admit_replay(a,b)


def test_control_failure_stops_before_any_candidate_request(driver):
    calls=[]
    def execute(arm,case,index):calls.append((arm,case,index));return request_fixture(driver)
    def control(case,pair):
        if case=='state':raise ValueError('control old action mismatch')
        return {'historical_actions_exact':True,'official_oracle':{'same_action_gate_passed':case!='state'}}
    with pytest.raises(ValueError,match='control'):
        driver.run_matrix(execute,control,lambda c,p,b:{'gate':False})
    assert calls and all(arm=='control' for arm,case,index in calls)


def test_all_seven_controls_precede_candidate_and_negative_gates_are_retained(driver):
    calls=[]
    def execute(arm,case,index):calls.append((arm,case,index));return request_fixture(driver)
    matrix=driver.run_matrix(execute,lambda c,p:{'historical_actions_exact':True,'official_oracle':{'same_action_gate_passed':c!='state'}},lambda c,p,b:{'same_action_gate_passed':False})
    assert len(calls)==28 and all(x[0]=='control' for x in calls[:14]) and all(x[0]=='candidate' for x in calls[14:])
    assert set(matrix['control'])==set(matrix['candidate'])==set(driver.VARIANTS)
    assert all(not row['qualification']['same_action_gate_passed'] for row in matrix['candidate'].values())


def test_control_gate_disposition_barrier_is_before_any_candidate(driver):
    calls=[]
    def execute(arm,case,index):calls.append((arm,case,index));return request_fixture(driver)
    with pytest.raises(ValueError,match='control|6of7'):
        driver.run_matrix(execute,lambda c,p:{'historical_actions_exact':True,'official_oracle':{'same_action_gate_passed':True}},lambda c,p,b:{'same_action_gate_passed':False})
    assert len(calls)==14 and all(x[0]=='control' for x in calls)


@pytest.mark.parametrize('case',['episode_start','current','history','prompt','noise'])
def test_five_missing_historical_step_traces_are_not_forged(driver,case):
    evidence=driver.legacy_step_check(case,request_fixture(driver)['tensors'],None)
    assert evidence=={'historical_step_trace_available':False,'scope':'current_control_new_capture_two_replay_only'}
    with pytest.raises(ValueError,match='historical|legacy'):
        driver.legacy_step_check(case,request_fixture(driver)['tensors'],{})


def test_baseline_state_legacy_raw_steps_are_required_and_compared(driver):
    raw=request_fixture(driver)['tensors']
    with pytest.raises(ValueError,match='legacy|historical'):driver.legacy_step_check('state',raw,None)
    old={k:v.clone() for k,v in raw.items()};assert driver.legacy_step_check('state',raw,old)['historical_step_trace_available']
    old['step8.velocity']+=1
    with pytest.raises(ValueError,match='legacy|historical'):driver.legacy_step_check('state',raw,old)


def legacy_header_fixture():
    source='2959bd8710038c7dcdebdd17ac645b7feed4333f'
    file={'path':'/checkpoint/model.safetensors','bytes':42,'mtime_ns':7}
    directory={'path':'/checkpoint','weights_size_bytes':42,'weights_mtime_ns':7,
        'config':{'path':'/checkpoint/config.json','size_bytes':8,'mtime_ns':9},
        'camera_order':['top_head','hand_left','hand_right'],'profile_fields':{'num_steps':10},'identity_note':'stat only'}
    runtime={'torch':'2.11.0+cu128'}
    header={'engine':'native','variant':'baseline','job_id':'313766','artifact_complete':True,
        'execution_mode':'diagnostic_eager','diagnostic_cleanup':{'passed':True},
        'config':{'use_cuda_graph':False,'numerical_profile':'official_local_fp32'},
        'experimental_projection':{'fold_spatial':True,'effective_projection':'per_camera_hwd','source_revision':source,
            'precondition':{'job_id':'313766','source_revision':source,'checkpoint':copy.deepcopy(file),'operator_exact':True}},
        'historical_official_reference':{'job_id':'308690'},'prior_graph_output':{'exact':True},'suffix_order_control':None,
        'frozen_control_evidence':{'native_replay_exact':True,'prefix_preserved':True},
        'checkpoint':copy.deepcopy(directory),'runtime':dict(runtime)}
    return header,directory,file,runtime


@pytest.mark.parametrize('bad',['none','bytes','mtime','directory','config','runtime'])
def test_real_legacy_two_checkpoint_schemas_are_admitted_before_factory(driver,bad):
    header,directory,file,runtime=legacy_header_fixture();called=[]
    if bad=='bytes':header['experimental_projection']['precondition']['checkpoint']['bytes']+=1
    if bad=='mtime':header['experimental_projection']['precondition']['checkpoint']['mtime_ns']+=1
    if bad=='directory':header['checkpoint']['path']='/other'
    if bad=='config':header['checkpoint']['config']['mtime_ns']+=1
    if bad=='runtime':header['runtime']['torch']='other'
    def factory():called.append(True);return 'legacy trace reader'
    call=lambda:driver.admit_legacy_header(header,variant='baseline',checkpoint=directory,weight_stat=file,runtime=runtime,factory=factory)
    if bad=='none':assert call()=='legacy trace reader' and called==[True]
    else:
        with pytest.raises(ValueError,match='checkpoint|baseline|legacy|runtime'):call()
        assert not called


def test_original_padded_gate_cannot_be_rescued_by_public_slice(driver):
    control=request_fixture(driver);candidate=clone_request(control)
    candidate['tensors']['actions'][0,0,31]=1.04
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    q=driver.qualify_candidate(candidate,control,arrays)
    assert q['same_action_gate_passed'] is False and q['public20_gate_passed'] is True
    assert q['gate']=={'atol':.01,'rtol':.01,'minimum_cosine':.999,'legacy_gate_domain':'normalized_padded_40x32'}
    assert q['actions32']['outside_tolerance']==1


def test_candidate_cannot_change_native_prefill_or_conditions(driver):
    control=request_fixture(driver);candidate=clone_request(control);candidate['tensors']['prefix.k']+=1
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    with pytest.raises(ValueError,match='prefix|condition'):driver.qualify_candidate(candidate,control,arrays)


@pytest.mark.parametrize('changed_control',[False,True])
def test_aggregate_preserves_old_six_of_seven_and_accepts_negative_candidate(driver,changed_control):
    matrix={'control':{},'candidate':{}}
    for case in driver.VARIANTS:
        passed=case!='state' or changed_control
        matrix['control'][case]={'qualification':{'official_oracle':{'same_action_gate_passed':passed}}}
        matrix['candidate'][case]={'qualification':{'same_action_gate_passed':False}}
    if changed_control:
        with pytest.raises(ValueError,match='control|6of7'):driver.gate_counts(matrix)
    else:assert driver.gate_counts(matrix)=={'control':6,'candidate':0}


def support_fixture(driver):
    shapes=driver.order.expected_shapes();h=torch.zeros(1,41,1024,dtype=torch.bfloat16);v=torch.zeros(1,40,32,dtype=torch.bfloat16)
    one={k:h if shape[-1]==1024 else v for k,shape in shapes.items()};raw={};refs={}
    for case in ('baseline','state'):
        refs[case]={'native':dict(one),'old':dict(one)}
        for arm in ('control','tree'):raw.update({arm+'.'+case+'.'+k:t for k,t in one.items()})
    r={'kind':'hy_live_source_tree_RMS_full_Expert_replay','status':'diagnostic_complete','job_id':'316722',
        'source_revision':'d8cdfec96aac64e31454c4c65cac5f05cd931706','raw_tensors':5200,'golden_norm_data_used':False,
        'model_constructed':False,'production_norm_replaced':False}
    return r,raw,refs


@pytest.mark.parametrize('bad',['none','job','dummy','tree','control'])
def test_actual_r32_all_raw_support_before_model_not_report_counts(driver,bad):
    r,raw,refs=support_fixture(driver);called=[]
    if bad=='job':r['job_id']='316721'
    if bad=='dummy':del raw['tree.baseline.gold_head.step9.velocity'];raw['dummy']=torch.zeros(1)
    if bad in ('tree','control'):raw[bad+'.state.propagated.step9.velocity']=torch.ones(1,40,32,dtype=torch.bfloat16)
    def factory():called.append(True);return 'context'
    call=lambda:driver.admit_full_support(r,raw,refs,factory=factory)
    if bad=='none':assert call()=='context' and called==[True]
    else:
        with pytest.raises(ValueError,match='support|raw|R32|control|tensor'):call()
        assert not called


def test_no_completed_report_before_backend_close_and_all_requests_clean(driver,tmp_path):
    requests=[request_fixture(driver) for _ in range(28)];path=tmp_path/'result.json'
    state={'execution':'diagnostic_eager','cleanup':{'passed':False}}
    with pytest.raises(ValueError,match='cleanup|backend'):driver.publish_completed(path,{'kind':'test'},state,requests)
    assert not path.exists()
    state['cleanup']['passed']=True;requests[-1]['scope']['complete']=False
    with pytest.raises(ValueError,match='request|complete'):driver.publish_completed(path,{'kind':'test'},state,requests)
    assert not path.exists()


def test_help_and_allocation_guard_before_model_or_input_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--preflight-report' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--checkpoint','/missing','--fixtures','/missing','--prior','/missing',
        '--support-root','/missing','--preflight-report','/missing','--source-revision','a'*40,'--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()
