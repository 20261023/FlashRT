import importlib.util
import copy
from contextlib import contextmanager
import json
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


def disk_policy_inputs(driver,tmp_path):
    from test_hyvla_proj1_ablation_example import completed_record,saved_arrays
    from safetensors.torch import save_file
    _,checkpoint,weight_stat,_=legacy_header_fixture()
    fixtures=tmp_path/'fixtures';fixtures.mkdir();(fixtures/'manifest.json').write_text(json.dumps({'job_id':'1','source_revision':'a'*40}))
    prior=tmp_path/'results/hy-proj1-fold-309681/camera_proj1'
    runtime=completed_record()['runtime']
    for case in driver.VARIANTS:
        saved=saved_arrays();saved['actions32']=saved['actions32'].copy()
        if case=='state':saved['actions32'][0,0]=.04
        saved['actions20']=saved['actions32'][:,:20].copy()
        saved.update(replay_actions32=np.stack([saved['actions32']]*2),graph_actions32=saved['actions32'].copy(),eager_actions32=saved['actions32'].copy())
        arrays={'noise':saved['noise'],'state':saved['input_state'],'prompt':saved['input_prompt'],
            'expected_actions32':saved['oracle_actions32'],'expected_actions20':saved['oracle_actions20']}
        np.savez(fixtures/(case+'.npz'),**arrays)
        record=completed_record(case!='state',arm='camera_proj1',variant=case)
        record.update(job_id='309681',checkpoint=checkpoint,runtime=runtime,fixture=driver.validator.fixture_record(fixtures/(case+'.npz')))
        record['official_oracle']=driver.validator.oracle_comparison(saved['actions32'],arrays,sampler='euler',num_steps=10)
        record['experimental_merger_proj1'].update(fold_spatial=True,effective_projection='per_camera_hwd')
        record['experimental_merger_proj1']['precondition'].update(job_id='309681',fold_spatial=True)
        folder=prior/case;folder.mkdir(parents=True)
        (folder/'result.json').write_text(json.dumps(record));(folder/'child_exit.json').write_text(json.dumps({'returncode':3 if case=='state' else 0}))
        np.savez(folder/'actions.npz',**saved)
        if case in ('baseline','state'):
            header,_,_,_=legacy_header_fixture();header.update(variant=case,checkpoint=checkpoint,runtime=runtime)
            old=tmp_path/'results/hy-folded-state-trace-313766/native'/case;old.mkdir(parents=True)
            (old/'result.json').write_text(json.dumps(header))
            # This helper transports the legacy payload; its step-role admission
            # is exercised separately by legacy_step_check, not invented here.
            save_file({'legacy.transport_sentinel':torch.tensor([3.])},str(old/'trace.safetensors'))
    return fixtures,prior,checkpoint,weight_stat,runtime


@pytest.mark.parametrize('bad',['none','noise','fixture_stat','checkpoint'])
def test_shared_disk_input_admission_preserves_seven_priors_and_both_checkpoint_schemas(driver,tmp_path,bad):
    fixtures,prior,checkpoint,weight_stat,runtime=disk_policy_inputs(driver,tmp_path);called=[]
    if bad=='noise':
        p=prior/'baseline/actions.npz'
        with np.load(p,allow_pickle=False) as z:saved={k:z[k].copy() for k in z.files}
        saved['noise'][0,0]+=1;np.savez(p,**saved)
    if bad=='fixture_stat':
        p=fixtures/'baseline.npz'
        with np.load(p,allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
        np.savez(p,**values,unexpected=np.zeros(100))
    if bad=='checkpoint':checkpoint=copy.deepcopy(checkpoint);checkpoint['config']['mtime_ns']+=1
    def consume():
        out=driver.load_policy_inputs(fixtures,prior,task=tmp_path,checkpoint=checkpoint,weight_stat=weight_stat,runtime=runtime)
        called.append('backend_factory');return out
    if bad=='none':
        data,priors,records,legacy,oldrecords=consume()
        assert set(data)==set(priors)==set(records)==set(driver.VARIANTS) and set(legacy)==set(oldrecords)=={'baseline','state'}
        assert all(records[c]['admission']['admitted'] for c in driver.VARIANTS)
        assert torch.equal(legacy['baseline']['legacy.transport_sentinel'],torch.tensor([3.]))
    else:
        with pytest.raises(ValueError,match='prior|identity|checkpoint|input|Folded'):consume()
        assert not called


class CPURequestBackend:
    def __init__(self):
        from test_hyvla_expert_policy_scope_example import Front
        self.frontend=Front();self.frontend.num_steps=10
        self.frontend._embedding_times=[1.+i*(-1./10) for i in range(10)];self.calls=0
    def infer(self,observation,noise):
        self.calls+=1
        result=self.frontend.run()
        return {'actions':result[0].numpy().copy(),
            'sampler_metadata':{'sampler':'euler','scheduled_steps':10,'nfe':10,'timesteps':[1.+i*(-1./10) for i in range(10)],
                'numerical_profile':'official_local_fp32','operation_dtypes':{'euler_increment':'bfloat16'},
                'embedding_timesteps':self.frontend._embedding_times,'profile_scope':'local_math_hooks_not_official_exact'},
            'backend_metadata':{'model':'hyvla','backend':'flashrt','sampler':'euler','hardware':'cuda_sm80','precision':'bf16',
                'framework':'torch','action_domain':'normalized-padded-model-actions'}}


def request_arrays():
    return {**{k:np.zeros((3,2,2),np.uint8) for k in ('top_head','hand_left','hand_right')},
        'state':np.zeros(20,np.float32),'prompt':np.asarray('unit'),'noise':np.zeros((40,32),np.float32)}


def test_shared_executor_retains_native_expert_and_waits_for_auxiliary_cleanup_before_saving(driver,tmp_path):
    backend=CPURequestBackend();events=[]
    @contextmanager
    def extra(tensors):
        state={'complete':False,'cleanup':{'passed':False}};events.append('enter')
        tensors['auxiliary.value']=torch.ones(1)
        yield state
        events.append('cleanup');state.update(complete=True,cleanup={'passed':True})
    record=driver.execute_request(backend,request_arrays(),output=tmp_path,arm='candidate',variant='baseline',index=0,
        expert_enabled=False,separate_block=None,synchronize=lambda:None,request_context=extra)
    assert backend.calls==1 and record['scope']['enabled'] is False and record['scope']['nfe']==10
    assert all(s['norm']['original_norm_calls']==129 for s in record['scope']['steps'])
    saved=json.loads((tmp_path/'candidate/baseline/replay0/result.json').read_text())
    assert saved['status']=='request_captured_pending_backend_cleanup' and saved['operator_scope']['complete']
    assert saved['operator_scope']['cleanup']['passed'] and events==['enter','cleanup']


def test_shared_executor_never_infers_when_expert_flag_requires_missing_template(driver,tmp_path):
    backend=CPURequestBackend()
    with pytest.raises(ValueError,match='unbound|separate'):
        driver.execute_request(backend,request_arrays(),output=tmp_path,arm='candidate',variant='baseline',index=0,
            expert_enabled=True,separate_block=None,synchronize=lambda:None)
    assert backend.calls==0 and not (tmp_path/'candidate/baseline/replay0/result.json').exists()


def test_shared_executor_auxiliary_failure_keeps_partial_without_success_sidecar(driver,tmp_path):
    backend=CPURequestBackend()
    @contextmanager
    def broken(tensors):
        tensors['auxiliary.value']=torch.ones(1)
        yield {'complete':False,'cleanup':{'passed':False}}
        raise RuntimeError('aux cleanup failed')
    with pytest.raises(RuntimeError,match='aux cleanup'):
        driver.execute_request(backend,request_arrays(),output=tmp_path,arm='control',variant='baseline',index=0,
            expert_enabled=False,separate_block=None,synchronize=lambda:None,request_context=broken)
    folder=tmp_path/'control/baseline/replay0'
    assert (folder/'unqualified_partial.safetensors').exists() and (folder/'failure.json').exists() and not (folder/'result.json').exists()


@pytest.mark.parametrize('bad',['arm','variant','index','expert_flag'])
def test_shared_executor_rejects_unbound_request_identity_before_any_output_or_inference(driver,tmp_path,bad):
    backend=CPURequestBackend();output=tmp_path/'unused'
    options={'arm':'control','variant':'baseline','index':0,'expert_enabled':False}
    if bad=='arm':options['arm']='other'
    if bad=='variant':options['variant']='../../escape'
    if bad=='index':options['index']=2
    if bad=='expert_flag':options['expert_enabled']='false'
    with pytest.raises(ValueError,match='request|identity|flag'):
        driver.execute_request(backend,request_arrays(),output=output,separate_block=None,synchronize=lambda:None,**options)
    assert backend.calls==0 and not output.exists()
