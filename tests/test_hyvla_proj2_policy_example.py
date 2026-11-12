import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType

import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/proj2_policy.py'


@pytest.fixture
def policy2():
    spec=importlib.util.spec_from_file_location('hy_proj2_policy_test',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tiny_owner(policy2):
    from test_hyvla_merger_stages_example import weights
    owner=policy2.merger.native_owner(weights())
    cls=type('ActualCameraOwner',(),{'merger_forward':policy2.ablation.camera_merger})
    pipe=cls();pipe.W=owner.W;pipe._camera_proj1_fold_spatial=True
    return pipe,torch.arange(12,dtype=torch.bfloat16).reshape(3,4,1)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_merger_ast_rebinds_real_owner_and_captures_all_four_live_fields(policy2,enabled):
    pipe,x=tiny_owner(policy2);before=x.clone();original=pipe.merger_forward(x,grid=2)
    del pipe._camera_proj1_encoding_views;raw={};state={}
    with policy2.merger_request(pipe,enabled=enabled,tensors=raw,state=state,synchronize=lambda:None):
        result=pipe.merger_forward(x,grid=2)
    assert state['complete'] and state['cleanup']['passed'] and state['merger_calls']==state['proj2_calls']==1
    assert len(state['proj2_views'])==(3 if enabled else 1)
    assert all(v['shares_input_storage'] for v in state['proj2_views'])
    assert set(raw)=={'vision.merger_input','vision.proj2_activation','vision.proj2_output','vision.merged'}
    assert torch.equal(result,original) and torch.equal(x,before)
    assert torch.equal(raw['vision.proj2_output'].reshape(result.shape),result)
    assert state['proj2_activation_layout']['shape']==[3,1,1,2]
    if enabled:
        assert [v['storage_offset'] for v in state['proj2_views']]==[0,2,4]
        assert [v['shape'] for v in state['proj2_views']]==[[1,1,1,2]]*3
    assert 'merger_forward' not in pipe.__dict__ and '_camera_proj1_encoding_views' not in pipe.__dict__ and '_proj2_policy_active' not in pipe.__dict__


@pytest.mark.parametrize('bad',['signature','bias','duplicate'])
def test_source_drift_rejects_before_callback_execution(policy2,tmp_path,bad):
    import inspect
    source=inspect.getsource(policy2.ablation.camera_merger)
    if bad=='signature':source=source.replace('grid=14','grid=16')
    if bad=='bias':source=source.replace('F.linear(x, W._mg_proj2_w, W._mg_proj2_b)','F.linear(x, W._mg_proj2_w, None)')
    if bad=='duplicate':source=source.replace('    return x.reshape(B, -1, C)','    x = F.linear(x, W._mg_proj2_w, W._mg_proj2_b)\n    return x.reshape(B, -1, C)')
    file=tmp_path/'drift.py';file.write_text(source);calls=[]
    with pytest.raises(ValueError,match='source|AST|PROJ2|signature'):
        policy2.adapt_merger(file,lambda *a:calls.append(a))
    assert not calls


@pytest.mark.parametrize('bad',['nested','body','sync','second_call'])
def test_merger_scope_restores_existing_metadata_and_binding_on_all_failures(policy2,bad):
    pipe,x=tiny_owner(policy2);original=pipe.merger_forward;pipe.merger_forward=original
    old_views=['prior-metadata'];pipe._camera_proj1_encoding_views=old_views;state={}
    def sync():
        if bad=='sync':raise RuntimeError('sync failed')
    with pytest.raises((ValueError,RuntimeError),match='(?i)nested|body|sync|count|call|cleanup'):
        with policy2.merger_request(pipe,enabled=True,tensors={},state=state,synchronize=sync):
            if bad=='body':raise RuntimeError('body failed')
            if bad=='nested':
                with policy2.merger_request(pipe,enabled=True,tensors={},state={},synchronize=lambda:None):pass
            pipe.merger_forward(x,grid=2)
            if bad=='second_call':pipe.merger_forward(x,grid=2)
    assert not state['complete'] and pipe.merger_forward is original and pipe._camera_proj1_encoding_views is old_views
    assert '_proj2_policy_active' not in pipe.__dict__


def test_candidate_callback_cannot_mutate_weight_and_still_publish_complete(policy2,monkeypatch):
    pipe,x=tiny_owner(policy2);original=policy2.ablation.per_camera_proj1
    def broken(value,w,b,**kw):
        if not kw.get('fold_spatial'):w.add_(1)
        return original(value,w,b,**kw)
    monkeypatch.setattr(policy2.ablation,'per_camera_proj1',broken);state={}
    with pytest.raises(ValueError,match='readonly|read-only|weight|operand'):
        with policy2.merger_request(pipe,enabled=True,tensors={},state=state,synchronize=lambda:None):pipe.merger_forward(x,grid=2)
    assert not state['complete'] and 'merger_forward' not in pipe.__dict__


def frozen_proof(policy2,tmp_path):
    ck=tmp_path/'model.safetensors';ck.write_bytes(b'checkpoint stat only')
    stat=policy2.merger.file_record(ck);runtime={'torch':'2.11.0+cu128'}
    x=torch.zeros(3,196,1152,dtype=torch.bfloat16);a=torch.zeros(3,7,7,2048,dtype=torch.bfloat16);m=a.reshape(3,49,2048)
    raw={};references={}
    for origin in ('gold','native'):
        references[origin]={'input':x,'activation':a,'native':m,'official':m}
        for key,value in (('input',x),('control_activation',a),('candidate_activation',a),('control_proj2',a),('candidate_proj2',a),
                ('original_merged',m),('control_merged',m),('candidate_merged',m),('official_reference',m)):
            raw[origin+'.'+key]=value
    p1=torch.zeros(3,14,14,2048,dtype=torch.bfloat16);raw.update({'proj1.control':p1,'proj1.reference':p1});references['proj1']=p1
    child={'kind':'hy_camera_proj1_exact_frozen_precondition','operator_exact':True,'job_id':'9','source_revision':'a'*40,'checkpoint':stat,'fold_spatial':True}
    report={'kind':'hy_complete_merger_proj2_frozen_precondition','status':'operator_precondition_complete','operator_exact':True,
        'job_id':'9','source_revision':'a'*40,'checkpoint':stat,'runtime':runtime,'torch_git_version':'b'*40,'model_constructed':False,
        'weight_read_events':8,'selected_checkpoint_keys':[policy2.merger.CHECKPOINT_PREFIX+k for k in policy2.merger.WEIGHTS],
        'proj1_child':child,'raw_proof':{'tensors':20,'original_and_observed_controls_exact':2,'candidate_endpoints_exact':2,'PROJ1_child_exact':True}}
    report['observations']={}
    for origin in ('gold','native'):
        report['observations'][origin]={}
        for arm,enabled in (('control',False),('candidate',True)):
            views=[]
            for i in range(3 if enabled else 1):
                v=copy.deepcopy(policy2.ablation._layout(a));v['shares_input_storage']=True
                if enabled:v['shape'][0]=1;v['storage_offset']=i*a.stride(0)
                views.append(v)
            report['observations'][origin][arm]={'enabled':enabled,'complete':True,'cleanup':{'passed':True,'errors':[]},
                'merger_calls':1,'proj2_calls':1,'proj2_views':views,'proj1_views':[{} for _ in range(3)],
                'merger_input_layout':policy2.ablation._layout(x),'proj2_activation_layout':policy2.ablation._layout(a),'tensor_metadata':{}}
    return report,raw,references,stat,runtime


@pytest.mark.parametrize('bad',['none','job','dummy','activation','candidate','child','child_kind','forged_pair','forged_child_pair','operator_cleanup','view_alias','callback_count'])
def test_complete_precondition_has_strict_new_identity_and_twenty_actual_raw_roles_before_backend(policy2,tmp_path,bad):
    report,raw,refs,ck,runtime=frozen_proof(policy2,tmp_path);called=[]
    if bad=='job':report['job_id']='8'
    if bad=='dummy':del raw['native.candidate_activation'];raw['dummy']=torch.ones(1)
    if bad=='activation':raw['gold.candidate_activation']=raw['gold.candidate_activation']+1
    if bad=='candidate':raw['gold.candidate_merged']=raw['gold.candidate_merged']+1
    if bad=='child':raw['proj1.control']=raw['proj1.control']+1
    if bad=='child_kind':report['proj1_child']['kind']=report['kind']
    if bad=='forged_pair':
        for field in ('candidate_merged','candidate_proj2','official_reference'):raw['gold.'+field]=raw['gold.'+field]+1
    if bad=='forged_child_pair':
        for field in ('proj1.control','proj1.reference'):raw[field]=raw[field]+1
    if bad=='operator_cleanup':report['observations']['gold']['candidate']['cleanup']['passed']=False
    if bad=='view_alias':report['observations']['gold']['candidate']['proj2_views'][0]['shares_input_storage']=False
    if bad=='callback_count':report['observations']['gold']['candidate']['proj2_calls']=0
    def factory():called.append(True);return 'no model built'
    call=lambda:policy2.admit_precondition(report,raw,references=refs,job_id='9',source_revision='a'*40,checkpoint=ck,runtime=runtime,torch_git='b'*40,factory=factory)
    if bad=='none':assert call()=='no model built' and called==[True]
    else:
        with pytest.raises(ValueError,match='precondition|raw|activation|endpoint|child|PROJ1|identity|role'):call()
        assert not called


def policy_frames(policy2,P=151):
    from test_hyvla_native_policy_tree_example import request_fixture,clone_request
    control=request_fixture(policy2.legacy);raw=control['tensors'];mm=torch.zeros(1,P,dtype=torch.bool);mm[:,2:149]=True
    perm=torch.cat((mm[0].nonzero().flatten(),(~mm[0]).nonzero().flatten()))
    raw.update({'prefix.modality_mask':mm,'prefix.permutation':perm,'prefix.embeddings':torch.zeros(1,P,2048,dtype=torch.bfloat16),
        'prefix.k':torch.zeros(1),'prefix.v':torch.zeros(1,dtype=torch.bfloat16),'vision.merger_input':torch.zeros(3,196,1152,dtype=torch.bfloat16),
        'vision.proj2_activation':torch.zeros(3,7,7,2048,dtype=torch.bfloat16),'vision.proj2_output':torch.zeros(3,7,7,2048,dtype=torch.bfloat16),
        'vision.merged':torch.zeros(3,49,2048,dtype=torch.bfloat16)})
    raw['prefix.sorted_embeddings']=raw['prefix.embeddings'][:,perm].clone();control['scope']['enabled']=False
    control['operator_scope']={'merger_input_layout':policy2.ablation._layout(raw['vision.merger_input']),
        'proj2_activation_layout':policy2.ablation._layout(raw['vision.proj2_activation'])}
    candidate=clone_request(control);candidate['scope']=copy.deepcopy(control['scope']);candidate['operator_scope']=copy.deepcopy(control['operator_scope'])
    return control,candidate


@pytest.mark.parametrize('P',[151,155])
def test_intended_visual_and_kv_changes_are_allowed_but_nonvisual_prefix_fixed(policy2,P):
    control,candidate=policy_frames(policy2,P);r=candidate['tensors']
    r['vision.proj2_output'][0,0,0,0]=.5;r['vision.merged']=r['vision.proj2_output'].reshape(3,49,2048).clone()
    mm=r['prefix.modality_mask'][0];r['prefix.embeddings'][:,mm]=r['vision.merged'].reshape(1,147,2048)
    r['prefix.sorted_embeddings']=r['prefix.embeddings'][:,r['prefix.permutation']].clone();r['prefix.k']+=1;r['prefix.v']+=1
    r['step1.latent_before']+=1
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    q=policy2.qualify_merger_candidate(candidate,control,arrays)
    assert q['same_action_gate_passed'] and q['fixed_inputs_and_non_image_prefix_exact_control']
    assert not q['effects_vs_control']['prefix.k']['exact'] and not q['effects_vs_control']['vision.merged']['exact']


@pytest.mark.parametrize('bad',['merger_input','activation','nonimage','mask','expert'])
def test_single_variable_qualification_rejects_unintended_input_or_expert_changes(policy2,bad):
    control,candidate=policy_frames(policy2);r=candidate['tensors']
    if bad=='merger_input':r['vision.merger_input']+=1
    if bad=='activation':r['vision.proj2_activation']+=1
    if bad=='nonimage':
        r['prefix.embeddings'][0,0,0]=1
        r['prefix.sorted_embeddings']=r['prefix.embeddings'][:,r['prefix.permutation']].clone()
    if bad=='mask':r['prefix.pad_mask']+=1
    if bad=='expert':candidate['scope']['enabled']=True
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    with pytest.raises(ValueError,match='invariant|input|Expert|non-image|prefix|condition|sorted'):
        policy2.qualify_merger_candidate(candidate,control,arrays)


def test_zero_numerical_effect_is_valid_not_a_forced_improvement(policy2):
    control,candidate=policy_frames(policy2)
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    assert policy2.qualify_merger_candidate(candidate,control,arrays)['effects_vs_control']['vision.merged']['exact']


def test_complete_frozen_producer_runs_actual_original_and_two_observed_mergers(policy2):
    from test_hyvla_merger_stages_example import weights
    values=tuple(torch.ones(1,2,2,1,dtype=torch.bfloat16) for _ in range(3));native=torch.arange(12,dtype=torch.bfloat16).reshape(3,4,1)
    history={'gold_values':values,'native_input':native,'official_strides':[[4,2,1,1]]*3,
        'native_metadata':{'shape':[3,4,1],'live_stride':[4,1,4],'dtype':'torch.bfloat16'}}
    act=torch.full((3,1,1,2),1.953125,dtype=torch.bfloat16);out=act.reshape(3,1,2)
    refs={o:{'input':torch.cat(values).reshape(3,4,1) if o=='gold' else native,'activation':act,'native':out,'official':out} for o in ('gold','native')}
    refs['proj1']=torch.full((3,2,2,2),2.,dtype=torch.bfloat16)
    raw,observations,p1views=policy2.complete_frozen_controls(weights(),history,refs,device='cpu',synchronize=lambda:None)
    assert len(raw)==20 and len(p1views)==3 and torch.equal(raw['proj1.control'],refs['proj1'])
    for origin in ('gold','native'):
        assert torch.equal(raw[origin+'.input'],refs[origin]['input'])
        for arm,count in (('control',1),('candidate',3)):
            assert observations[origin][arm]['complete'] and observations[origin][arm]['cleanup']['passed']
            assert len(observations[origin][arm]['proj2_views'])==count
        assert torch.equal(raw[origin+'.candidate_merged'],out)


def test_candidate_request_operator_never_enables_the_old_expert_triple(policy2,tmp_path):
    from test_hyvla_native_policy_tree_example import CPURequestBackend,request_arrays
    backend=CPURequestBackend();pipe,x=tiny_owner(policy2);actual=backend.frontend.pipe
    actual.W=pipe.W;actual._camera_proj1_fold_spatial=True
    actual.merger_forward=MethodType(policy2.ablation.camera_merger,actual)
    original=backend.infer
    def infer(observation,noise):
        actual.merger_forward(x,grid=2)
        return original(observation,noise)
    backend.infer=infer
    record=policy2.execute_policy_request(backend,request_arrays(),output=tmp_path,arm='candidate',variant='baseline',index=0,synchronize=lambda:None)
    assert len(record['tensors'])==64 and record['scope']['enabled'] is False
    assert all(s['norm']['original_norm_calls']==129 for s in record['scope']['steps'])
    assert record['operator_scope']['enabled'] and record['operator_scope']['complete']
    assert len(record['operator_scope']['proj2_views'])==3 and actual.separate_seen==0


def test_help_and_allocation_guard_before_any_weight_or_model_read(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--precondition-report' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--arm','preflight','--reference-root','/missing','--source-revision','a'*40,'--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()
