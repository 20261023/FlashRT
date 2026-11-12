import copy
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import MethodType,SimpleNamespace

import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/policy_factorial.py'


@pytest.fixture
def factorial():
    spec=importlib.util.spec_from_file_location('hy_factorial_test',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


@pytest.mark.parametrize('group,E,P',[('base',False,False),('Eonly',True,False),('Ponly',False,True),('both',True,True)])
def test_group_adapter_wires_independent_factors_and_unique_global_identity(factorial,tmp_path,group,E,P):
    from test_hyvla_native_policy_tree_example import CPURequestBackend,request_arrays
    from test_hyvla_merger_stages_example import weights
    from test_hyvla_expert_policy_scope_example import separate
    backend=CPURequestBackend();pipe=backend.frontend.pipe;owner=factorial.p2.merger.native_owner(weights())
    pipe.W=owner.W;pipe._camera_proj1_fold_spatial=True
    pipe.merger_forward=MethodType(factorial.ablation.camera_merger,pipe)
    original=backend.infer;x=torch.ones(3,4,1,dtype=torch.bfloat16)
    def infer(observation,noise):pipe.merger_forward(x,grid=2);return original(observation,noise)
    backend.infer=infer
    r=factorial.execute_group_request(backend,request_arrays(),group=group,output=tmp_path,variant='baseline',index=0,
        separate_template=separate,synchronize=lambda:None)
    assert r['scope']['enabled'] is E and r['operator_scope']['enabled'] is P
    assert r['operator_scope']['E'] is E and r['operator_scope']['P'] is P
    assert r['operator_scope']['global_request_id']==group+'.baseline.r0' and r['operator_scope']['factorial_group']==group
    assert r['global_path'].startswith('groups/'+group+'/')
    assert all(s['norm']['original_norm_calls']==(0 if E else 129) for s in r['scope']['steps'])
    assert len(r['operator_scope']['proj2_views'])==(3 if P else 1) and pipe.separate_seen==(320 if E else 0)


def test_unknown_group_and_shared_identity_guard_are_not_relaxed(factorial,tmp_path):
    from test_hyvla_native_policy_tree_example import CPURequestBackend,request_arrays
    backend=CPURequestBackend()
    with pytest.raises(ValueError,match='group|factor'):
        factorial.execute_group_request(backend,request_arrays(),group='../../escape',output=tmp_path/'unused',variant='baseline',index=0,separate_template=None,synchronize=lambda:None)
    assert not (tmp_path/'unused').exists() and backend.calls==0
    with pytest.raises(ValueError,match='identity|request'):
        factorial.legacy.execute_request(backend,request_arrays(),output=tmp_path/'old_unused',arm='both',variant='baseline',index=0,expert_enabled=False,separate_block=None,synchronize=lambda:None)
    assert not (tmp_path/'old_unused').exists()


def toy_execute(factorial,calls):
    from test_hyvla_native_policy_tree_example import request_fixture
    def run(group,case,index):calls.append((group,case,index));return request_fixture(factorial.legacy)
    return run


@pytest.mark.parametrize('bad_group',['base','Eonly','Ponly'])
def test_any_known_cell_failure_blocks_all_joint_requests(factorial,bad_group):
    calls=[]
    def known(group,case,pair,matrix):
        if group==bad_group and case=='state':raise ValueError('known raw/reference mismatch')
        return {'raw_reference_exact':True}
    with pytest.raises(ValueError,match='known|reference'):
        factorial.run_factorial(toy_execute(factorial,calls),known,lambda group,case,first,matrix:{'same_action_gate_passed':False})
    assert calls and all(c[0]!='both' for c in calls)


def test_exactly_three_known_cells_precede_joint_and_negative_joint_is_valid(factorial):
    calls=[]
    matrix=factorial.run_factorial(toy_execute(factorial,calls),lambda *a:{'raw_reference_exact':True},
        lambda group,case,first,matrix:{'same_action_gate_passed':False})
    assert len(calls)==56 and {c[0] for c in calls[:42]}=={'base','Eonly','Ponly'} and all(c[0]=='both' for c in calls[42:])
    assert list(matrix)==['base','Eonly','Ponly','both'] and all(not row['qualification']['same_action_gate_passed'] for row in matrix['both'].values())


def test_false_known_admission_flag_is_not_treated_as_an_approved_cell(factorial):
    calls=[]
    with pytest.raises(ValueError,match='known|reference'):
        factorial.run_factorial(toy_execute(factorial,calls),lambda *a:{'raw_reference_exact':False},lambda *a:{'same_action_gate_passed':False})
    assert len(calls)==2


@pytest.mark.parametrize('bad',['none','dtype','value','source','missing_vision'])
def test_actual_known_raw_cell_admission_binds_reference_and_new_E_vision_provenance(factorial,bad):
    from test_hyvla_native_policy_tree_example import clone_request
    cells=cells_fixture(factorial);matrix={'base':{'baseline':{'requests':[cells['base'],copy.deepcopy(cells['base'])]}}}
    refs={g:{} for g in ('base','Eonly','Ponly')}
    for group,job,source in (('base','317665','a507813727816c1bf201bbcf759c0df87d57d905'),
            ('Eonly','316975','54a661d52a7b3a9d4619eb4f3639ff5ae7f29e91'),('Ponly','317665','a507813727816c1bf201bbcf759c0df87d57d905')):
        old=copy.deepcopy(cells[group])
        if group=='Eonly':old['tensors']={k:v for k,v in old['tensors'].items() if k not in factorial.p2.VISION_FIELDS}
        refs[group]['baseline']={'job_id':job,'source_revision':source,'variant':'baseline',**old}
    pair=[copy.deepcopy(cells['Eonly']) for _ in range(2)]
    if bad=='dtype':pair[0]['tensors']['step0.velocity']=pair[0]['tensors']['step0.velocity'].float()
    if bad=='value':pair[1]['tensors']['actions']+=1
    if bad=='source':refs['Eonly']['baseline']['source_revision']='wrong'
    if bad=='missing_vision':del pair[0]['tensors']['vision.proj2_activation']
    call=lambda:factorial.admit_known_cell('Eonly','baseline',pair,matrix,refs)
    if bad=='none':
        proof=call();assert proof['raw_reference_exact'] and proof['matched_historical_fields']==60
        assert proof['new_vision_fields_exact_current_base']==4 and proof['reference_job_id']=='316975'
    else:
        with pytest.raises(ValueError,match='known|reference|raw|vision|source'):call()


def known_header(factorial,group,inputs,runtime,checkpoint):
    job,source,arm=factorial.KNOWN[group];is_E=group=='Eonly'
    return {'kind':'hy_native_policy_expert_phase_source_tree_diagnostic' if is_E else 'hy_native_policy_single_PROJ2_diagnostic',
        'status':'diagnostic_complete','job_id':job,'source_revision':source,'runtime':runtime,'checkpoint':checkpoint,
        'torch_git_version':'b'*40,'inputs':inputs,'requests':28,'actual_nfe':280,'weight_instances':1,
        'backend_cleanup':{'passed':True},'graphs_enabled':False,'performance_measurement':False,'production_default_changed':False,
        'matrix':{a:{c:{'request_paths':[f'{a}/{c}/replay{i}' for i in (0,1)],'qualification':{}} for c in factorial.legacy.VARIANTS} for a in ('control','candidate')}}


@pytest.mark.parametrize('bad',['none','kind','input_case','checkpoint','raw_path','runtime','cleanup'])
def test_known_parent_header_cannot_be_substituted_across_sources_cases_or_runtime(factorial,bad):
    inputs={c:{'fixture':{'manifest':{'variant':c}},'admission':{'admitted':True}} for c in factorial.legacy.VARIANTS}
    runtime={'torch':'unit'};checkpoint={'path':'unit','config':{'bytes':17}}
    report=known_header(factorial,'Eonly',copy.deepcopy(inputs),runtime,copy.deepcopy(checkpoint))
    if bad=='kind':report['kind']='hy_native_policy_single_PROJ2_diagnostic'
    if bad=='input_case':report['inputs']['state']=report['inputs']['baseline']
    if bad=='checkpoint':report['checkpoint']['config']['bytes']=18
    if bad=='raw_path':report['matrix']['candidate']['baseline']['request_paths'][0]='candidate/state/replay0'
    if bad=='runtime':report['runtime']={'torch':'other'}
    if bad=='cleanup':report['backend_cleanup']['passed']=False
    call=lambda:factorial.admit_known_header(report,'Eonly',inputs=inputs,runtime=runtime,checkpoint=checkpoint,torch_git='b'*40)
    if bad=='none':call()
    else:
        with pytest.raises(ValueError,match='known|reference|identity|path|input'):call()


@pytest.mark.parametrize('bad',['none','raw_replay','sidecar_identity','metadata','both_velocity_dtype','both_NFE','both_shape'])
def test_real_disk_known_pair_checks_both_captures_and_records_exact_reference_files(factorial,tmp_path,bad):
    from safetensors.torch import save_file
    record=full_capture(factorial,'Eonly');record['tensors']={k:v for k,v in record['tensors'].items() if k not in factorial.p2.VISION_FIELDS}
    for i in (0,1):
        folder=tmp_path/'candidate/baseline'/f'replay{i}';folder.mkdir(parents=True)
        meta={k:v for k,v in record.items() if k not in ('tensors','operator_scope')}
        meta.update(status='request_captured_pending_backend_cleanup',request_id=f'candidate.baseline.r{i}',path=f'candidate/baseline/replay{i}')
        if bad=='sidecar_identity' and i==1:meta['request_id']='candidate.state.r1'
        if bad=='metadata' and i==1:meta['backend_metadata']={'precision':'fp32'}
        if bad=='both_NFE':meta['sampler_metadata']={**meta['sampler_metadata'],'nfe':9}
        raw={k:v.clone() for k,v in record['tensors'].items()}
        if bad=='raw_replay' and i==1:raw['step9.velocity']+=1
        if bad=='both_velocity_dtype':raw['step0.velocity']=raw['step0.velocity'].float()
        if bad=='both_shape':raw['prepared.state']=raw['prepared.state'][0]
        save_file(raw,str(folder/'trace.safetensors'));(folder/'result.json').write_text(json.dumps(meta))
    call=lambda:factorial.load_known_pair(tmp_path,group='Eonly',case='baseline')
    if bad=='none':
        ref=call();assert len(ref['tensors'])==60 and len(ref['reference_files'])==4
        assert ref['job_id']=='316975' and ref['variant']=='baseline'
    else:
        with pytest.raises(ValueError,match='known|reference|replay|metadata'):call()


@pytest.mark.parametrize('bad',['none','frozen_source_stat','source_path','missing_role','both_dtype','both_sampler'])
def test_known_reference_loader_real_seven_case_disk_contract_before_model_consumer(factorial,tmp_path,bad):
    from safetensors.torch import save_file
    inputs={c:{'fixture':{'manifest':{'variant':c}},'admission':{'admitted':True}} for c in factorial.legacy.VARIANTS}
    runtime={'torch':'unit'};checkpoint={'path':'unit'}
    for group in ('Eonly','Ponly'):
        job,source,_=factorial.KNOWN[group];rnd='round033' if group=='Eonly' else 'round036'
        root=tmp_path/'results'/('hy-native-policy-tree-'+job if group=='Eonly' else 'hy-proj2-policy-'+job);root.mkdir(parents=True)
        (root/'SOURCE_GIT_COMMIT').write_text(source+'\n')
        report=known_header(factorial,group,inputs,runtime,checkpoint);report['bf16_reduced_precision_reduction']=True
        mapping=({'driver':factorial.legacy.__file__,'scope':factorial.legacy.policy.__file__,'RMS':factorial.legacy.expert.__file__,'tree':factorial.legacy.expert.mt.__file__}
            if group=='Eonly' else {k:v['path'] for k,v in factorial.p2._source_records().items()})
        report['source_records']={}
        for name,path in mapping.items():
            relative=Path(path).resolve().relative_to(ROOT);frozen=tmp_path/'snapshots'/rnd/'code'/relative;frozen.parent.mkdir(parents=True,exist_ok=True)
            frozen.write_bytes(Path(path).read_bytes());report['source_records'][name]=factorial.p2.merger.file_record(frozen)
        for arm in ('control','candidate'):
            sample=full_capture(factorial,'Eonly' if group=='Eonly' else 'base' if arm=='control' else 'Ponly')
            common_raw={k:v.contiguous() for k,v in sample['tensors'].items() if group!='Eonly' or k not in factorial.p2.VISION_FIELDS}
            if bad=='missing_role' and group=='Eonly' and arm=='candidate':common_raw.pop('step0.velocity')
            if bad=='both_dtype' and group=='Eonly' and arm=='candidate':common_raw['step0.velocity']=common_raw['step0.velocity'].float()
            seed=root/(arm+'-fixture.safetensors');save_file(common_raw,str(seed))
            for case in factorial.legacy.VARIANTS:
                for i in (0,1):
                    record={k:copy.deepcopy(v) for k,v in sample.items() if k not in ('tensors','operator_scope')}
                    if bad=='both_sampler' and group=='Eonly' and arm=='candidate':record['sampler_metadata']['nfe']=9
                    folder=root/'native'/arm/case/f'replay{i}';folder.mkdir(parents=True)
                    record.update(status='request_captured_pending_backend_cleanup',request_id=f'{arm}.{case}.r{i}',path=f'{arm}/{case}/replay{i}')
                    record['scope']['request_id']=record['request_id']
                    (folder/'trace.safetensors').hardlink_to(seed);(folder/'result.json').write_text(json.dumps(record))
        if bad=='frozen_source_stat' and group=='Eonly':Path(report['source_records']['scope']['path']).write_bytes(b'changed source')
        if bad=='source_path' and group=='Eonly':report['source_records']['scope']=report['source_records']['driver']
        (root/'native/result.json').write_text(json.dumps(report))
    consumed=[]
    def consume():
        result=factorial.load_known_references(tmp_path,inputs=inputs,runtime=runtime,checkpoint=checkpoint,torch_git='b'*40,reduction_flag=True)
        consumed.append('model consumer');return result
    if bad=='none':
        refs,records=consume()
        assert set(refs)=={'base','Eonly','Ponly'} and all(set(r)==set(factorial.legacy.VARIANTS) for r in refs.values())
        assert refs['Eonly']['noise']['job_id']=='316975' and len(refs['base']['state']['tensors'])==64
        assert all(len(records[g]['requests'])==7 for g in refs)
    else:
        with pytest.raises(ValueError,match='Known|known|reference|source|raw'):consume()
        assert not consumed


def cells_fixture(factorial):
    from test_hyvla_proj2_policy_example import policy_frames
    base,P=policy_frames(factorial.p2);base['scope']['enabled']=False
    cells={}
    for group,E,pflag in (('base',False,False),('Eonly',True,False),('Ponly',False,True),('both',True,True)):
        r=copy.deepcopy(base);r['scope']['enabled']=E
        r['operator_scope'].update(enabled=pflag,E=E,P=pflag,factorial_group=group,global_request_id=group+'.baseline.r0')
        raw=r['tensors']
        for s in range(10):raw[f'step{s}.time']=torch.tensor([1.-s*.1])
        if pflag:
            raw['vision.proj2_output'][0,0,0,0]=.5;raw['vision.merged']=raw['vision.proj2_output'].reshape(3,49,2048).clone()
            mm=raw['prefix.modality_mask'][0];raw['prefix.embeddings'][:,mm]=raw['vision.merged'].reshape(1,147,2048)
            raw['prefix.sorted_embeddings']=raw['prefix.embeddings'][:,raw['prefix.permutation']].clone();raw['prefix.k']+=1;raw['prefix.v']+=1
        cells[group]=r
    return cells


@pytest.mark.parametrize('group',['Eonly','Ponly','both'])
def test_factorial_qualifier_respects_fixed_P_prefix_and_allowed_P_effects(factorial,group):
    cells=cells_fixture(factorial)
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    q=factorial.qualify_cell(group,cells[group],cells,arrays)
    assert q['same_action_gate_passed'] and q['all_factorial_fixed_conditions_exact']
    assert q['fixed_P_prefix_invariant_checked']==(group in ('Eonly','both'))


@pytest.mark.parametrize('bad',['Eonly_KV','both_KV','activation','nonimage','flags'])
def test_joint_invariants_reject_prefill_leakage_or_wrong_factor_binding(factorial,bad):
    cells=cells_fixture(factorial);group='both'
    if bad=='Eonly_KV':group='Eonly';cells[group]['tensors']['prefix.k']+=1
    if bad=='both_KV':cells[group]['tensors']['prefix.k']+=1
    if bad=='activation':cells[group]['tensors']['vision.proj2_activation']+=1
    if bad=='nonimage':
        r=cells[group]['tensors'];r['prefix.embeddings'][0,0,0]+=1;r['prefix.sorted_embeddings']=r['prefix.embeddings'][:,r['prefix.permutation']].clone()
    if bad=='flags':cells[group]['scope']['enabled']=False
    arrays={'expected_actions32':np.ones((40,32),np.float32),'expected_actions20':np.ones((40,20),np.float32)}
    with pytest.raises(ValueError,match='prefix|invariant|factor|input|condition'):
        factorial.qualify_cell(group,cells[group],cells,arrays)


@pytest.mark.parametrize('joint,expected',[(6.,0.),(8.,2.)])
def test_literal_additive_and_nonadditive_interaction_formula(factorial,joint,expected):
    cells=cells_fixture(factorial)
    for group,v in {'base':1.,'Eonly':3.,'Ponly':4.,'both':joint}.items():
        cells[group]['tensors']['step0.velocity'].fill_(v);cells[group]['tensors']['actions'].fill_(v)
    arrays,stats=factorial.interactions({g:r['tensors'] for g,r in cells.items()})
    assert set(arrays)=={'first_velocity','actions'}
    for value in arrays.values():assert value.dtype==torch.float64 and torch.equal(value,torch.full((1,40,32),expected,dtype=torch.float64))
    assert stats['first_velocity']['nonzero_elements']==(0 if expected==0 else 1280)


def test_first_velocity_interaction_rejects_different_initial_latent(factorial):
    cells=cells_fixture(factorial);cells['both']['tensors']['step0.latent_before']+=1
    with pytest.raises(ValueError,match='initial|condition|interaction'):
        factorial.interactions({g:r['tensors'] for g,r in cells.items()})


def test_new_publisher_never_uses_old_28_request_coverage(factorial,tmp_path):
    with pytest.raises(ValueError,match='backend|cleanup'):
        factorial.publish_completed(tmp_path/'result.json',{}, {'cleanup':{'passed':False}}, {}, {})
    matrix={g:{c:{'requests':[{} for _ in range(2)]} for c in factorial.legacy.VARIANTS} for g in ('base','Eonly','Ponly','both')}
    matrix['both']['noise']['requests'].pop()
    with pytest.raises(ValueError,match='56|coverage|requests'):
        factorial.publish_completed(tmp_path/'result.json',{}, {'execution':'diagnostic_eager','cleanup':{'passed':True}}, matrix,{})
    assert not (tmp_path/'result.json').exists()


def full_capture(factorial,group,case='baseline',index=0):
    E,P=factorial.GROUPS[group];record=cells_fixture(factorial)[group];raw=record['tensors'];n=151
    specifications={'prefix.k':((32,1,n,4,128),torch.float32),'prefix.v':((32,1,n,4,128),torch.bfloat16),
        'prefix.pad_mask':((1,n),torch.bool),'prefix.block_mask':((1,n),torch.bool),'prefix.mask_sorted':((1,1,n,n),torch.bool),
        'prefix.cos_sorted':((1,1,n,128),torch.float32),'prefix.sin_sorted':((1,1,n,128),torch.float32),
        'prepared.state':((1,32),torch.bfloat16),'prepared.tokens':((1,64),torch.long),'prepared.token_mask':((1,64),torch.bool),
        'input.noise':((1,40,32),torch.float32),'suffix.mask_sorted':((1,1,41,n+41),torch.bool),
        'suffix.cos':((1,1,41,128),torch.float32),'suffix.sin':((1,1,41,128),torch.float32),'suffix.state_embedding':((1,1,1024),torch.bfloat16)}
    for key,(shape,dtype) in specifications.items():raw[key]=torch.zeros((),dtype=dtype).expand(shape)
    step={'block_calls':32,'attention_calls':32,'prefix_readonly':True,
        'norm':{'calls':129,'original_norm_calls':0 if E else 129,'use_tree':E,'complete':True,'restored':True,
            'by_role':{'torch.bfloat16:1024':65,'torch.float32:128':64}}}
    if E:step['order']={'attention_calls':32,'cleanup':{'passed':True}}
    record['scope'].update(prefill_calls=1,generation=1,prefix_rows=n,n_vis=147,prefix_segments={'idx_ranges':[],'full_ranges':[]},steps=[copy.deepcopy(step) for _ in range(10)])
    record['operator_scope'].update(complete=True,cleanup={'passed':True},merger_calls=1,proj2_calls=1,
        proj1_views=[{}]*3,proj2_views=[{'shares_input_storage':True}]*(3 if P else 1),global_request_id=f'{group}.{case}.r{index}')
    record['group']=group;local='control' if group=='base' else 'candidate';record['request_id']=f'{local}.{case}.r{index}'
    record['scope']['request_id']=record['request_id']
    record['path']=f'{local}/{case}/replay{index}';record['global_path']=f'groups/{group}/'+record['path']
    return record


@pytest.mark.parametrize('bad',['none','velocity_dtype','cache_dtype','state_shape','backend','time','segments','local_identity','path','scope_id'])
def test_complete_capture_admits_typed_actual_geometry_and_rejects_hidden_phase_drift(factorial,bad):
    r=full_capture(factorial,'both')
    if bad=='velocity_dtype':r['tensors']['step0.velocity']=r['tensors']['step0.velocity'].float()
    if bad=='cache_dtype':r['tensors']['prefix.k']=r['tensors']['prefix.k'].to(torch.bfloat16)
    if bad=='state_shape':r['tensors']['prepared.state']=torch.zeros(32,dtype=torch.bfloat16)
    if bad=='backend':r['backend_metadata']['precision']='fp32'
    if bad=='time':r['tensors']['step0.time']=torch.zeros(2)
    if bad=='segments':r['scope']['generation']=2
    if bad=='local_identity':r['request_id']='candidate.noise.r0'
    if bad=='path':r['global_path']='groups/Ponly/candidate/baseline/replay0'
    if bad=='scope_id':r['scope']['request_id']='candidate.state.r0'
    if bad=='none':factorial.validate_capture(r,'both','baseline',0)
    else:
        with pytest.raises(ValueError,match='Factorial|factorial|metadata|geometry|dtype|generation'):
            factorial.validate_capture(r,'both','baseline',0)


def test_complete_56_publisher_writes_fourteen_interactions_before_completed_report(factorial,tmp_path):
    from safetensors.torch import load_file
    matrix={}
    for group in factorial.GROUPS:
        template=full_capture(factorial,group);matrix[group]={}
        for case in factorial.legacy.VARIANTS:
            pair=[]
            for index in (0,1):
                r={**template,'operator_scope':dict(template['operator_scope']),'scope':dict(template['scope'])}
                r['operator_scope']['global_request_id']=f'{group}.{case}.r{index}'
                local='control' if group=='base' else 'candidate';r['request_id']=f'{local}.{case}.r{index}';r['scope']['request_id']=r['request_id']
                r['path']=f'{local}/{case}/replay{index}';r['global_path']=f'groups/{group}/'+r['path']
                pair.append(r)
            matrix[group][case]={'requests':pair,'known_reference':{'raw_reference_exact':True}}
    interactions={c+'.'+key:torch.zeros(1,40,32,dtype=torch.float64) for c in factorial.legacy.VARIANTS for key in ('first_velocity','actions')}
    factorial.publish_completed(tmp_path/'result.json',{'status':'verified'}, {'execution':'diagnostic_eager','cleanup':{'passed':True}},matrix,interactions)
    assert len(load_file(str(tmp_path/'interaction_outputs.safetensors')))==14
    assert json.loads((tmp_path/'result.json').read_text())=={'status':'verified'}


def test_help_and_allocation_guard_before_model_or_reference_reads(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--precondition-report' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--reference-root','/missing','--precondition-report','/missing','--fixtures','/missing','--prior','/missing',
        '--source-revision','a'*40,'--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()
