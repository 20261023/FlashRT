import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/merger_stages.py'


@pytest.fixture
def merger():
    s=importlib.util.spec_from_file_location('hy_merger_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def weights():
    bf=torch.bfloat16
    return {'proj1.weight':torch.zeros(2,1,dtype=bf),'proj1.bias':torch.full((2,),2.,dtype=bf),
        'proj2.weight':torch.eye(2,dtype=bf),'proj2.bias':torch.zeros(2,dtype=bf),
        'pooler.predictor.0.weight':torch.zeros(2,4,dtype=bf),'pooler.predictor.0.bias':torch.zeros(2,dtype=bf),
        'pooler.predictor.2.weight':torch.zeros(2,2,dtype=bf),'pooler.predictor.2.bias':torch.zeros(2,dtype=bf)}


@pytest.fixture
def official_source():
    root=os.environ.get('HYVLA_OFFICIAL_SOURCE')
    if not root:pytest.skip('Version-bound official source supplied by HYVLA_OFFICIAL_SOURCE')
    path=Path(root)/'hy_vla/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    assert path.is_file();return path


def test_literal_bf16_fourteen_stages_preserve_pooling_and_gelu_order(merger):
    x=torch.ones(3,2,2,1,dtype=torch.bfloat16);stages,layouts=merger.staged_merger(x,weights())
    expected={'input':((3,2,2,1),1.),'proj1':((3,2,2,2),2.),'new_x':((3,1,1,4,2),2.),
        'pooled':((3,1,1,4,2),2.),'fused':((3,1,1,4,4),2.),'predictor0':((3,1,1,4,2),0.),
        'predictor_activation':((3,1,1,4,2),0.),'score':((3,1,1,4,2),0.),
        'probabilities':((3,1,1,4,2),.25),'products':((3,1,1,4,2),.5),
        'pooled_sum':((3,1,1,2),2.),'activation':((3,1,1,2),1.953125),
        'proj2':((3,1,1,2),1.953125),'merged':((3,1,2),1.953125)}
    assert set(stages)==set(expected)
    for name,(shape,value) in expected.items():
        assert stages[name].dtype==torch.bfloat16 and torch.equal(stages[name],torch.full(shape,value,dtype=torch.bfloat16))
        assert layouts[name]['shape']==list(shape) and layouts[name]['dtype']=='torch.bfloat16'
    assert layouts['pooled']['live_stride'][-2]==0


def test_stages_reproduce_unchanged_actual_native_hwd_api_on_nonuniform_input(merger):
    w=weights();w['proj1.weight'].fill_(.25)
    x=torch.arange(12,dtype=torch.bfloat16).reshape(3,4,1);before=x.clone()
    raw,layout=merger.observe_merger(x,w,form='native_hwd',grid=2)
    owner=merger.native_owner(w)
    actual=merger.ablation.camera_merger(owner,x,grid=2)
    assert set(raw)=={'actual_endpoint',*merger.STAGES} and torch.equal(raw['merged'],actual)
    assert torch.equal(raw['actual_endpoint'],actual) and torch.equal(x,before)
    assert layout['actual_input_views'][0]['shape']==[2,2,1]


def test_official_ast_api_calls_three_cameras_and_records_preconcat_layouts(merger,official_source):
    w=weights();w['proj1.weight'].fill_(.25)
    inputs=tuple(torch.full((1,2,2,1),v,dtype=torch.bfloat16) for v in (1.,2.,3.))
    owner=merger.official_owner(official_source,w);seen=[]
    real=owner.proj1.forward
    def record(x):seen.append(tuple(x.shape));return real(x)
    owner.proj1.forward=record
    raw,layout=merger.observe_merger(inputs,w,form='official',grid=2,official=owner)
    assert seen==[(1,2,2,1)]*3
    assert torch.equal(raw['actual_endpoint'],raw['merged'])
    assert len(layout['per_camera'])==3 and layout['concatenation_is_artifact_only'] is True
    assert all(r['input']['shape']==[1,2,2,1] for r in layout['per_camera'])


def test_own_actual_endpoint_is_hard_gate_not_a_self_report(merger,official_source):
    w=weights();owner=merger.official_owner(official_source,w);original=owner.forward
    owner.forward=lambda *a,**kw:original(*a,**kw)+1
    with pytest.raises(ValueError,match='endpoint|Merger'):
        merger.observe_merger((torch.ones(1,2,2,1,dtype=torch.bfloat16),)*3,w,form='official',grid=2,official=owner)


@pytest.mark.parametrize('bad',['dtype','geometry','finite','missing_weight','weight_dtype','weight_shape','form','input_list'])
def test_unsupported_merger_inputs_never_fallback_or_change_precision(merger,bad):
    w=weights();x=torch.ones(3,4,1,dtype=torch.bfloat16);form='native_hwd'
    if bad=='dtype':x=x.float()
    if bad=='geometry':x=x[:,:3]
    if bad=='finite':x[0,0,0]=float('nan')
    if bad=='missing_weight':w.pop('proj2.bias')
    if bad=='weight_dtype':w['proj1.bias']=w['proj1.bias'].float()
    if bad=='weight_shape':w['pooler.predictor.0.weight']=torch.ones(2,5,dtype=torch.bfloat16)
    if bad=='form':form='batched'
    if bad=='input_list':x=[x]
    with pytest.raises(ValueError,match='Merger|BF16|weight|geometry|form|Tensor'):
        merger.observe_merger(x,w,form=form,grid=2)


def test_ambiguous_official_method_source_rejects_before_execution(merger,tmp_path):
    src=tmp_path/'duplicate.py'
    src.write_text('class _HYViT2MLPProjector:\n def forward(self,x): return x\n def forward(self,x): return x\n')
    with pytest.raises(ValueError,match='method|Merger'):merger.official_owner(src,weights())


@pytest.mark.parametrize('bad',['none','dummy','dtype','nonfinite','endpoint'])
def test_complete_sixty_raw_roles_with_actual_endpoints_are_required(merger,bad):
    one,_=merger.observe_merger(torch.ones(3,4,1,dtype=torch.bfloat16),weights(),form='native_hwd',grid=2)
    raw={origin+'.'+form+'.'+name:value.clone() for origin in ('gold','native') for form in ('official','native_hwd') for name,value in one.items()}
    if bad=='dummy':del raw['gold.official.actual_endpoint'];raw['dummy']=torch.zeros(1)
    if bad=='dtype':raw['gold.native_hwd.score']=raw['gold.native_hwd.score'].float()
    if bad=='nonfinite':raw['native.official.proj1'].fill_(float('nan'))
    if bad=='endpoint':raw['native.native_hwd.actual_endpoint']+=1
    if bad=='none':assert merger.validate_raw(raw,batch=3,grid=2,in_channels=1,out_channels=2)['tensors']==60
    else:
        with pytest.raises(ValueError,match='raw|endpoint|role'):merger.validate_raw(raw,batch=3,grid=2,in_channels=1,out_channels=2)


def test_first_difference_is_measured_and_preserves_input_confound(merger):
    a,_=merger.staged_merger(torch.ones(3,2,2,1,dtype=torch.bfloat16),weights())
    b={k:v.clone() for k,v in a.items()};b['input']+=1;b['merged']+=1
    result=merger.compare_stages(a,b)
    assert result['first_different_stage']=='input' and result['stages']['proj1']['exact']
    assert result['stages']['merged']['unequal_elements']==6


def test_historical_zero_offset_layout_reconstruction_does_not_invent_old_offset(merger):
    values=[torch.arange(4,dtype=torch.bfloat16).reshape(1,2,2,1)+i for i in range(3)]
    meta={'shape':[3,4,1],'live_stride':[4,1,4],'dtype':'torch.bfloat16'}
    official,native,proof=merger.rebuild_inputs(values,[[4,2,1,1]]*3,meta,device='cpu')
    assert len(official)==3 and native.shape==(3,4,1) and native.stride()==(4,1,4)
    assert torch.equal(native,torch.cat(values).reshape(3,4,1)) and proof['storage_offset_assumption']==0
    assert proof['historical_storage_offset_observed'] is False
    with pytest.raises(ValueError,match='layout|span|stride'):
        merger.rebuild_inputs(values,[[10**12,10**12,10**12,1]]*3,meta,device='cpu')


def test_raw_failure_never_publishes_success_report(merger,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*a,**kw):raise OSError('raw failure')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw failure'):merger.publish(tmp_path,{'x':torch.ones(1)},{'status':'complete'})
    assert not (tmp_path/'result.json').exists()


@pytest.mark.parametrize('bad',['none','missing','wrong_path','stat','source'])
def test_manifest_binds_all_actual_paths_and_stats_before_weight_reader(merger,tmp_path,bad):
    roles={}
    for origin,job,engine in (('gold','hy-trace-308690','official'),('native','hy-folded-state-trace-313766','native'),
                            ('current','hy-native-policy-tree-316975','native/control')):
        for case in ('baseline','state'):
            for field,filename in (('report','result.json'),('raw','trace.safetensors')):
                folder=tmp_path/'results'/job/engine/case
                if origin=='current':folder=folder/'replay0'
                roles[f'{origin}.{case}.{field}']=folder/filename
    roles.update({'r33':tmp_path/'results/hy-native-policy-tree-316975/native/result.json',
        'runtime.official':tmp_path/'results/hy-rms-stages-315256/official/result.json',
        'runtime.native':tmp_path/'results/hy-rms-stages-315256/native/result.json'})
    for file in roles.values():file.parent.mkdir(parents=True,exist_ok=True);file.write_text('{}')
    source=tmp_path/'snapshots/round007/diagnostic/official/hy_vla/hunyuan_vl_mot/modeling_hunyuan_vl_mot.py'
    source.parent.mkdir(parents=True);source.write_text('# revision-bound source')
    ck=tmp_path/'checkpoint/model.safetensors';ck.parent.mkdir();ck.write_bytes(b'no tensor data')
    manifest={'kind':'hy_frozen_complete_merger_inputs','task_root':str(tmp_path),'checkpoint':merger.file_record(ck),
        'official_source':merger.file_record(source),'records':{k:merger.file_record(v) for k,v in roles.items()}}
    called=[]
    if bad=='missing':manifest['records'].pop('native.state.raw')
    if bad=='wrong_path':manifest['records']['gold.baseline.raw']=manifest['records']['gold.state.raw']
    if bad=='stat':roles['gold.baseline.raw'].write_text('changed')
    if bad=='source':source.write_text('changed')
    def read():called.append(True);return 'no actual weights'
    call=lambda:merger.admit_file_records(manifest,factory=read)
    if bad=='none':assert call()=='no actual weights' and called==[True]
    else:
        with pytest.raises(ValueError,match='record|source|path|role|stat'):call()
        assert not called


@pytest.mark.parametrize('engine,which',[('official','gold'),('native','native')])
def test_runtime_specific_historical_endpoint_anchor_is_a_hard_gate(merger,engine,which):
    one,_=merger.observe_merger(torch.ones(3,4,1,dtype=torch.bfloat16),weights(),form='native_hwd',grid=2)
    raw={o+'.'+f+'.'+k:v.clone() for o in ('gold','native') for f in ('official','native_hwd') for k,v in one.items()}
    gold=one['actual_endpoint'].clone();native=gold.clone();bridge=native.clone()
    assert merger.admit_endpoints(raw,engine=engine,gold=gold,native=native,current_bridge=bridge)['historical_anchor_exact']
    raw[which+'.'+('official' if engine=='official' else 'native_hwd')+'.actual_endpoint']+=1
    with pytest.raises(ValueError,match='anchor|endpoint'):merger.admit_endpoints(raw,engine=engine,gold=gold,native=native,current_bridge=bridge)


@pytest.mark.parametrize('bad',['none','gold_oracle','gold_load','native_job','current_source','current_enabled','checkpoint','runtime'])
def test_historical_header_roles_reject_before_any_checkpoint_tensor_read(merger,bad):
    from test_hyvla_native_policy_tree_example import legacy_header_fixture
    native,directory,ck,runtime=legacy_header_fixture()
    native['torch']='2.11.0+cu128'
    gold={'engine':'official','variant':'baseline','job_id':'308690','oracle':{'exact':True},
        'load_report':{'missing_inference_keys':[],'load_unexpected_keys':[],'load_missing_keys':['model.dual_tower.expert.lm_head.weight']},
        'torch':'2.7.1+cu128'}
    current={'status':'request_captured_pending_backend_cleanup','request_id':'control.baseline.r0',
        'scope':{'enabled':False,'complete':True,'cleanup':{'passed':True},'nfe':10,'prefill_calls':1}}
    r33={'kind':'hy_native_policy_expert_phase_source_tree_diagnostic','status':'diagnostic_complete','job_id':'316975',
        'source_revision':'54a661d52a7b3a9d4619eb4f3639ff5ae7f29e91','runtime':runtime,'checkpoint':directory,
        'precondition':{'checkpoint':ck},'backend_cleanup':{'passed':True},'original_padded_gate_passes':{'control':6,'candidate':6}}
    if bad=='gold_oracle':gold['oracle']['exact']=False
    if bad=='gold_load':gold['load_report']['missing_inference_keys']=['needed']
    if bad=='native_job':native['job_id']='other'
    if bad=='current_source':r33['source_revision']='a'*40
    if bad=='current_enabled':current['scope']['enabled']=True
    if bad=='checkpoint':r33['checkpoint']=dict(directory,path='/other')
    if bad=='runtime':native['runtime']={'torch':'other'}
    call=lambda:merger.admit_history_headers(gold,native,current,r33,checkpoint=ck,variant='baseline')
    if bad=='none':assert call()['historical_input_job']=='313766'
    else:
        with pytest.raises(ValueError,match='Merger|baseline|runtime|checkpoint|header'):call()


def test_driver_performs_all_four_nonuniform_combinations_and_preserves_negative_metrics(merger,official_source):
    values=[torch.full((1,2,2,1),float(i),dtype=torch.bfloat16) for i in range(3)]
    native_values=[v+1 for v in values];meta={'shape':[3,4,1],'live_stride':[4,1,4],'dtype':'torch.bfloat16'}
    w=weights();w['proj1.weight'].fill_(.25)
    arrays,details,contrasts=merger.run_combinations({'gold':values,'native':native_values},[[4,2,1,1]]*3,meta,w,
        official=merger.official_owner(official_source,w),device='cpu',grid=2)
    assert len(arrays)==60 and len(details)==4
    assert contrasts['origins.official']['first_different_stage']=='input'
    assert contrasts['origins.native_hwd']['first_different_stage']=='input'
    assert not contrasts['origins.official']['stages']['merged']['exact']


def test_help_and_allocation_guard_precede_any_weight_or_input_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--inputs' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--inputs','/missing','--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
