import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_order_replay.py'
QKV_SOURCE='eddf597d58d29653bf78233b92c8540a2d623d9e'
GOLD_SOURCE='fe66081a2ace3501fe2a51216eac1b4e47f40267'


@pytest.fixture
def order():
    s=importlib.util.spec_from_file_location('hy_order_driver_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def raw_arrays(order):
    a=torch.zeros(1,41,1024,dtype=torch.bfloat16);v=torch.zeros(1,40,32,dtype=torch.bfloat16)
    return {name:(v if '.velocity' in name else a) for name in order.expected_shapes()}


@pytest.mark.parametrize('bad',['none','missing','dtype','shape','finite','extra'])
def test_complete_raw_contract_rejects_partial_or_changed_representations(order,bad):
    arrays=raw_arrays(order);name='teacher.step0.layer0.residual'
    if bad=='missing':del arrays[name]
    if bad=='dtype':arrays[name]=arrays[name].float()
    if bad=='shape':arrays[name]=arrays[name][:,:,:1]
    if bad=='finite':arrays[name]=arrays[name].clone().fill_(float('nan'))
    if bad=='extra':arrays['unknown']=torch.ones(1)
    if bad=='none':assert order.validate_raw(arrays)['tensors']==1300
    else:
        with pytest.raises(ValueError,match='raw|tensor|BF16|shape|finite'):order.validate_raw(arrays)


def history_report(order):
    return {'kind':'recorded_Expert_layer_local_and_propagated_replay_not_policy_gate','status':'diagnostic_complete',
        'engine':'native','variant':'state','job_id':'314589','source_revision':QKV_SOURCE,'projection':'separate',
        'runtime':{'torch':'2.11'},'checkpoint':{'path':'/checkpoint/model.safetensors','bytes':1,'mtime_ns':2},
        'selected_checkpoint_keys':order.layers.all_selected_keys(),'weights_loaded':363,'model_constructed':False,'gold_h0_exact_steps':10,
        'reference_producer':{'job_id':'314485','source_revision':GOLD_SOURCE},
        'historical_native_baseline':{'required':False,'passed':True},
        'projection_control':{'job_id':'314589','source_revision':QKV_SOURCE,'path':'/results/native-fused/state','fused1300exact':True}}


@pytest.mark.parametrize('bad',['none','projection','runtime','case','keys','headproof','legacy_order'])
def test_historical_split_role_is_not_fused_or_implicitly_current(order,bad):
    r=history_report(order);kw={'job_id':'314589','source_revision':QKV_SOURCE,'engine':'native','variant':'state',
        'checkpoint':r['checkpoint'].copy(),'runtime':r['runtime'].copy(),'gold_job':'314485','gold_source':GOLD_SOURCE,
        'fused_path':Path('/results/native-fused/state')}
    if bad=='projection':r['projection']='fused'
    if bad=='runtime':r['runtime']={'torch':'2.7'}
    if bad=='case':r['variant']='baseline'
    if bad=='keys':r['selected_checkpoint_keys'][0]='wrong'
    if bad=='headproof':r['projection_control']['fused1300exact']=False
    if bad=='legacy_order':r['source_revision']=kw['source_revision']='a'*40
    if bad=='none':assert order.admit_historical_split(r,**kw) is None
    else:
        with pytest.raises(ValueError,match='split|history|source|role'):order.admit_historical_split(r,**kw)


def test_current_sorted_control_requires_real_new_identity_and_source_records(order):
    r=history_report(order);sources={k:{'path':'/'+k,'bytes':1,'mtime_ns':2} for k in ('driver','core','native','adapter','suffix')}
    inputs={'manifest':{'path':'/inputs.json','bytes':1,'mtime_ns':2}}
    r.update(kind='hy_split_projection_order_replay',status='sorted_split_reproduced',job_id='9',source_revision='a'*40,
        attention_order='sorted',attention_calls=640,source_records=sources,
        historical_split_sorted={'exact':True,'tensors':1300,'job_id':'314589','source_revision':QKV_SOURCE},
        adapter_preflight={'exact':True,'steps':10},artifact_complete=True,
        input_records={'manifest':inputs['manifest'].copy()})
    kw={'job_id':'9','source_revision':'a'*40,'engine':'native','variant':'state','checkpoint':r['checkpoint'],
        'runtime':r['runtime'],'sources':sources,'input_records':inputs,'history_job':'314589','history_source':QKV_SOURCE}
    assert order.admit_current_sorted(r,**kw) is None
    r['attention_order']='original'
    with pytest.raises(ValueError,match='sorted|control'):order.admit_current_sorted(r,**kw)
    r['attention_order']='sorted';r['source_revision']='b'*40
    with pytest.raises(ValueError,match='sorted|control'):order.admit_current_sorted(r,**kw)
    r['source_revision']='a'*40;r['input_records']['manifest']['mtime_ns']=3
    with pytest.raises(ValueError,match='sorted|control'):order.admit_current_sorted(r,**kw)


def test_sorted_gate_cannot_pass_flags_when_actual_array_differs(order):
    historical=raw_arrays(order);current={k:v for k,v in historical.items()}
    assert order.sorted_gate(current,historical)['exact']
    key='propagated.step9.velocity';current[key]=current[key].clone();current[key][0,0,0]=1
    assert order.sorted_gate(current,historical)['exact'] is False


def test_current_bad_second_case_stops_before_alternate_compilation(order):
    historical=raw_arrays(order);wrong={k:v for k,v in historical.items()}
    key='teacher.step0.layer0.output';wrong[key]=wrong[key]+1;called=[]
    def compile_owner():called.append(True);raise AssertionError('unadmitted owner compiled')
    with pytest.raises(ValueError,match='control|tensor|replay'):
        order.admitted_owner({'baseline':historical,'state':wrong},{'baseline':historical,'state':historical},compile_owner)
    assert not called


def test_preflight_raw_failure_does_not_publish_final_success(order,tmp_path,monkeypatch):
    import safetensors.torch
    real=safetensors.torch.save_file
    def fail(tensors,path,*a,**k):
        if Path(path).name=='adapter_preflight.safetensors':raise OSError('preflight write failed')
        return real(tensors,path,*a,**k)
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='preflight write'):
        order.publish_case(tmp_path,{'x':torch.ones(1)},{'context':torch.ones(1)},{'status':'sorted_split_reproduced'})
    assert not (tmp_path/'result.json').exists()


def test_adapter_preflight_replays_nonzero_gold_context_before_weights(order):
    owner=order.layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2,projection='separate')
    q=torch.zeros(1,2,2,2);q[:,0,:,0]=math.sqrt(2);q=q.transpose(1,2)
    k=torch.tensor([[[[0.,0.]],[[math.log(3),0.]],[[10.,0.]],[[0.,0.]],[[-10.,0.]]]])
    v=torch.tensor([[[[2.,0.]],[[6.,0.]],[[100.,0.]],[[2.,0.]],[[-100.,0.]]]],dtype=torch.bfloat16)
    mask=torch.tensor([[[True,True,False,False,False],[False,True,False,True,False]]])
    base='attention.step0.layer0.'
    gold={base+'q':q.clone(),base+'k':k,base+'v':v,base+'mask':mask,
          base+'output':torch.tensor([[[5.,0.,4.,0.],[5.,0.,4.,0.]]],dtype=torch.bfloat16)}
    meta={base+'q':{'shape':list(q.shape),'live_stride':list(q.stride()),'storage_offset':0,'dtype':str(q.dtype)}}
    arrays,proof=order.adapter_preflight(owner,gold,meta,np.array([2,0,1]),num_steps=1,device='cpu')
    assert proof['exact'] and proof['steps']==1 and proof['weights_loaded']==0
    torch.testing.assert_close(arrays['step0.context'],gold[base+'output'],atol=0,rtol=0)
    gold[base+'output']=torch.zeros_like(gold[base+'output'])
    with pytest.raises(ValueError,match='preflight|context'):order.adapter_preflight(owner,gold,meta,np.array([2,0,1]),num_steps=1,device='cpu')


def test_fixed_manifest_rejects_cross_job_output_or_path_escape(order,tmp_path):
    r=tmp_path/'results';snap=tmp_path/'snapshots';current=r/'new9'
    ref=lambda name,job,src:{'root':str(r/name),'job_id':job,'source_revision':src}
    m={'schema_version':1,'job_id':'9','source_revision':'a'*40,'current_root':str(current),'snapshots_root':str(snap),
       'checkpoint':str(tmp_path/'checkpoint'),'official_attention_source':str(snap/'round007/official.py'),
       'gold':ref('gold','314485',GOLD_SOURCE),'qkv':ref('qkv','314589',QKV_SOURCE),'native':ref('native','313766','b'*40)}
    kw={'job_id':'9','source_revision':'a'*40,'engine':'native','arm':'sorted','output':current/'native-sorted'}
    assert order.validate_inputs(m,**kw) is None
    with pytest.raises(ValueError,match='output|manifest'):order.validate_inputs(m,**{**kw,'output':current/'native-original'})
    m['qkv']['root']=str(tmp_path/'outside-results')
    with pytest.raises(ValueError,match='path|root|manifest'):order.validate_inputs(m,**kw)


def test_order_driver_help_and_allocation_guard_before_manifest_access(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--arm' in r.stdout and '--inputs' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--arm','original','--inputs','/missing',
        '--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
