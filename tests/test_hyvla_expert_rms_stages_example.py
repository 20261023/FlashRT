import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_rms_stages.py'


@pytest.fixture
def rms():
    s=importlib.util.spec_from_file_location('hy_rms_stage_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_literal_bf16_stages_preserve_variance_cast_and_weight_order(rms):
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16);w=torch.tensor([2.,.5],dtype=torch.bfloat16)
    out=rms.rms_stages(x,w)
    assert tuple(out)==('float_input','square','variance','variance_plus_epsilon','inverse_rsqrt','scaled','cast_back','weighted_output')
    torch.testing.assert_close(out['square'],torch.tensor([[1.,4.],[9.,16.]]),atol=0,rtol=0)
    torch.testing.assert_close(out['variance'],torch.tensor([[2.5],[12.5]]),atol=0,rtol=0)
    torch.testing.assert_close(out['weighted_output'],torch.tensor([[1.265625,.6328125],[1.6953125,.56640625]],dtype=torch.bfloat16),atol=0,rtol=0)
    assert out['cast_back'].dtype==out['weighted_output'].dtype==torch.bfloat16
    torch.testing.assert_close(x,torch.tensor([[1.,2.],[3.,4.]],dtype=torch.bfloat16),atol=0,rtol=0)


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_stage_output_exactly_replays_actual_native_source_norm(rms,dtype):
    owner=rms.layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=dtype).transpose(0,1);w=torch.tensor([1.5,.5],dtype=torch.bfloat16)
    actual=owner._rms_norm(x,w);stages=rms.rms_stages(x,w)
    torch.testing.assert_close(stages['weighted_output'],actual,atol=0,rtol=0)
    assert stages['weighted_output'].dtype==dtype


@pytest.mark.parametrize('bad',['nan','overflow','dtype','weight','epsilon'])
def test_stages_reject_unadmitted_dtype_geometry_and_nonfinite(rms,bad):
    x=torch.ones(1,2);w=torch.ones(2,dtype=torch.bfloat16);eps=1e-5
    if bad=='nan':x[0,0]=float('nan')
    if bad=='overflow':x.fill_(1e30)
    if bad=='dtype':x=x.double()
    if bad=='weight':w=w[:1]
    if bad=='epsilon':eps=.01
    with pytest.raises(ValueError,match='RMS|finite|dtype|epsilon|weight'):rms.rms_stages(x,w,epsilon=eps)


def block_fixture(rms):
    from test_hyvla_expert_layers_example import weights,block_args
    owner=rms.layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2,projection='separate')
    x,k,v,mask,cos,sin=block_args()
    return owner,lambda:rms.layers.replay_block(owner,x,weights(),(torch.ones(2,dtype=torch.bfloat16),)*2,k,v,mask,cos,sin,prefix_rows=1)


def test_capture_observes_four_real_norms_without_replacing_block_math(rms):
    owner,run=block_fixture(rms);before=owner._rms_norm;raw={};meta={};expected=run()
    with rms.capture_norms(owner,raw,meta) as state:actual=run()
    assert state['complete'] and state['calls']==4 and len(raw)==12
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert owner._rms_norm is before
    for i in range(4):
        out=rms.rms_stages(raw[f'n{i}.input'],raw[f'n{i}.weight'])
        torch.testing.assert_close(out['weighted_output'],raw[f'n{i}.observed_output'],atol=0,rtol=0)
    assert raw['n0.input'].dtype==torch.bfloat16 and raw['n1.input'].dtype==torch.float32
    assert raw['n2.input'].dtype==torch.float32 and raw['n3.input'].dtype==torch.bfloat16


@pytest.mark.parametrize('failure',['body','count'])
def test_capture_failure_restores_instance_binding_and_never_completes(rms,failure):
    owner,run=block_fixture(rms);before=owner._rms_norm;state={}
    with pytest.raises((RuntimeError,ValueError),match='body|RMS|norm|calls'):
        with rms.capture_norms(owner,{}, {}) as state:
            if failure=='body':raise RuntimeError('body failed')
            owner._rms_norm(torch.ones(1,2),torch.ones(2,dtype=torch.bfloat16))
    assert owner._rms_norm is before and not state.get('complete',False)


def test_current_old_reference_identity_does_not_accept_historical_job_as_current(rms):
    cp={'path':'/ckpt','bytes':1,'mtime_ns':2};sources={'driver':{'path':'/code','bytes':1,'mtime_ns':2}}
    r={'kind':'hy_fixed_input_rms_stages','status':'diagnostic_complete','engine':'official','job_id':'9',
       'source_revision':'a'*40,'checkpoint':cp,'source_records':sources,'probe_ids':list(rms.PROBES),
       'block_oracles_exact':True,'norm_calls':16,'own_stage_replay_exact':True,'torch':'2.7.1+cu128'}
    kw={'job_id':'9','source_revision':'a'*40,'checkpoint':cp,'sources':sources}
    assert rms.admit_reference(r,**kw) is None
    r['job_id']='315024'
    with pytest.raises(ValueError,match='reference|job'):rms.admit_reference(r,**kw)


def test_raw_failure_does_not_publish_final_success(rms,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*a,**k):raise OSError('raw write failure')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write'):rms.publish(tmp_path,{'x':torch.ones(1)},{'status':'complete'})
    assert not (tmp_path/'result.json').exists()


def test_help_and_allocation_guard_before_history_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--history-inputs' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--history-inputs','/missing','--source-revision','a'*40,
        '--output',str(out),'--reference-rms','/missing'],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()


def reference_fixture(rms):
    raw={};metadata={};bf=torch.bfloat16
    probes=('baseline.s0.l1','baseline.s2.l0','state.s0.l1','state.s2.l0')
    def desc(x):return {'shape':list(x.shape),'live_stride':list(x.stride()),'storage_offset':0,'dtype':str(x.dtype)}
    for probe in probes:
        metadata[probe]={}
        for field in ('residual','output'):raw[f'own.{probe}.block.{field}']=torch.zeros(1,41,1024,dtype=bf)
        for n,shape,dtype in ((0,(1,41,1024),bf),(1,(1,16,41,128),torch.float32),(2,(1,4,41,128),torch.float32),(3,(1,41,1024),bf)):
            x=torch.zeros(shape,dtype=dtype);w=torch.ones(shape[-1],dtype=bf);stages=rms.rms_stages(x,w)
            base=f'own.{probe}.n{n}';observed=stages['weighted_output'].clone()
            raw.update({base+'.input':x,base+'.weight':w,base+'.observed_output':observed})
            raw.update({base+'.stage.'+name:value for name,value in stages.items()})
            metadata[probe][f'n{n}']={'input':desc(x),'weight':desc(w),'output':desc(observed)}
    report={'kind':'hy_fixed_input_rms_stages','status':'diagnostic_complete','engine':'official','job_id':'9',
        'source_revision':'a'*40,'checkpoint':{'path':'/ckpt','bytes':1,'mtime_ns':2},'source_records':{'driver':{'path':'/code','bytes':1,'mtime_ns':2}},
        'probe_ids':list(probes),'block_oracles_exact':True,'norm_calls':16,'own_stage_replay_exact':True,'torch':'2.7.1+cu128','norm_metadata':metadata}
    return raw,report


@pytest.mark.parametrize('bad',['none','missing_observed','wrong_observed','dtype','shape','finite','metadata','overlap','huge_offset'])
def test_reference_raw_roles_and_actual_endpoints_are_checked_before_owner(rms,bad):
    raw,r=reference_fixture(rms);base='own.baseline.s0.l1.n0';called=[]
    if bad=='missing_observed':del raw[base+'.observed_output'];raw['dummy']=torch.zeros(1)
    if bad=='wrong_observed':raw[base+'.observed_output']=raw[base+'.observed_output']+1
    if bad=='dtype':raw['own.baseline.s0.l1.n1.input']=raw['own.baseline.s0.l1.n1.input'].bfloat16()
    if bad=='shape':raw[base+'.weight']=raw[base+'.weight'][:1]
    if bad=='finite':raw[base+'.input']=raw[base+'.input'].clone().fill_(float('nan'))
    if bad=='metadata':r['norm_metadata']['baseline.s0.l1']['n0']['input']['dtype']='torch.float32'
    if bad=='overlap':r['norm_metadata']['baseline.s0.l1']['n1']['input']['live_stride']=[0,0,0,0]
    if bad=='huge_offset':r['norm_metadata']['baseline.s0.l1']['n0']['input']['storage_offset']=10**12
    assert len(raw)==184
    def factory():called.append(True);return 'owner'
    call=lambda:rms.admitted_reference_owner(r,raw,job_id='9',source_revision='a'*40,checkpoint=r['checkpoint'],sources=r['source_records'],factory=factory)
    if bad=='none':assert call()=='owner' and called==[True]
    else:
        with pytest.raises(ValueError,match='reference|RMS|role|endpoint|layout|metadata'):call()
        assert not called


@pytest.mark.parametrize('different',[False,True])
def test_stage_summary_does_not_invent_a_runtime_difference(rms,different):
    result=rms.stage_comparison_summary({'a':{'exact':True},'b':{'exact':not different}})
    assert result['comparison_count']==2 and result['differing_stage_count']==int(different)
    assert result['runtime_stage_difference_observed'] is different
