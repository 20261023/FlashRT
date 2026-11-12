import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_mean_tree.py'


@pytest.fixture
def expert():
    s=importlib.util.spec_from_file_location('hy_expert_tree_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.mark.parametrize('dtype,value',[(torch.bfloat16,2.),(torch.float32,1.9999899864196777)])
def test_live_input_norm_keeps_original_cast_then_weight_order_without_native_mean(expert,dtype,value,monkeypatch):
    x=torch.ones(1,128,dtype=dtype);w=torch.full((128,),2.,dtype=torch.bfloat16);before=x.clone()
    def forbidden(*a,**k):raise AssertionError('native mean fallback')
    monkeypatch.setattr(torch.Tensor,'mean',forbidden)
    out=expert.tree_rms(x,w)
    assert out.dtype==dtype and torch.equal(out,torch.full((1,128),value,dtype=dtype)) and torch.equal(x,before)


@pytest.mark.parametrize('bad',['dtype','weight','epsilon','shape','nonfinite'])
def test_rms_rejects_unsupported_input_instead_of_silent_native_fallback(expert,bad):
    x=torch.ones(1,128);w=torch.ones(128,dtype=torch.bfloat16);eps=1e-5
    if bad=='dtype':x=x.double()
    if bad=='weight':w=w.float()
    if bad=='epsilon':eps=.01
    if bad=='shape':w=w[:1]
    if bad=='nonfinite':x[0,0]=float('nan')
    with pytest.raises(ValueError,match='RMS|finite|dtype|epsilon|weight'):expert.tree_rms(x,w,epsilon=eps)


@pytest.mark.parametrize('use_tree',[False,True])
def test_norm_scope_counts_actual_roles_and_restores_original(expert,use_tree):
    calls=[]
    def original(x,w):calls.append(True);return x
    owner=SimpleNamespace(_rms_norm=original)
    with expert.norm_scope(owner,use_tree=use_tree,expected_calls=2) as state:
        owner._rms_norm(torch.ones(1,1024,dtype=torch.bfloat16),torch.ones(1024,dtype=torch.bfloat16))
        owner._rms_norm(torch.ones(1,128),torch.ones(128,dtype=torch.bfloat16))
    assert owner._rms_norm is original and state['complete'] and state['restored'] and state['calls']==2
    assert state['by_role']=={'torch.bfloat16:1024':1,'torch.float32:128':1}
    assert len(calls)==(0 if use_tree else 2) and state['original_norm_calls']==len(calls)


@pytest.mark.parametrize('bad',['body','missing','extra'])
def test_scope_failures_never_leave_standalone_owner_modified(expert,bad):
    original=lambda x,w:x;owner=SimpleNamespace(_rms_norm=original)
    with pytest.raises((ValueError,RuntimeError),match='body|count|extra|calls'):
        with expert.norm_scope(owner,use_tree=True,expected_calls=1) as state:
            if bad=='body':raise RuntimeError('body failure')
            if bad=='extra':
                for _ in range(2):owner._rms_norm(torch.ones(1,128),torch.ones(128,dtype=torch.bfloat16))
    assert owner._rms_norm is original and state['restored'] and not state['complete']


def support_fixture(expert):
    from test_hyvla_mean_tree_example import raw_fixture
    raw,capture=raw_fixture(expert.mt);old={}
    for p in expert.rms.PROBES:
        for n in range(4):old[f'own.{p}.n{n}.stage.variance']=capture[f'{p}.n{n}.unprofiled_variance'].clone()
    r={'kind':'hy_source_explicit_mean_tree_diagnostic','status':'diagnostic_complete','job_id':'316521',
        'source_revision':'19e87bcccf875449e4d52fc4fa67566de8b19257','golden_variance_used_as_input':False,
        'raw_proof':{'tensors':96,'native_baselines_exact':16,'squares_exact':16},
        'summary':{'source_old_exact':16,'source_native_exact':16},'model_constructed':False,'checkpoint_tensors_read':0}
    return r,raw,old,capture


@pytest.mark.parametrize('bad',['none','job','dummy','old_mean','native_mean','golden'])
def test_local_support_validates_actual_selected_means_before_owner_not_header_counts(expert,bad):
    report,raw,old,capture=support_fixture(expert);called=[]
    if bad=='job':report['job_id']='316520'
    if bad=='dummy':del raw['baseline.s0.l1.n0.vector_ascending'];raw['dummy']=torch.zeros(1)
    if bad=='old_mean':raw['baseline.s0.l1.n0.vector_ascending']+=1
    if bad=='native_mean':raw['baseline.s0.l1.n0.vector_descending']+=1
    if bad=='golden':report['golden_variance_used_as_input']=True
    def factory():called.append(True);return 'owner'
    call=lambda:expert.admit_support(report,raw,old,capture,factory=factory)
    if bad=='none':assert call()=='owner' and called==[True]
    else:
        with pytest.raises(ValueError,match='support|raw|R31|mean'):call()
        assert not called


def full_outputs(expert):
    h=torch.zeros(1,41,1024,dtype=torch.bfloat16);v=torch.zeros(1,40,32,dtype=torch.bfloat16)
    return {k:h if len(shape)==3 and shape[-1]==1024 else v for k,shape in expert.order.expected_shapes().items()}


@pytest.mark.parametrize('bad',['none','dummy','changed','dtype'])
def test_all1300_native_control_roles_are_a_hard_gate(expert,bad):
    ref=full_outputs(expert);current=dict(ref)
    if bad=='dummy':del current['gold_head.step9.velocity'];current['dummy']=ref['gold_head.step9.velocity']
    if bad=='changed':current['propagated.step9.layer31.output']=torch.ones_like(ref['propagated.step9.layer31.output'])
    if bad=='dtype':current['teacher.step0.layer0.residual']=ref['teacher.step0.layer0.residual'].float()
    if bad=='none':assert expert.baseline_gate(current,ref)['tensors']==1300
    else:
        with pytest.raises(ValueError,match='raw|baseline|control|tensor|BF16'):expert.baseline_gate(current,ref)


def test_changed_gold_head_is_reported_as_outcome_not_an_unchanged_negative_control(expert):
    ref=full_outputs(expert);current=dict(ref);current['gold_head.step9.velocity']=torch.ones_like(ref['gold_head.step9.velocity'])
    metrics,summary=expert.compare_outputs(current,ref)
    assert len(metrics)==1300 and summary['exact_tensors']==1299
    assert summary['gold_head_exact_steps']==9 and summary['velocity_exact_steps']==10
    assert summary['boundary_exact']['propagated.output']==320


def test_help_and_allocation_guard_precede_input_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--support-root' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--history-root','/missing','--support-root','/missing','--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
