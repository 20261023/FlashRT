import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_rms_handoff.py'


@pytest.fixture
def handoff():
    spec=importlib.util.spec_from_file_location('hy_handoff_test',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('dtype',[torch.bfloat16,torch.float32])
def test_inverse_cut_uses_supplied_inverse_without_recomputing_upstream(handoff,dtype,monkeypatch):
    x=torch.tensor([[1.,2.],[3.,4.]],dtype=dtype);w=torch.tensor([2.,.5],dtype=torch.bfloat16)
    value=torch.tensor([[.5],[.25]])
    def forbidden(*a,**k):raise AssertionError('upstream rsqrt executed')
    monkeypatch.setattr(torch,'rsqrt',forbidden)
    result=handoff.resume_rms(x,w,'inverse_rsqrt',value)
    torch.testing.assert_close(result['weighted_output'],torch.tensor([[1.,.5],[1.5,.5]],dtype=dtype),atol=0,rtol=0)
    assert tuple(result)==('inverse_rsqrt','scaled','cast_back','weighted_output')
    assert torch.equal(value,torch.tensor([[.5],[.25]])) and torch.equal(x,torch.tensor([[1.,2.],[3.,4.]],dtype=dtype))


@pytest.mark.parametrize('cut,value',[
    ('variance',[[4.-1e-5]]),('variance_plus_epsilon',[[4.]]),
    ('scaled',[[.5,1.]]),('cast_back',[[.5,1.]])])
def test_each_cut_resumes_at_exact_injected_value(handoff,cut,value):
    x=torch.tensor([[1.,2.]],dtype=torch.bfloat16);w=torch.tensor([2.,.5],dtype=torch.bfloat16)
    v=torch.tensor(value,dtype=torch.bfloat16 if cut=='cast_back' else torch.float32)
    before=v.clone();result=handoff.resume_rms(x,w,cut,v)
    torch.testing.assert_close(result['weighted_output'],torch.tensor([[1.,.5]],dtype=torch.bfloat16),atol=0,rtol=0)
    assert next(iter(result))==cut and torch.equal(result[cut],before) and torch.equal(v,before)


@pytest.mark.parametrize('bad',['cut','shape','dtype','finite','epsilon','weight','negative'])
def test_invalid_handoff_never_silently_broadcasts_or_changes_precision(handoff,bad):
    x=torch.ones(1,2,dtype=torch.bfloat16);w=torch.ones(2,dtype=torch.bfloat16)
    cut='variance';v=torch.ones(1,1);eps=1e-5
    if bad=='cut':cut='square'
    if bad=='shape':v=torch.ones(2)
    if bad=='dtype':v=v.double()
    if bad=='finite':v.fill_(float('nan'))
    if bad=='epsilon':eps=.01
    if bad=='weight':w=w.float()
    if bad=='negative':v.fill_(-1)
    with pytest.raises(ValueError,match='RMS|cut|finite|epsilon|dtype|shape'):
        handoff.resume_rms(x,w,cut,v,epsilon=eps)


def history_fixture(handoff):
    from test_hyvla_expert_rms_stages_example import reference_fixture
    old,ro=reference_fixture(handoff.rms)
    ro.update(epsilon=1e-5,stages=list(handoff.rms.STAGES),runtime={'torch':'2.7.1+cu128'},bf16_reduced_precision_reduction=True)
    new={k:v.clone() for k,v in old.items()};rn=copy.deepcopy(ro)
    rn.update(engine='native',torch='2.11.0+cu128',runtime={'torch':'2.11.0+cu128'},
        current_old_reference={'job_id':'9','source_revision':'a'*40,'path':'/results/official'},fixed_old_layouts={})
    for p in ('baseline.s0.l1','baseline.s2.l0','state.s0.l1','state.s2.l0'):
        for n in range(4):
            own=f'own.{p}.n{n}';fixed=f'fixed_old.{p}.n{n}'
            for suffix in ('observed_output',*('stage.'+s for s in handoff.rms.STAGES)):
                new[fixed+'.'+suffix]=old[own+'.'+suffix].clone()
            meta=ro['norm_metadata'][p][f'n{n}']
            rn['fixed_old_layouts'][fixed]={'input':meta['input'],'weight':meta['weight']}
    return old,ro,new,rn


@pytest.mark.parametrize('bad',['none','oldjob','newsource','dummy','actual','fixed_actual','fixed_dtype','fixed_square','fixed_layout','missing_fixed','epsilon'])
def test_admit_history_binds_full_raw_evidence_not_counts_or_flags(handoff,bad):
    old,ro,new,rn=history_fixture(handoff);base='fixed_old.baseline.s0.l1.n0'
    if bad=='oldjob':ro['job_id']='8'
    if bad=='newsource':rn['source_revision']='b'*40
    if bad=='dummy':del old['own.baseline.s0.l1.n0.observed_output'];old['dummy']=torch.zeros(1)
    if bad=='actual':old['own.baseline.s0.l1.n0.observed_output']+=1
    if bad=='fixed_actual':new[base+'.observed_output']+=1
    if bad=='fixed_dtype':new[base+'.stage.variance']=new[base+'.stage.variance'].double()
    if bad=='fixed_square':new[base+'.stage.square']+=1
    if bad=='fixed_layout':rn['fixed_old_layouts'][base]['input']['storage_offset']=10**12
    if bad=='missing_fixed':del new[base+'.stage.scaled'];new['dummy']=torch.zeros(1)
    if bad=='epsilon':rn['epsilon']=.01
    call=lambda:handoff.admit_history(ro,old,rn,new,job_id='9',source_revision='a'*40)
    if bad=='none':assert call()['old_tensors']==184
    else:
        with pytest.raises(ValueError,match='RMS|reference|history|fixed|role'):call()


def native_owner(handoff):
    return handoff.rms.layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',projection='separate')


def test_all_native_baseline_and_self_reinjection_arrays_are_hard_gates(handoff):
    old,ro,new,rn=history_fixture(handoff)
    arrays,rows=handoff.run_handoffs(old,new,ro['norm_metadata'],native_owner(handoff),device='cpu')
    assert len(arrays)==784 and len(rows)==80
    assert all(row['native_control_exact'] and row['weighted_output_exact_to_old'] for row in rows.values())
    assert handoff.summarize_handoffs(rows)['exact_endpoints_by_cut']=={
        'variance':16,'variance_plus_epsilon':16,'inverse_rsqrt':16,'scaled':16,'cast_back':16}


def test_baseline_mismatch_cannot_be_interpreted_as_a_handoff_effect(handoff):
    old,ro,new,rn=history_fixture(handoff)
    new['fixed_old.baseline.s0.l1.n0.stage.variance']+=1
    with pytest.raises(ValueError,match='baseline'):
        handoff.run_handoffs(old,new,ro['norm_metadata'],native_owner(handoff),device='cpu')


def test_failed_native_value_control_blocks_handoff_result(handoff,monkeypatch):
    old,ro,new,rn=history_fixture(handoff);original=handoff.resume_rms
    def broken(*args,**kw):
        out=original(*args,**kw);out['weighted_output']=out['weighted_output']+1;return out
    monkeypatch.setattr(handoff,'resume_rms',broken)
    with pytest.raises(ValueError,match='control'):
        handoff.run_handoffs(old,new,ro['norm_metadata'],native_owner(handoff),device='cpu')


def test_summary_keeps_a_negative_handoff_outcome_negative(handoff):
    old,ro,new,rn=history_fixture(handoff)
    _,rows=handoff.run_handoffs(old,new,ro['norm_metadata'],native_owner(handoff),device='cpu')
    rows['baseline.s0.l1.n0.variance']['weighted_output_exact_to_old']=False
    rows['baseline.s0.l1.n0.variance']['first_different_stage']='inverse_rsqrt'
    summary=handoff.summarize_handoffs(rows)
    assert summary['exact_endpoints_by_cut']['variance']==15 and summary['cut_count']==80
    assert summary['first_remaining_stage_counts']=={'inverse_rsqrt':1}


def test_raw_failure_never_leaves_a_success_report(handoff,tmp_path,monkeypatch):
    import safetensors.torch
    def broken(*a,**k):raise OSError('raw failure')
    monkeypatch.setattr(safetensors.torch,'save_file',broken)
    with pytest.raises(OSError,match='raw failure'):handoff.rms.publish(tmp_path,{'x':torch.ones(1)},{'status':'diagnostic_complete'})
    assert not (tmp_path/'result.json').exists()


def test_cli_help_and_allocation_guard_precede_history_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--history-root' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--history-root','/missing','--source-revision','b'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
