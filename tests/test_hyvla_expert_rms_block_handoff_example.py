import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_rms_block_handoff.py'


@pytest.fixture
def block():
    s=importlib.util.spec_from_file_location('hy_block_handoff_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def small_fixture(block):
    from test_hyvla_expert_rms_stages_example import block_fixture
    owner,run=block_fixture(block.rms);captured={};metadata={}
    with block.rms.capture_norms(owner,captured,metadata):expected=run()
    reference=dict(captured)
    for n in range(4):
        stages=block.rms.rms_stages(captured[f'n{n}.input'],captured[f'n{n}.weight'])
        reference[f'n{n}.stage.variance']=stages['variance']
    return owner,run,reference,metadata,expected


def test_guarded_four_norm_intervention_composes_and_restores_actual_block(block):
    owner,run,ref,meta,expected=small_fixture(block);original=owner._rms_norm;calls=[]
    def changed(x,w):calls.append(True);return original(x,w)*.5
    owner._rms_norm=changed;raw={};details={}
    with block.variance_scope(owner,ref,meta,tensors=raw,details=details) as state:actual=run()
    assert state['complete'] and state['calls']==4 and state['attempted']==4 and len(calls)==4
    assert owner._rms_norm is changed and len(raw)==20
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,atol=0,rtol=0)
    for n in range(4):
        assert torch.equal(raw[f'n{n}.returned_output'],ref[f'n{n}.observed_output'])
        assert torch.equal(raw[f'n{n}.variance'],ref[f'n{n}.stage.variance'])


@pytest.mark.parametrize('bad',['input','weight','layout'])
def test_mismatching_live_role_stops_before_native_call_or_variance_use(block,bad):
    owner,run,ref,meta,_=small_fixture(block);called=[];x=ref['n0.input'].clone();w=ref['n0.weight'].clone()
    if bad=='input':x=x+1
    if bad=='weight':w=w+1
    if bad=='layout':meta['n0']['input']['storage_offset']=1
    def forbidden(*a):called.append(True);raise AssertionError('unrelated variance used')
    owner._rms_norm=forbidden;raw={};details={}
    with pytest.raises(block.NormInputMismatch):
        with block.variance_scope(owner,ref,meta,tensors=raw,details=details) as state:owner._rms_norm(x,w)
    assert not called and owner._rms_norm is forbidden
    assert state['calls']==0 and state['attempted']==1 and not state['complete']
    assert set(raw)=={'n0.input','n0.weight'} and state['input_mismatch']['norm']=='n0'


@pytest.mark.parametrize('bad',['body','missing','extra'])
def test_intervention_scope_rejects_incomplete_counts_and_restores_binding(block,bad):
    owner,run,ref,meta,_=small_fixture(block);before=owner._rms_norm
    with pytest.raises((RuntimeError,ValueError),match='body|calls|extra'):
        with block.variance_scope(owner,ref,meta,tensors={},details={}) as state:
            if bad=='body':raise RuntimeError('body failure')
            if bad=='extra':
                run();owner._rms_norm(ref['n0.input'],ref['n0.weight'])
    assert owner._rms_norm is before


@pytest.mark.parametrize('bad',['none','dummy','output','dtype'])
def test_native_baseline_needs_all_real_roles_exact_not_count(block,bad):
    from test_hyvla_expert_rms_stages_example import reference_fixture
    old,report=reference_fixture(block.rms);probe='baseline.s0.l1'
    observed={k.removeprefix('own.'+probe+'.'):v.clone() for k,v in old.items() if k.startswith('own.'+probe+'.') and '.stage.' not in k}
    if bad=='dummy':del observed['n0.observed_output'];observed['dummy']=torch.zeros(1)
    if bad=='output':observed['block.output']+=1
    if bad=='dtype':observed['block.residual']=observed['block.residual'].float()
    if bad=='none':assert block.admit_baseline(observed,old,probe=probe)['tensors']==14
    else:
        with pytest.raises(ValueError,match='baseline'):block.admit_baseline(observed,old,probe=probe)


def test_negative_block_outcome_is_measured_not_a_diagnostic_failure(block):
    ref={'residual':torch.zeros(1,2,dtype=torch.bfloat16),'output':torch.zeros(1,2,dtype=torch.bfloat16)}
    actual={'residual':ref['residual'].clone(),'output':torch.ones(1,2,dtype=torch.bfloat16)}
    result=block.compare_block(actual,ref)
    assert result['residual']['exact'] and not result['output']['exact'] and result['output']['unequal_elements']==2


def test_partial_input_mismatch_cannot_publish_completed_block_roles(block):
    owner,run,ref,meta,_=small_fixture(block);raw={};details={}
    with pytest.raises(block.NormInputMismatch):
        with block.variance_scope(owner,ref,meta,tensors=raw,details=details) as state:
            owner._rms_norm(ref['n0.input']+1,ref['n0.weight'])
    assert block.validate_intervention(raw,state,ref)['status']=='input_mismatch'
    raw['block.output']=torch.zeros(1)
    with pytest.raises(ValueError,match='role'):block.validate_intervention(raw,state,ref)


def test_summary_separates_input_mismatch_from_complete_but_different_output(block):
    rows={p:{'status':'input_mismatch','norm_calls':0} for p in block.rms.PROBES}
    rows[block.rms.PROBES[0]]={'status':'complete','norm_calls':4,'old_block_comparison':{'residual':{'exact':True},'output':{'exact':False}}}
    summary=block.summarize(rows)
    assert summary=={'probes':4,'complete':1,'input_mismatches':3,'residual_exact':1,'output_exact':0}


def test_help_and_allocation_guard_precede_input_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--history-root' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--history-root','/missing','--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()


def input_chain_fixture(block,tmp_path):
    results=tmp_path/'results';history=results/'hy-split-order-315024';history.mkdir(parents=True)
    gold=results/'gold';native=results/'native'
    for case in ('baseline','state'):
        (gold/'official'/case).mkdir(parents=True);(native/'native'/case).mkdir(parents=True)
        (gold/'official'/case/'trace.safetensors').write_bytes(b'gold')
        (native/'native'/case/'trace.safetensors').write_bytes(b'native')
    manifest=history/'inputs.json'
    manifest.write_text(json.dumps({'current_root':str(history),'gold':{'root':str(gold)},'native':{'root':str(native/'native')}}))
    hist={'job_id':'315024','source_revision':'5248e14b544aa8edc17b58f1b61186b5c08ca492',
        'manifest':block.rms.layers._file_record(manifest),'records':{}}
    for case in ('baseline','state'):
        folder=history/'native-original'/case;folder.mkdir(parents=True)
        records={name:block.rms.layers._file_record(path) for name,path in (
            ('manifest',manifest),('gold_trace',gold/'official'/case/'trace.safetensors'),('native_trace',native/'native'/case/'trace.safetensors'))}
        report=folder/'result.json';report.write_text(json.dumps({'input_records':records}))
        hist['records'][case]={'report':block.rms.layers._file_record(report)}
    return {'historical_block_reference':hist},results


@pytest.mark.parametrize('bad',['none','report_stat','empty_records','other_trace','trace_stat','other_report'])
def test_historical_report_and_used_input_paths_are_bound_before_owner_or_weights(block,tmp_path,bad):
    rn,results=input_chain_fixture(block,tmp_path);hist=rn['historical_block_reference'];case='baseline'
    report=Path(hist['records'][case]['report']['path']);data=json.loads(report.read_text())
    trace=Path(data['input_records']['gold_trace']['path']);called=[]
    if bad=='report_stat':report.write_text(report.read_text()+' ')
    if bad in ('empty_records','other_trace'):
        if bad=='empty_records':data['input_records']={}
        else:
            other=results/'other.safetensors';other.write_bytes(b'gold')
            data['input_records']['gold_trace']=block.rms.layers._file_record(other)
        report.write_text(json.dumps(data))
        hist['records'][case]['report']=block.rms.layers._file_record(report)
    if bad=='trace_stat':os.utime(trace,ns=(trace.stat().st_atime_ns,trace.stat().st_mtime_ns+1000000))
    if bad=='other_report':
        other=results/'report.json';other.write_text(report.read_text())
        hist['records'][case]['report']=block.rms.layers._file_record(other)
    def factory():called.append('owner');return 'owner'
    def consume():
        m,paths,owner=block.admitted_input_owner(rn,results_root=results,factory=factory)
        called.append('checkpoint_tensor_reader');return paths,owner
    if bad=='none':
        paths,owner=consume();assert owner=='owner' and called==['owner','checkpoint_tensor_reader']
        assert paths[case]['gold_trace']==trace
    else:
        with pytest.raises(ValueError,match='record|historical|input|report|RMS'):consume()
        assert not called


def test_mismatch_plus_sync_failure_aborts_at_probe_boundary_and_restores_bindings(block):
    import numpy as np
    owner,run,ref,meta,_=small_fixture(block);norm=owner._rms_norm;attn=owner._attn;scope={}
    def sync():raise RuntimeError('synthetic sync failure')
    attention=block.rms.layers.attention_order_scope(owner,np.array([0]),suffix_rows=3,order='original',device='cpu',state=scope,synchronize=sync)
    def mismatch():return owner._rms_norm(ref['n0.input']+1,ref['n0.weight'])
    with pytest.raises(RuntimeError,match='cleanup|sync'):
        block.run_guarded_block(owner,ref,meta,run=mismatch,attention_context=attention,scope=scope)
    assert owner._rms_norm is norm and owner._attn is attn and not scope['cleanup']['passed']
