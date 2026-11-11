import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def order():
    path=ROOT/'examples/vla_autotune/hy_trace/suffix_key_order.py'
    if not path.exists():pytest.fail('suffix key-order intervention is missing')
    spec=importlib.util.spec_from_file_location('hy_suffix_order',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_inverse_columns_restore_prefix_and_leave_suffix_identity(order):
    result=order.inverse_columns(torch.tensor([2,0,1]),suffix_rows=2,device='cpu')
    assert result.tolist()==[1,2,0,3,4]


@pytest.mark.parametrize('perm',[torch.tensor([0,0]),torch.tensor([0.,1.]),torch.tensor([0,2])])
def test_invalid_prefix_permutation_never_silently_coerces_or_duplicates(order,perm):
    with pytest.raises(ValueError,match='permutation'):
        order.inverse_columns(perm,suffix_rows=2,device='cpu')


def attention_function():
    path=ROOT/'examples/vla_autotune/hy_trace/frozen_eager_attention.py'
    spec=importlib.util.spec_from_file_location('hy_order_attn_source',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.load_method(ROOT/'flash_rt/models/hyvla/pipeline_thor.py','HyVLAThorBF16Pipeline','_attn')


def arrays():
    return (torch.ones(1,2,2,2),torch.arange(20.).reshape(1,2,5,2),
        torch.arange(20.,dtype=torch.bfloat16).reshape(1,2,5,2),
        torch.tensor([[[[True,False,True,True,False],[False,True,True,True,True]]]]))


def test_same_actual_adapter_recovers_original_attention_from_sorted_inputs(order):
    fn=attention_function();owner=SimpleNamespace(numerical_profile='official_local_fp32',head_dim=2)
    original=lambda *args:fn(owner,*args)
    q,k,v,mask=arrays();expected=original(q,k,v,mask)
    sort=torch.tensor([2,0,1,3,4]);columns=order.inverse_columns(sort[:3],suffix_rows=2,device='cpu')
    sorted_k,sorted_v,sorted_mask=k.index_select(2,sort),v.index_select(2,sort),mask.index_select(-1,sort)
    actual=order.attention_in_original_order(original,q,sorted_k,sorted_v,sorted_mask,columns=columns)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    torch.testing.assert_close(sorted_k,k.index_select(2,sort),atol=0,rtol=0)


def frontend():
    class Pipe:
        numerical_profile='official_local_fp32';head_dim=2
        _attn=attention_function()
    return SimpleNamespace(pipe=Pipe(),device=torch.device('cpu'),chunk=1,use_cuda_graph=False,
                           _graph_cache={},_vit_graph_cache={})


def test_context_is_eager_only_suffix_only_and_restores_class_binding_after_drain(order):
    front=frontend();state={};events=[]
    assert not hasattr(front,'use_graph')
    q,k,v,mask=arrays();sort=torch.tensor([2,0,1,3,4]);expected=front.pipe._attn(q,k,v,mask)
    with order.suffix_original_order(front,sort[:3],state=state,synchronize=lambda:events.append('sync')):
        actual=front.pipe._attn(q,k.index_select(2,sort),v.index_select(2,sort),mask.index_select(-1,sort))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert events==['sync'] and state['cleanup']['passed'] and state['attention_calls']==1
    assert '_attn' not in front.pipe.__dict__
    front.use_cuda_graph=True
    with pytest.raises(ValueError,match='eager|graph'):
        with order.suffix_original_order(front,sort[:3],state={},synchronize=lambda:None):pass


@pytest.mark.parametrize('body_failure',[False,True])
def test_sync_failure_retains_columns_and_restores_binding_without_masking_primary(order,body_failure):
    front=frontend();state={}
    def sync():raise RuntimeError('sync failed')
    with pytest.raises(RuntimeError,match='body failed' if body_failure else 'sync failed') as error:
        with order.suffix_original_order(front,torch.tensor([2,0,1]),state=state,synchronize=sync):
            if body_failure:raise RuntimeError('body failed')
    assert '_attn' not in front.pipe.__dict__ and state['cleanup']['passed'] is False
    assert state['_failed_owners']
    if body_failure:assert any('sync failed' in x for x in error.value.__notes__)


def test_order_proof_binds_current_job_revision_and_method_source(order,tmp_path):
    path=tmp_path/'native.py';path.write_text('source')
    stat=path.stat();proof={'kind':'hy_suffix_original_order_exact_precondition','operator_exact':True,
        'job_id':'9','source_revision':'a'*40,'native_method_source':{'path':str(path.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns},
        'cases':{'baseline':{'steps':10,'exact':True},'state':{'steps':10,'exact':True}}}
    order.validate_order_proof(proof,job_id='9',source_revision='a'*40,native_source=path)
    proof['job_id']='8'
    with pytest.raises(ValueError,match='proof|job|source'):
        order.validate_order_proof(proof,job_id='9',source_revision='a'*40,native_source=path)


def test_order_preflight_raw_write_failure_has_no_final_json(order,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*args,**kwargs):raise OSError('raw write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write failed'):order.publish(tmp_path,{'x':torch.ones(1)},{'operator_exact':True})
    assert not (tmp_path/'result.json').exists()


def test_sorted_frozen_baseline_raw_arrays_cannot_be_replaced_by_passed_flags(order):
    current={'native_kv.step0.velocity':torch.tensor([1.],dtype=torch.bfloat16)}
    old={'native_kv.step0.velocity':torch.tensor([1.],dtype=torch.bfloat16)}
    assert order.admit_sorted_baseline(current,old)['exact']
    old['native_kv.step0.velocity'].add_(1)
    with pytest.raises(ValueError,match='baseline|tensor'):order.admit_sorted_baseline(current,old)


def test_sorted_baseline_header_binds_original_eager_fold_and_historical_reference(order):
    report={'engine':'native','job_id':'313766','artifact_complete':True,'execution_mode':'diagnostic_eager',
        'diagnostic_cleanup':{'passed':True},'config':{'use_cuda_graph':False},
        'experimental_projection':{'fold_spatial':True,'effective_projection':'per_camera_hwd'},
        'historical_official_reference':{'job_id':'308690'},'checkpoint':{'path':'/checkpoint'},'runtime':{'torch':'2.11'}}
    check=lambda:order.admit_sorted_baseline_report(report,job_id='313766',reference_job='308690',
        checkpoint={'path':'/checkpoint'},runtime={'torch':'2.11'})
    assert check()['admitted']
    report['suffix_order_control']={'enabled':True}
    with pytest.raises(ValueError,match='baseline|order'):check()


def test_original_order_mask_admission_rejects_different_attention_eligibility(order):
    expected=torch.tensor([[[True,False,True,True,False],[False,True,True,True,True]]])
    perm=torch.tensor([2,0,1]);columns=torch.tensor([2,0,1,3,4])
    smask=expected[:,None].index_select(-1,columns)
    assert order.admit_original_mask(smask,expected,perm)['exact']
    wrong=expected.clone();wrong[0,0,0]=False
    with pytest.raises(ValueError,match='mask|eligibility'):order.admit_original_mask(smask,wrong,perm)


def test_order_cli_help_and_allocation_guard_before_any_input_io(tmp_path):
    script=ROOT/'examples/vla_autotune/hy_trace/suffix_key_order.py'
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    help_result=subprocess.run([sys.executable,str(script),'--help'],env=env,text=True,capture_output=True)
    assert help_result.returncode==0 and '--reference-source-revision' in help_result.stdout
    output=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(script),'--reference','/missing','--reference-ops','/missing',
        '--official-attention-source','/missing','--reference-job','314150','--reference-source-revision','b'*40,
        '--source-revision','a'*40,'--output',str(output)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not output.exists()
