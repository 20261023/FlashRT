import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/frozen_suffix.py'


@pytest.fixture
def suffix():
    if not SCRIPT.exists():pytest.fail('frozen suffix probe is missing')
    spec=importlib.util.spec_from_file_location('hy_frozen_suffix',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def weights():
    bf=torch.bfloat16
    return SimpleNamespace(_state_w=torch.eye(2,dtype=bf),_state_b=torch.zeros(2,dtype=bf),
        _ain_w=torch.tensor([[1.],[2.]],dtype=bf),_ain_b=torch.tensor([1.,0.],dtype=bf),
        _atmlp_in_w=torch.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.]],dtype=bf),_atmlp_in_b=torch.zeros(2,dtype=bf),
        _atmlp_out_w=torch.eye(2,dtype=bf),_atmlp_out_b=torch.zeros(2,dtype=bf))


def test_suffix_keeps_state_then_action_tokens_time_broadcast_and_bf16_rounding(suffix):
    state=torch.tensor([[1.,-1.]],dtype=torch.bfloat16);x=torch.tensor([[[0.],[1.]]]);time=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
    actual=suffix.build_suffix(state,x,time,weights())
    expected=torch.tensor([[[1.,-1.],[1.7578125,1.7578125],[2.859375,3.921875]]],dtype=torch.bfloat16)
    torch.testing.assert_close(actual['suffix'],expected,atol=0,rtol=0)
    torch.testing.assert_close(x,torch.tensor([[[0.],[1.]]]),atol=0,rtol=0)
    assert actual['time_input'].shape==(1,2,4)


def test_suffix_probe_matches_actual_native_pre_expert_source_body(suffix):
    tree=ast.parse((ROOT/'flash_rt/models/hyvla/pipeline_thor.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HyVLAThorBF16Pipeline')
    original=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='evaluate_velocity')
    body=[]
    for node in original.body:
        if isinstance(node,ast.For):break
        body.append(node)
    fn=ast.FunctionDef(name='native_suffix',args=original.args,body=body+[ast.Return(value=ast.Name(id='hs',ctx=ast.Load()))],decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={'torch':torch,'F':torch.nn.functional}
    exec(compile(module,'actual_native_suffix','exec'),scope)
    w=weights();state=torch.tensor([[1.,-1.]],dtype=torch.bfloat16);x=torch.tensor([[[0.],[1.]]]);time=torch.tensor([[1.,2.]],dtype=torch.bfloat16)
    result=suffix.build_suffix(state,x,time,w)
    actual=scope['native_suffix'](SimpleNamespace(W=w),x,time,result['state_embedding'],None,None,None,None,None,0)
    torch.testing.assert_close(result['suffix'],actual,atol=0,rtol=0)


@pytest.mark.parametrize('fused',[False,True])
def test_projection_controls_share_values_but_really_change_gemm_n_geometry(suffix,fused):
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.bfloat16)
    q=torch.eye(2,dtype=torch.bfloat16);k=torch.tensor([[1.,1.]],dtype=torch.bfloat16);v=torch.tensor([[1.,-1.]],dtype=torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,acc_events=True) as p:
        actual=suffix.project_qkv(x,q,k,v,fused=fused)
    expected=torch.tensor([[[1.,2.,3.,-1.],[3.,4.,7.,-1.]]],dtype=torch.bfloat16)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    shapes=[e.input_shapes[1] for e in p.events() if e.name=='aten::mm']
    assert shapes==([[2,4]] if fused else [[2,2],[2,1],[2,1]])


def test_rotary_tables_and_query_norm_keep_fp32_outputs(suffix):
    inv=torch.tensor([1.],dtype=torch.bfloat16);positions=torch.tensor([[0,1]])
    cos,sin=suffix.rope_tables(inv,positions)
    assert cos.dtype==sin.dtype==torch.float32 and cos.shape==(1,1,2,2)
    qkv=torch.tensor([[[1.,0.,0.,0.],[1.,0.,0.,0.]]],dtype=torch.bfloat16)
    normalized=suffix.normalized_query(qkv,query_width=2,head_dim=2,cos=cos,sin=sin,weight=torch.ones(2,dtype=torch.bfloat16))
    assert normalized.dtype==torch.float32 and normalized.shape==(1,1,2,2)
    torch.testing.assert_close(normalized[0,0,0],torch.tensor([1.4141994,0.]),atol=1e-6,rtol=0)


def test_rms_probe_matches_actual_profile_for_flat_and_batched_inputs(suffix):
    tree=ast.parse((ROOT/'flash_rt/models/hyvla/pipeline_thor.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='HyVLAThorBF16Pipeline')
    fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_rms_norm')
    scope={'torch':torch,'F':torch.nn.functional};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_rms','exec'),scope)
    x=torch.tensor([[[1.,2.],[3.,4.]]],dtype=torch.bfloat16);weight=torch.tensor([1.5,.5],dtype=torch.bfloat16)
    owner=SimpleNamespace(numerical_profile='official_local_fp32',rms_eps=1e-5)
    for value in (x,x[0],x.float()):
        torch.testing.assert_close(suffix.local_rms(value,weight),scope['_rms_norm'](owner,value,weight),atol=0,rtol=0)


@pytest.mark.parametrize('bad',['fold','native_engine','official_job','checkpoint','execution'])
def test_historical_admission_rejects_changed_producers_checkpoint_and_fold_scope(suffix,bad):
    official={'engine':'official','job_id':'308690','variant':'state','oracle':{'exact':True}}
    stat={'path':'/checkpoint/model.safetensors','bytes':10,'mtime_ns':20}
    native={'engine':'native','job_id':'313766','variant':'state','artifact_complete':True,
        'execution_mode':'diagnostic_eager','config':{'use_cuda_graph':False},'diagnostic_cleanup':{'passed':True},
        'experimental_projection':{'fold_spatial':True,'source_revision':'a'*40,'precondition':{
            'checkpoint':stat.copy(),'operator_exact':True,'job_id':'313766','source_revision':'a'*40}}}
    suffix.admit_sources(official,native,variant='state',official_job='308690',native_job='313766',checkpoint=stat)
    if bad=='fold':native['experimental_projection']['fold_spatial']=False
    if bad=='native_engine':native['engine']='official'
    if bad=='official_job':official['job_id']='other'
    if bad=='checkpoint':stat['mtime_ns']+=1
    if bad=='execution':native['execution_mode']='graph'
    with pytest.raises(ValueError,match='source|fold|identity'):
        suffix.admit_sources(official,native,variant='state',official_job='308690',native_job='313766',checkpoint=stat)


def test_atomic_probe_publication_does_not_leave_success_json_on_raw_failure(suffix,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*args,**kwargs):raise OSError('raw write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write failed'):
        suffix.publish(tmp_path,{'x':torch.ones(1)},{'status':'diagnostic_complete'})
    assert not (tmp_path/'result.json').exists()


def test_help_and_gpu_allocation_guard_before_any_checkpoint_read(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    help_result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert help_result.returncode==0
    output=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--checkpoint','/missing',
        '--official','/missing','--native','/missing','--source-revision','0'*40,'--output',str(output)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not output.exists()
