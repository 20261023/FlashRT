import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_mlp.py'


@pytest.fixture
def mlp():
    spec=importlib.util.spec_from_file_location('hy_expert_mlp',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class TinyMLP(torch.nn.Module):
    def __init__(self,bias=False):
        super().__init__()
        for name in ('gate_proj','up_proj','down_proj'):
            layer=torch.nn.Linear(2,2,bias=bias,dtype=torch.bfloat16)
            with torch.no_grad():layer.weight.copy_(torch.eye(2))
            setattr(self,name,layer)
        self.fail=False;self.change_up=False
    def forward(self,x):
        gate=self.gate_proj(x)
        if self.fail:raise RuntimeError('injected MLP failure')
        up=self.up_proj(x+1 if self.change_up else x)
        return self.down_proj(torch.nn.functional.silu(gate)*up)


def test_literal_nonzero_bf16_controls_preserve_separate_and_fused_math(mlp):
    x=torch.tensor([[1.,2.],[-1.,0.]],dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16)
    out=mlp.mlp_controls(x,w,w,w)
    expected=torch.tensor([[.73046875,3.515625],[.26953125,0.]],dtype=torch.bfloat16)
    for arm in ('separate','fused'):
        torch.testing.assert_close(out[arm+'.output'],expected,atol=0,rtol=0)
        torch.testing.assert_close(out[arm+'.gate'],x,atol=0,rtol=0)
        torch.testing.assert_close(out[arm+'.up'],x,atol=0,rtol=0)


def test_fused_control_really_changes_gemm_n_and_keeps_same_input(mlp):
    x=torch.tensor([[1.,2.],[-1.,0.]],dtype=torch.bfloat16);w=torch.eye(2,dtype=torch.bfloat16)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,acc_events=True) as p:
        mlp.mlp_controls(x,w,w,w)
    shapes=[e.input_shapes[1] for e in p.events() if e.name=='aten::mm']
    assert shapes==[[2,2],[2,2],[2,2],[2,4],[2,2]]
    torch.testing.assert_close(x,torch.tensor([[1.,2.],[-1.,0.]],dtype=torch.bfloat16),atol=0,rtol=0)


def test_observer_records_real_module_events_and_restores_all_hooks(mlp):
    model=TinyMLP();x=torch.tensor([[1.,-1.],[2.,0.]],dtype=torch.bfloat16).t();saved={};meta={}
    expected=model(x)
    with mlp.observe_mlp(model,saved,meta,num_steps=2) as state:
        for _ in range(2):torch.testing.assert_close(model(x),expected,atol=0,rtol=0)
    assert state['complete'] and state['steps']==2 and state['events']==10
    assert len(saved)==10 and meta['expert0.mlp.step0.input']['live_stride']==list(x.stride())
    torch.testing.assert_close(saved['expert0.mlp.step1.output'],expected,atol=0,rtol=0)
    x.add_(10);assert saved['expert0.mlp.step1.input'].max()<10
    for module in model.modules():assert not module._forward_hooks and not module._forward_pre_hooks


@pytest.mark.parametrize('failure',['body','count','bias','dtype','geometry','different_up'])
def test_bad_capture_is_fatal_and_hooks_are_removed(mlp,failure):
    model=TinyMLP(bias=failure=='bias');model.fail=failure=='body';model.change_up=failure=='different_up'
    x=torch.ones(2,2,dtype=torch.float32 if failure=='dtype' else torch.bfloat16)
    if failure=='geometry':x=x[None]
    state={}
    with pytest.raises((RuntimeError,ValueError),match='MLP|BF16|bias|input|step'):
        with mlp.observe_mlp(model,{}, {},num_steps=2) as state:model(x)
    assert not state.get('complete',False)
    for module in model.modules():assert not module._forward_hooks and not module._forward_pre_hooks


def test_mlp_capture_proof_binds_source_job_checkpoint_and_complete_steps(mlp,tmp_path):
    source=tmp_path/'mlp.py';source.write_text('source');s=source.stat()
    cp={'path':'/checkpoint','bytes':10,'mtime_ns':20}
    report={'engine':'official','job_id':'9','variant':'state','source_revision':'a'*40,'checkpoint':cp.copy(),
        'expert_mlp_capture':{'complete':True,'steps':10,'events':50,'module':'expert.model.layers.0.mlp_v',
            'source':{'path':str(source.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}}}
    check=lambda:mlp.validate_mlp_capture(report,job_id='9',source_revision='a'*40,variant='state',checkpoint=cp,mlp_source=source)
    assert check() is None
    report['expert_mlp_capture']['steps']=9
    with pytest.raises(ValueError,match='MLP|capture'):check()
    report['expert_mlp_capture']['steps']=10;report['checkpoint']={**cp,'bytes':11}
    with pytest.raises(ValueError,match='MLP|checkpoint'):check()
    report['checkpoint']=cp.copy();report['job_id']='8'
    with pytest.raises(ValueError,match='MLP|job'):check()


def test_failed_raw_write_does_not_publish_success(mlp,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*a,**k):raise OSError('injected raw write failure')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write'):mlp.publish(tmp_path,{'x':torch.ones(1)},{'status':'complete'})
    assert not (tmp_path/'result.json').exists()


def test_cli_help_and_allocation_guard_before_checkpoint_access(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    help_result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert help_result.returncode==0 and '--official-mlp-source' in help_result.stdout
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--reference','/missing',
        '--checkpoint','/missing','--official-mlp-source','/missing','--source-revision','a'*40,'--output',str(out)],
        env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not out.exists()
