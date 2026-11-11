import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/frozen_eager_attention.py'


@pytest.fixture
def frozen():
    if not SCRIPT.exists():pytest.fail('frozen eager attention controls are missing')
    spec=importlib.util.spec_from_file_location('hy_frozen_eager',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_method_extraction_never_executes_module_class_or_unrelated_methods(frozen,tmp_path):
    path=tmp_path/'source.py'
    path.write_text('raise AssertionError("module executed")\nclass Operator:\n'
        '    raise AssertionError("class executed")\n'
        '    def attention(self, value):\n        return value+2\n'
        '    def other(self):\n        raise AssertionError("other")\n')
    fn=frozen.load_method(path,'Operator','attention')
    assert fn(None,3)==5


def test_live_layout_restoration_keeps_singleton_stride_offset_and_values(frozen):
    values=torch.tensor([[[1.,2.],[3.,4.]]])
    info={'shape':[1,2,2],'live_stride':[999,1,4],'storage_offset':3,'dtype':'torch.float32'}
    actual=frozen.restore_layout(values,info,device='cpu')
    assert actual.stride()==(999,1,4) and actual.storage_offset()==3
    torch.testing.assert_close(actual,values,atol=0,rtol=0)
    assert not actual.is_contiguous()


@pytest.mark.parametrize('change',['overlap','negative','shape','dtype'])
def test_invalid_recorded_layout_rejects_before_operator_execution(frozen,change):
    info={'shape':[2,2],'live_stride':[2,1],'storage_offset':0,'dtype':'torch.float32'}
    if change=='overlap':info['live_stride']=[1,1]
    if change=='negative':info['storage_offset']=-1
    if change=='shape':info['shape']=[1,4]
    if change=='dtype':info['dtype']='torch.bfloat16'
    with pytest.raises(ValueError,match='layout|stride|dtype|shape|offset'):
        frozen.restore_layout(torch.ones(2,2),info,device='cpu')


def functions(frozen,head_dim):
    from test_hyvla_eager_attention_capture_example import Tower
    official=Tower().eager_attention_forward
    fn=frozen.load_method(ROOT/'flash_rt/models/hyvla/pipeline_thor.py','HyVLAThorBF16Pipeline','_attn')
    owner=SimpleNamespace(numerical_profile='official_local_fp32',head_dim=head_dim)
    return official,lambda *args:fn(owner,*args)


def test_four_arms_use_nonunit_scale_gqa_and_correct_output_head_order(frozen):
    # H=2, KV=1, D=2: head0 scores[0,log3] -> (5,0), head1 uniform -> (4,0).
    q=torch.tensor([[[[math.sqrt(2),0.],[0.,0.]],[[math.sqrt(2),0.],[0.,0.]]]])
    k=torch.tensor([[[[0.,0.]],[[math.log(3),0.]]]])
    v=torch.tensor([[[[2.,0.]],[[6.,0.]]]],dtype=torch.bfloat16)
    mask=torch.ones(1,2,2,dtype=torch.bool)
    official,native=functions(frozen,2)
    outputs,metadata=frozen.attention_controls(q,k,v,mask,prefix_rows=0,prefix_perm=torch.empty(0,dtype=torch.long),
        official_fn=official,native_fn=native)
    expected=torch.tensor([[[5.,0.,4.,0.],[5.,0.,4.,0.]]],dtype=torch.bfloat16)
    for name,value in outputs.items():
        if name.endswith('.output'):torch.testing.assert_close(value,expected,atol=0,rtol=0)
    assert metadata['readonly_inputs_preserved'] is True


def test_sorted_prefix_reorders_key_value_and_mask_columns_but_not_suffix_or_query(frozen):
    q=torch.ones(1,2,2,2);k=torch.arange(10.).reshape(1,5,1,2);v=k.bfloat16()
    mask=torch.tensor([[[True,False,True,True,False],[False,True,True,True,True]]])
    official,native=functions(frozen,2);seen=[]
    def observe(q,k,v,mask):
        seen.append((q.clone(),k.clone(),v.clone(),mask.clone()));return native(q,k,v,mask)
    before=[x.clone() for x in (q,k,v,mask)]
    outputs,metadata=frozen.attention_controls(q,k,v,mask,prefix_rows=3,prefix_perm=torch.tensor([2,0,1]),
        official_fn=official,native_fn=observe)
    assert len(seen)==3
    torch.testing.assert_close(seen[2][1][0,0,:,0],torch.tensor([4.,0.,2.,6.,8.]),atol=0,rtol=0)
    assert torch.equal(seen[2][3],mask[:,None,:,torch.tensor([2,0,1,3,4])])
    assert torch.equal(seen[1][0],seen[2][0])
    assert metadata['layouts']['native_contiguous_original_order']['k']['live_stride']==metadata['layouts']['native_contiguous_sorted_order']['k']['live_stride']
    for actual,old in zip((q,k,v,mask),before):torch.testing.assert_close(actual,old,atol=0,rtol=0)


@pytest.mark.parametrize('perm',[torch.tensor([0,0]),torch.tensor([0.,1.]),torch.tensor([0,2])])
def test_bad_prefix_permutation_never_calls_either_operator(frozen,perm):
    def forbidden(*args):pytest.fail('invalid prefix reached operator')
    with pytest.raises(ValueError,match='permutation'):
        frozen.attention_controls(torch.ones(1,2,2,2),torch.ones(1,4,1,2),torch.ones(1,4,1,2,dtype=torch.bfloat16),
            torch.ones(1,2,4,dtype=torch.bool),prefix_rows=2,prefix_perm=perm,official_fn=forbidden,native_fn=forbidden)


def test_operator_readonly_violation_is_fatal(frozen):
    official,native=functions(frozen,2)
    def wrong(q,k,v,mask):q.add_(1);return native(q,k,v,mask)
    with pytest.raises(ValueError,match='mutat|read.only'):
        frozen.attention_controls(torch.ones(1,2,2,2),torch.ones(1,2,1,2),torch.ones(1,2,1,2,dtype=torch.bfloat16),
            torch.ones(1,2,2,dtype=torch.bool),prefix_rows=0,prefix_perm=torch.empty(0,dtype=torch.long),official_fn=official,native_fn=wrong)


def test_fresh_capture_admission_rejects_wrong_job_source_or_incomplete_capture(frozen,tmp_path):
    source=tmp_path/'attention.py';source.write_text('class Operator: pass\n');stat=source.stat()
    report={'engine':'official','job_id':'9','variant':'state','source_revision':'a'*40,
        'artifact_complete':True,'status':'recorded_oracle_reproduced','oracle':{'exact':True},
        'prefix_invariance':{'prefix.k':True},'eager_attention_capture':{'complete':True,
            'num_layers':32,'num_steps':10,'phases':[{'phase':p,'attention_calls':32}
                for p in ('prefix',*(f'step{i}' for i in range(10)))]},
        'historical_attention_admission':{'exact':True},
        'official_attention_source':{'path':str(source.resolve()),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}
    check=lambda:frozen.admit_capture_report(report,job_id='9',source_revision='a'*40,variant='state',attention_source=source)
    assert check() is None
    report['job_id']='8'
    with pytest.raises(ValueError,match='capture|job|source'):check()
    report['job_id']='9';report['eager_attention_capture']['complete']=False
    with pytest.raises(ValueError,match='capture'):check()
    report['eager_attention_capture']['complete']=True
    report['eager_attention_capture']['phases'][1]['phase']='step7'
    with pytest.raises(ValueError,match='capture|phase'):check()


def test_raw_write_failure_never_leaves_final_operator_success_json(frozen,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*args,**kwargs):raise OSError('raw write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write failed'):frozen.publish(tmp_path,{'x':torch.ones(1)},{'status':'diagnostic_complete'})
    assert not (tmp_path/'result.json').exists()


def test_unadmitted_capture_never_compiles_either_source_method(frozen,tmp_path,monkeypatch):
    def forbidden(*args):raise AssertionError('source method compiled before capture admission')
    monkeypatch.setattr(frozen,'load_method',forbidden)
    with pytest.raises(ValueError,match='capture|job|source'):
        frozen.load_admitted_operators({'engine':'official','job_id':'wrong'},job_id='9',
            source_revision='a'*40,variant='state',attention_source=tmp_path/'official.py',native_source=tmp_path/'native.py')


def test_help_and_allocation_guard_do_not_read_models_or_create_output(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert result.returncode==0
    output=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--reference','/missing',
        '--official-attention-source','/missing','--source-revision','0'*40,'--output',str(output)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not output.exists()
