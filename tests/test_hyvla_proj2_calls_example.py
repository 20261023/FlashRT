import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/proj2_calls.py'


@pytest.fixture
def proj2():
    spec=importlib.util.spec_from_file_location('hy_proj2_calls_test',SCRIPT)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def operands():
    x=(torch.arange(147).remainder(16).to(torch.bfloat16)/4).reshape(3,7,7,1).expand(3,7,7,2048).clone()
    return x,torch.eye(2048,dtype=torch.bfloat16),torch.full((2048,),.5,dtype=torch.bfloat16)


@pytest.mark.parametrize('arm',['batched_bias','per_camera_b1_bias','per_camera_hwd_bias','per_camera_flat_bias','flat147_bias','batched_mm_add','per_camera_mm_add'])
def test_all_seven_arms_preserve_nonuniform_rows_and_readonly_operands(proj2,arm):
    x,w,b=operands();before=x.clone();out=proj2.linear_arm(x,w,b,arm=arm)
    assert out.dtype==torch.bfloat16 and out.shape==(3,7,7,2048)
    assert torch.equal(out,x+.5) and torch.equal(x,before) and torch.equal(b,torch.full((2048,),.5,dtype=torch.bfloat16))


def test_explicit_mm_add_has_bf16_rounding_before_bias_not_fp32_promotion(proj2):
    x=torch.zeros(3,7,7,2048,dtype=torch.bfloat16);x[...,0]=256;x[...,1]=1
    w=torch.zeros(2048,2048,dtype=torch.bfloat16);w[0,0]=1;w[0,1]=.75
    b=torch.zeros(2048,dtype=torch.bfloat16);b[0]=.75
    out=proj2.linear_arm(x,w,b,arm='batched_mm_add')
    assert torch.equal(out[...,0],torch.full((3,7,7),256.,dtype=torch.bfloat16))
    assert torch.count_nonzero(out[...,1:])==0


@pytest.mark.parametrize('bad',['arm','dtype','weight','bias','shape','stride','finite','alignment'])
def test_bad_geometry_or_precision_cannot_reach_an_unqualified_projection(proj2,bad):
    x,w,b=operands();arm='batched_bias'
    if bad=='arm':arm='unknown'
    if bad=='dtype':x=x.float()
    if bad=='weight':w=w.float()
    if bad=='bias':b=b[:1]
    if bad=='shape':x=x[:,:6]
    if bad=='stride':x=x.transpose(1,2)
    if bad=='finite':x[0,0,0,0]=float('nan')
    if bad=='alignment':x=torch.zeros(x.numel()+1,dtype=torch.bfloat16)[1:].reshape(x.shape)
    with pytest.raises(ValueError,match='PROJ2|BF16|geometry|arm|alignment'):
        proj2.linear_arm(x,w,b,arm=arm)


def test_cpu_profile_parser_keeps_actual_lowering_shapes_not_a_predicted_label(proj2):
    events=[SimpleNamespace(name='proj2.test',input_shapes=[]),
        SimpleNamespace(name='aten::linear',input_shapes=[[3,7,7,2048],[2048,2048],[2048]]),
        SimpleNamespace(name='aten::addmm',input_shapes=[[2048],[147,2048],[2048,2048],[],[]]),
        SimpleNamespace(name='aten::cat',input_shapes=[[],[]])]
    r=proj2.summarize_cpu_events(events,'proj2.test')
    assert r['math_events'][1]['input_shapes'][1]==[147,2048] and len(r['math_events'])==2
    assert r['op_counts']=={'aten::linear':1,'aten::addmm':1}
    assert r['cuda_activity_requested'] is False and r['cuBLAS_algorithm_observed'] is False
    with pytest.raises(ValueError,match='profile|Linear|shape|label'):proj2.summarize_cpu_events(events[:1],'proj2.test')


def test_actual_cpu_profile_observes_linear_geometry_and_preserves_values(proj2,tmp_path):
    x,w,b=operands();value,record=proj2.profile_arm(x,w,b,arm='batched_bias',label='proj2.test',path=tmp_path/'trace.json')
    assert torch.equal(value,x+.5) and record['unprofiled_profiled_exact'] is True
    assert any(e['op']=='aten::addmm' and e['input_shapes'][1]==[147,2048] for e in record['lowering']['math_events'])
    assert (tmp_path/'trace.json').is_file()


def support(proj2):
    shapes={'input':(3,14,14,1152),'proj1':(3,14,14,2048),'fused':(3,7,7,4,4096),
        **{k:(3,7,7,4,2048) for k in ('new_x','pooled','predictor0','predictor_activation','score','probabilities','products')},
        **{k:(3,7,7,2048) for k in ('pooled_sum','activation','proj2')},'merged':(3,49,2048),'actual_endpoint':(3,49,2048)}
    one={k:torch.zeros(shape,dtype=torch.bfloat16) for k,shape in shapes.items()}
    raw={o+'.'+f+'.'+k:v for o in ('gold','native') for f in ('official','native_hwd') for k,v in one.items()}
    reports={e:{'kind':'hy_frozen_complete_merger_stages','status':'diagnostic_complete','engine':e,'job_id':'317360',
        'source_revision':'c990f271da94eceae2eeca0d4b0b66243cf02106','model_constructed':False,
        'raw_proof':{'tensors':60,'actual_endpoints_exact':4}} for e in ('official','native')}
    return reports,{'official':dict(raw),'native':dict(raw)}


@pytest.mark.parametrize('bad',['none','job','dummy','activation','endpoint','runtime_values'])
def test_all_r34_raw_support_and_equal_activations_precede_weight_reader(proj2,bad):
    reports,raw=support(proj2);called=[]
    if bad=='job':reports['native']['job_id']='other'
    if bad=='dummy':del raw['native']['gold.native_hwd.activation'];raw['native']['dummy']=torch.zeros(1)
    if bad=='activation':raw['native']['gold.native_hwd.activation']=raw['native']['gold.native_hwd.activation']+1
    if bad=='endpoint':raw['native']['native.native_hwd.actual_endpoint']=raw['native']['native.native_hwd.actual_endpoint']+1
    if bad=='runtime_values':raw['native']['native.native_hwd.score']=raw['native']['native.native_hwd.score']+1
    def factory():called.append(True);return 'no actual weight read'
    call=lambda:proj2.admit_support(reports,raw,factory=factory)
    if bad=='none':assert call()=='no actual weight read' and called==[True]
    else:
        with pytest.raises(ValueError,match='PROJ2|support|raw|endpoint|activation|runtime'):call()
        assert not called


def test_reference_arm_mismatch_rejects_but_other_arm_difference_is_retained(proj2):
    x,_,_=operands();reference={o:{'official':x,'native_hwd':x,'activation':x} for o in ('gold','native')}
    raw={o+'.activation':x for o in reference}
    raw.update({f'{o}.{phase}.{arm}':x for o in reference for phase in ('unprofiled','profiled') for arm in proj2.ARMS})
    raw['gold.profiled.flat147_bias']=x+1;raw['gold.unprofiled.flat147_bias']=x+1
    assert proj2.validate_raw(raw,reference)['tensors']==30
    raw['gold.profiled.batched_bias']=x+1;raw['gold.unprofiled.batched_bias']=x+1
    with pytest.raises(ValueError,match='anchor|baseline|endpoint'):proj2.validate_raw(raw,reference)


def test_cli_help_and_allocation_guard_before_any_reference_or_weight_read(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--reference-root' in r.stdout
    r=subprocess.run([sys.executable,str(SCRIPT),'--reference-root','/missing','--source-revision','a'*40,'--output',str(tmp_path/'unused')],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not (tmp_path/'unused').exists()
