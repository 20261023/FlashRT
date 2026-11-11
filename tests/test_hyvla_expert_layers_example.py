import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'examples/vla_autotune/hy_trace/expert_layers.py'


@pytest.fixture
def layers():
    spec=importlib.util.spec_from_file_location('hy_expert_layers',SCRIPT)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def native_owner(layers):
    return layers.load_native_owner(ROOT/'flash_rt/models/hyvla/pipeline_thor.py',n_heads=2,n_kv=1,head_dim=2)


def weights():
    # Zero Q/K and V; no attention contribution. Nonzero FFN is independently
    # calculable at input [1,1]: rms rounds to [1,1], silu rounds .73046875.
    qkv=torch.zeros(8,2,dtype=torch.bfloat16)
    o=torch.zeros(2,4,dtype=torch.bfloat16)
    gu=torch.cat((torch.eye(2),torch.eye(2))).bfloat16()
    return (qkv,o,gu,torch.eye(2,dtype=torch.bfloat16),torch.ones(2,dtype=torch.bfloat16),torch.ones(2,dtype=torch.bfloat16))


def block_args():
    return (torch.ones(1,2,2,dtype=torch.bfloat16),torch.zeros(1,2,3,2),
            torch.zeros(1,2,3,2,dtype=torch.bfloat16),torch.ones(1,1,2,3,dtype=torch.bool),
            torch.ones(1,1,2,2),torch.zeros(1,1,2,2))


def test_actual_native_body_replay_observes_residual_and_nonzero_ffn(layers):
    owner=native_owner(layers);x,k,v,mask,cos,sin=block_args();original_norm=owner._rms_norm
    residual,out=layers.replay_block(owner,x,weights(),(torch.ones(2,dtype=torch.bfloat16),)*2,
        k,v,mask,cos,sin,prefix_rows=1)
    torch.testing.assert_close(residual,torch.ones(1,2,2,dtype=torch.bfloat16),atol=0,rtol=0)
    torch.testing.assert_close(out,torch.full((1,2,2),1.734375,dtype=torch.bfloat16),atol=0,rtol=0)
    assert owner._rms_norm is original_norm
    assert torch.count_nonzero(k)==torch.count_nonzero(v)==0
    torch.testing.assert_close(x,torch.ones(1,2,2,dtype=torch.bfloat16),atol=0,rtol=0)


@pytest.mark.parametrize('bad',['prefix','input','norm_count'])
def test_replay_detects_mutation_and_restores_norm_binding_on_failure(layers,bad):
    owner=native_owner(layers);x,k,v,mask,cos,sin=block_args();original=owner._rms_norm
    real=owner._block
    def wrong(hs,n_vis,w_text,w_vis,qk,mask,cos,sin,kbuf,vbuf,off):
        if bad=='prefix':kbuf[:,:,:off].fill_(1)
        if bad=='input':hs.add_(1)
        if bad=='norm_count':return hs.clone()
        return real(hs,n_vis,w_text,w_vis,qk,mask,cos,sin,kbuf,vbuf,off)
    owner._block=wrong
    with pytest.raises(ValueError,match='prefix|mutat|norm|read.only'):
        layers.replay_block(owner,x,weights(),(torch.ones(2,dtype=torch.bfloat16),)*2,k,v,mask,cos,sin,prefix_rows=1)
    assert owner._rms_norm is original


def test_native_source_loader_does_not_execute_module_or_class_body(layers,tmp_path):
    source=tmp_path/'native.py'
    source.write_text('raise AssertionError("module")\ndef _rot_half(x): return x\n'
        'class HyVLAThorBF16Pipeline:\n    raise AssertionError("class")\n'
        '    def _rms_norm(self,x,w): return x*w\n'
        '    def _attn(self,q,k,v,m): return q\n'
        '    def _block(self,hs,*args): return hs\n')
    owner=layers.load_native_owner(source,n_heads=2,n_kv=1,head_dim=2)
    torch.testing.assert_close(owner._rms_norm(torch.tensor([2.]),torch.tensor([3.])),torch.tensor([6.]),atol=0,rtol=0)


def test_layer_weight_keys_and_fusion_order_match_real_checkpoint_contract(layers):
    keys=layers.layer_keys(7)
    assert len(keys)==11 and keys['q']=='model.dual_tower.expert.model.layers.7.self_attn.q_proj_v.weight'
    assert keys['kn']=='model.dual_tower.vlm.model.language_model.model.layers.7.self_attn.key_layernorm.weight'
    raw={name:torch.full((2,2),float(index),dtype=torch.bfloat16) for index,name in enumerate(('q','k','v','o','g','u','d','ln','post','qn','kn'),1)}
    raw['ln']=raw['post']=torch.ones(2,dtype=torch.bfloat16);raw['qn']=raw['kn']=torch.ones(2,dtype=torch.bfloat16)
    w,qk=layers.fuse_layer_weights(raw)
    assert w[0][:,0].tolist()==[1.,1.,2.,2.,3.,3.]
    assert w[2][:,0].tolist()==[5.,5.,6.,6.]
    assert len(w)==6 and len(qk)==2


def test_h0_admission_checks_state_and_every_step_not_only_step0(layers):
    rebuilt={'state_embedding':torch.zeros(1,1,2,dtype=torch.bfloat16),'suffix':torch.ones(1,2,2,dtype=torch.bfloat16)}
    layers.admit_h0(rebuilt,torch.ones(2,2,dtype=torch.bfloat16),torch.zeros(1,1,2,dtype=torch.bfloat16))
    with pytest.raises(ValueError,match='H0|suffix|state'):
        layers.admit_h0(rebuilt,torch.zeros(2,2,dtype=torch.bfloat16),torch.zeros(1,1,2,dtype=torch.bfloat16))


def test_layer_modes_use_gold_each_layer_versus_propagated_output(layers):
    gold=torch.full((1,2,2),3.,dtype=torch.bfloat16);prop=torch.full((1,2,2),7.,dtype=torch.bfloat16)
    actual=layers.layer_inputs(gold,prop)
    assert tuple(actual)==('teacher','propagated')
    torch.testing.assert_close(actual['teacher'],gold,atol=0,rtol=0)
    torch.testing.assert_close(actual['propagated'],prop,atol=0,rtol=0)
    actual['teacher'].add_(1)
    torch.testing.assert_close(gold,torch.full((1,2,2),3.,dtype=torch.bfloat16),atol=0,rtol=0)
    torch.testing.assert_close(actual['propagated'],prop,atol=0,rtol=0)


def test_final_head_matches_actual_native_shape_dtype_and_no_sign_or_dt(layers):
    owner=native_owner(layers);hs=torch.ones(1,3,2,dtype=torch.bfloat16)
    actual=layers.final_velocity(owner,hs,torch.ones(2,dtype=torch.bfloat16),torch.tensor([[1.,2.]],dtype=torch.bfloat16),
        torch.tensor([.5],dtype=torch.bfloat16),action_rows=2)
    torch.testing.assert_close(actual,torch.full((1,2,1),3.5,dtype=torch.bfloat16),atol=0,rtol=0)


def test_native_baseline_gate_checks_dtype_and_all_steps_and_fails_closed(layers):
    current=[torch.ones(1,2,1,dtype=torch.bfloat16),torch.zeros(1,2,1,dtype=torch.bfloat16)]
    baseline=[x.clone() for x in current]
    assert layers.baseline_gate(current,baseline,engine='native')['passed']
    baseline[1].add_(1)
    result=layers.baseline_gate(current,baseline,engine='native')
    assert result['passed'] is False and result['exact_steps']==1
    assert layers.baseline_gate(current,baseline,engine='official')['required'] is False
    baseline[1]=current[1].float()
    assert layers.baseline_gate(current,baseline,engine='native')['passed'] is False


def test_capture_header_requires_all_layer_chain_and_actual_source(layers,tmp_path):
    source=tmp_path/'official.py';source.write_text('official');s=source.stat()
    cp={'path':'/checkpoint','bytes':1,'mtime_ns':2}
    state={'complete':True,'num_layers':32,'num_steps':10,'steps':10,'events':970,'rows':41,'hidden':1024,
        'module':'expert.model','chain':{'exact':True,'layer_boundaries':320,'first_suffix_checked':True},
        'source':{'path':str(source.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}}
    report={'expert_layer_capture':state,'checkpoint':cp.copy()}
    check=lambda:layers.admit_layer_capture(report,checkpoint=cp,layer_source=source)
    assert check() is None
    state['chain']['first_suffix_checked']=False
    with pytest.raises(ValueError,match='capture|chain|layer'):check()
    state['chain']['first_suffix_checked']=True;state['events']=969
    with pytest.raises(ValueError,match='capture|layer'):check()


def test_baseline_header_rejects_a_different_source_or_intervened_trace(layers):
    cp={'path':'/checkpoint','bytes':1,'mtime_ns':2}
    extra={'fold_spatial':True,'effective_projection':'per_camera_hwd','source_revision':'b'*40,
        'precondition':{'operator_exact':True,'checkpoint':cp,'job_id':'7','source_revision':'b'*40}}
    report={'engine':'native','job_id':'7','artifact_complete':True,'execution_mode':'diagnostic_eager',
        'diagnostic_cleanup':{'passed':True},'config':{'use_cuda_graph':False,'numerical_profile':'official_local_fp32'},
        'experimental_projection':extra,'historical_official_reference':{'job_id':'308690'},
        'prior_graph_output':{'exact':True},'frozen_control_evidence':{'native_replay_exact':True,'prefix_preserved':True}}
    check=lambda:layers.admit_baseline_header(report,job_id='7',source_revision='b'*40,checkpoint=cp)
    assert check() is None
    extra['source_revision']='c'*40
    with pytest.raises(ValueError,match='baseline|source'):check()
    extra['source_revision']='b'*40;report['suffix_order_control']={}
    with pytest.raises(ValueError,match='baseline|order'):check()


def test_raw_publication_failure_cannot_leave_final_success(layers,tmp_path,monkeypatch):
    import safetensors.torch
    def fail(*args,**kwargs):raise OSError('raw write failed')
    monkeypatch.setattr(safetensors.torch,'save_file',fail)
    with pytest.raises(OSError,match='raw write'):layers.publish(tmp_path,{'x':torch.ones(1)},{'status':'complete'})
    assert not (tmp_path/'result.json').exists()


def test_help_and_allocation_guard_before_inputs_or_output(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert result.returncode==0 and '--baseline-job' in result.stdout
    out=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--reference','/missing','--checkpoint','/missing',
        '--baseline','/missing','--baseline-job','313766','--official-attention-source','/missing',
        '--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not out.exists()
