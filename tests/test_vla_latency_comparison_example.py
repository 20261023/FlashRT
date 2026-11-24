import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'examples/vla_autotune/latency_comparison.py'


@pytest.fixture
def comparison():
    if not MODULE.exists():pytest.fail('Latency evidence helpers not implemented')
    s=importlib.util.spec_from_file_location('vla_latency_test',MODULE);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def row(origin='project'):
    return {'model':'hyvla','origin':origin,'implementation':'Hy BF16 Euler10 graph' if origin=='project' else 'author HyVLA prepare + sample_actions',
        'evidence_kind':'measured','status':'complete','hardware':'NVIDIA A800-SXM4-80GB','job_id':'99','precision':'bf16',
        'input_signature':{'batch':1,'cameras':3,'history':6,'image_hw':[480,640],'chunk':40,'action_dim':32},
        'sampler':'euler','nfe':10,'boundary':'host_observation_noise_to_CPU_raw_chunk','statistic':'median','latency_ms':20.,'p95_ms':22.,
        'source_url':'https://github.com/20261023/FlashRT' if origin=='project' else 'https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA',
        'selection_status':'baseline' if origin=='project' else 'not_applicable'}


@pytest.mark.parametrize('implementation',['FlashRT BF16 Euler10','project Torch graph Euler20'])
def test_project_execution_cannot_be_relabelled_as_author_Native(comparison,implementation):
    value=row('author');value['implementation']=implementation
    with pytest.raises(ValueError,match='Native|author|implementation'):comparison.validate_row(value)


@pytest.mark.parametrize('field,value',[('hardware','RTX4090'),('nfe',8),('statistic','mean'),('boundary','GPU_only'),('precision','fp16'),('job_id','100')])
def test_different_hardware_work_or_boundary_prevents_speedup_ratio(comparison,field,value):
    ours=row();official=row('author');official[field]=value
    result=comparison.comparison_ratio(ours,official)
    assert result['ratio'] is None and result['reasons']


def test_matched_official_selftest_ratio_has_explicit_not_quality_interpretation(comparison):
    ours=row();official=row('author');official['latency_ms']=60.;official['p95_ms']=65.
    result=comparison.comparison_ratio(ours,official)
    assert result['ratio']==3. and result['reasons']==[] and 'quality' in result['interpretation']


def test_not_reported_is_NR_not_zero_and_baseline_is_not_winner(comparison):
    ours=row();official=row('author');official.update(evidence_kind='not_reported',status='not_reported',latency_ms=None,p95_ms=None,hardware=None,statistic=None)
    table=comparison.render_table([ours,official])
    assert 'NR' in table and '基线' in table and '最优' not in table
    assert comparison.comparison_ratio(ours,official)['ratio'] is None


@pytest.mark.parametrize('bad',['zero','nan','incomplete','private'])
def test_invalid_timing_or_private_source_never_enters_public_table(comparison,bad):
    value=row()
    if bad=='zero':value['latency_ms']=0.
    if bad=='nan':value['latency_ms']=float('nan')
    if bad=='incomplete':value['status']='failed'
    if bad=='private':value['source_url']='/data/run01/private/source'
    with pytest.raises(ValueError,match='latency|complete|source|finite'):comparison.validate_row(value)


def test_timer_excludes_warmup_and_counts_balanced_actual_requests(comparison):
    calls=[];syncs=[];clock=iter([0,1000000,2000000,4000000,5000000,8000000,9000000,13000000])
    def infer(request):calls.append(request);return np.full((2,3),request,dtype=np.float32)
    result=comparison.measure_calls(infer,[0,1],warmup=2,repeats=4,synchronize=lambda:syncs.append(True),clock_ns=lambda:next(clock),expected_shape=(2,3))
    assert calls==[0,1,0,1,0,1] and result['latency_ms']==[1.,2.,3.,4.] and result['request_indices']==[0,1,0,1]
    assert len(result['actions'])==4 and result['statistics']['p50_ms']==2.5 and result['statistics']['p95_ms']==pytest.approx(3.85)


@pytest.mark.parametrize('bad',['coverage','nonfinite','shape','body'])
def test_failed_timer_preserves_partial_and_never_completes(comparison,bad):
    count=[]
    def infer(request):
        count.append(request)
        if bad=='body' and len(count)==2:raise RuntimeError('official call failed')
        if bad=='nonfinite':return np.full((2,3),float('nan'),dtype=np.float32)
        if bad=='shape':return np.zeros((1,3),dtype=np.float32)
        return np.zeros((2,3),dtype=np.float32)
    with pytest.raises((ValueError,RuntimeError),match='coverage|finite|shape|failed') as caught:
        comparison.measure_calls(infer,[0,1],warmup=0,repeats=3 if bad=='coverage' else 4,synchronize=lambda:None,expected_shape=(2,3))
    if bad!='coverage':assert hasattr(caught.value,'latency_partial') and not caught.value.latency_partial['complete']


@pytest.fixture
def official():
    from flash_rt.vla.author_adapters import hyvla
    return hyvla


def test_original_author_prepare_and_sampler_receive_host_converted_inputs_and_fixed_noise(official):
    torch = pytest.importorskip('torch')
    calls=[]
    class Model:
        def sample_actions(self,*,images,img_masks,state,lang_tokens,lang_masks,noise):
            calls.append('author_sample')
            assert images[0].shape==(1,6,3,2,3) and images[0].dtype==torch.bfloat16 and float(images[0][0,0,0,0,0])==1.
            assert state.shape==(1,20) and lang_tokens.tolist()==[[7,8]] and noise.shape==(1,40,32)
            return noise+2
    class Policy:
        model=Model()
        def prepare_images(self,batch):
            calls.append('author_images');assert batch['task']==['test task']
            return [batch['observation.images.top_head']], [torch.ones(1,dtype=torch.bool)]
        def prepare_state(self,batch):calls.append('author_state');return batch['observation.state']
        def prepare_language(self,batch):
            calls.append('author_language');return torch.tensor([[7,8]]),torch.ones(1,2,dtype=torch.bool),None
    arrays={'prompt':np.array('test task'),'state':np.zeros(20,dtype=np.float32),'noise':np.ones((40,32),dtype=np.float32)}
    arrays.update({name:np.full((6,2,3,3),255,dtype=np.uint8) for name in ('top_head','hand_left','hand_right')})
    result=official.author_request(Policy(),arrays,device='cpu')
    assert result.dtype==np.float32 and result.shape==(40,32) and np.all(result==3.)
    assert calls==['author_images','author_state','author_language','author_sample'] and np.all(arrays['noise']==1.)


@pytest.mark.parametrize('failure',['none','wrong_count','body'])
def test_untimed_NFE_observation_restores_model_method_before_formal_timing(official,failure):
    class Model:
        def denoise_step(self,x):return x+1
    model=Model();old=dict(model.__dict__)
    def infer(request):
        for _ in range(2 if failure=='wrong_count' else 10):model.denoise_step(1)
        if failure=='body':raise RuntimeError('author control body failed')
        return np.zeros((40,32),dtype=np.float32)
    call=lambda:official.check_nfe_once(model,infer,None,expected=10,synchronize=lambda:None)
    if failure=='none':assert call()['observed_calls']==10
    else:
        with pytest.raises((ValueError,RuntimeError),match='NFE|body'):call()
    assert model.__dict__==old and model.denoise_step(3)==4


@pytest.mark.parametrize('bad',['none','root','missing_weight','wrong_source'])
def test_official_source_and_complete_load_contract_do_not_admit_project_replacement(official,tmp_path,bad):
    source=tmp_path/'official';(source/'hy_vla').mkdir(parents=True)
    for name in ('__init__.py','modeling_hy_vla.py','configuration_hy_vla.py'):(source/'hy_vla'/name).write_text('# source')
    (source/'AUTHOR_GIT_COMMIT').write_text('af57e7507ec5964b52fdf6296741e553cbcd3288\n')
    if bad=='root':(source/'AUTHOR_GIT_COMMIT').write_text('wrong')
    paths={'package':source/'hy_vla/__init__.py','policy':source/'hy_vla/modeling_hy_vla.py','config':source/'hy_vla/configuration_hy_vla.py'}
    if bad=='wrong_source':paths['policy']=tmp_path/'flash_rt.py';paths['policy'].write_text('# not author')
    call=lambda:official.admit_author(source,paths,missing=['model.dual_tower.expert.lm_head.weight'] if bad!='missing_weight' else ['model.visual.patch.weight'],unexpected=[])
    if bad=='none':assert call()['missing_inference_keys']==[]
    else:
        with pytest.raises(ValueError,match='author|source|weight|Native'):call()


def test_official_CLI_requires_real_allocation_before_loading_source_or_weights(tmp_path):
    from examples.vla_autotune.author_baseline import _require_step
    env = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
    from unittest.mock import patch
    with patch.dict(os.environ, env, clear=True), pytest.raises(RuntimeError, match='allocated'):
        _require_step()


@pytest.mark.parametrize('bad',['none','samples','NFE','cleanup','coverage','source','profile','graph','framework','checkpoint','observed-profile'])
def test_selected_worker_row_requires_complete_bound_measurement_not_a_label(comparison,bad):
    workload={'requests':[{'request_id':'a'},{'request_id':'b'}]}
    selection=row();selection.update(backend='flashrt',nfe=10,source_ref='git:unit')
    backend={'model':'hyvla','backend':'flashrt','precision':'bf16','sampler':'euler','num_steps':10,'framework':'torch',
        'checkpoint':'/private/checkpoint','hardware':'cuda_sm80','probeflow':None,'options':{'numerical_profile':'semantic_bf16','use_cuda_graph':True}}
    selection['backend_config']=copy.deepcopy(backend)
    result={'state':'measured','job_id':'99','gpu':{'name':'NVIDIA A800-SXM4-80GB'},'workload_request_ids':['a','b'],
        'candidate':{'source_ref':'git:unit','backend':backend},
        'backend_metadata':{'model':'hyvla','backend':'flashrt','precision':'bf16'},'quality':{'status':'passed'},
        'measurement':{'latency_ms':[1.,2.,3.,4.],'repeats':4,'sampler_metadata':[{'sampler':'euler','nfe':10,'numerical_profile':'semantic_bf16'} for _ in range(4)]}}
    if bad=='samples':result['measurement']['repeats']=5
    if bad=='NFE':result['measurement']['sampler_metadata'][2]['nfe']=8
    if bad=='cleanup':result['cleanup_failure']='not drained'
    if bad=='coverage':result['workload_request_ids']=['a']
    if bad=='source':result['candidate']['source_ref']='old source'
    if bad=='profile':backend['options']['numerical_profile']='official_local_fp32'
    if bad=='graph':backend['options']['use_cuda_graph']=False
    if bad=='framework':backend['framework']='jax'
    if bad=='checkpoint':backend['checkpoint']='/private/another-checkpoint'
    if bad=='observed-profile':result['measurement']['sampler_metadata'][2]['numerical_profile']='official_local_fp32'
    call=lambda:comparison.selected_measurement(result,workload,selection)
    if bad=='none':
        value=call();assert value['latency_ms']==2.5 and value['nfe']==10 and value['origin']=='project'
        assert 'backend_config' not in value
    else:
        message = 'frontend|framework|selected' if bad == 'framework' else 'measurement|NFE|cleanup|coverage|source|selected'
        with pytest.raises(ValueError,match=message):call()


def test_fast_selected_graph_requires_observed_execution_coverage(comparison):
    selection=row();selection.update(model='fastwam',backend='torch',nfe=20,source_ref='git:unit',
        backend_config={'model':'fastwam','backend':'torch','precision':'bf16','sampler':'euler','num_steps':20,'framework':'torch',
            'checkpoint':'/private/FastWAM.pt','hardware':'cuda_sm80','options':{'use_cuda_graph':True}})
    record={'state':'measured','job_id':'99','gpu':{'name':'NVIDIA A800-SXM4-80GB'},'workload_request_ids':['a'],
        'candidate':{'source_ref':'git:unit','backend':copy.deepcopy(selection['backend_config'])},
        'backend_metadata':{'model':'fastwam','backend':'torch','precision':'bf16','execution_coverage':{'cuda_graph':False}},
        'quality':{'status':'passed'},'measurement':{'latency_ms':[1.],'repeats':1,'sampler_metadata':[{'sampler':'euler','nfe':20}]}}
    with pytest.raises(ValueError,match='selected.*graph'):comparison.selected_measurement(record,{'requests':[{'request_id':'a'}]},selection)


def test_official_output_arrays_are_saved_before_completed_JSON(tmp_path,monkeypatch):
    from flash_rt.vla.autotune.author import publish_outputs
    samples={'latency_ms':[1.,2.],'request_ids':['a','a'],'actions':[np.zeros((40,32),dtype=np.float32)]*2}
    def broken(*a,**kw):raise OSError('Native raw output failed')
    monkeypatch.setattr(np,'savez_compressed',broken)
    with pytest.raises(OSError,match='raw output'):
        publish_outputs(tmp_path,{'status':'complete','identity':{'output_domain':'normalized'}},**samples)
    assert not (tmp_path/'result.json').exists()
