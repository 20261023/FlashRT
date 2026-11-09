import ast
import copy
from contextlib import nullcontext
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT/'examples/vla_autotune/hy_trace/proj1_camera_ablation.py'


def ablation():
    if not SCRIPT.exists(): pytest.fail('camera-PROJ1 ablation module is missing')
    spec = importlib.util.spec_from_file_location('hy_camera_ablation', SCRIPT)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_camera_projection_preserves_values_order_and_native_singleton_strides():
    module = ablation()
    base = torch.arange(36., dtype=torch.float64).reshape(3, 2, 2, 3)
    native = torch.empty_strided(base.shape, (12, 2, 1, 4), dtype=base.dtype); native.copy_(base)
    observed = []
    weight = torch.tensor([[1., 0., -1.], [.5, 1., 0.]], dtype=base.dtype)
    result = module.per_camera_proj1(native, weight, torch.tensor([1., -2.], dtype=base.dtype),
        observer=lambda view: observed.append((tuple(view.shape), view.stride())))
    expected = torch.stack([-torch.ones(12), 4.5*torch.arange(12.)-1], -1).double().reshape(3, 2, 2, 2)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)
    torch.testing.assert_close(native, base, atol=0, rtol=0)
    assert observed == [((1, 2, 2, 3), (12, 2, 1, 4))]*3


@pytest.mark.parametrize('batch', [1, 3])
def test_copied_merger_keeps_production_pooler_gelu_and_proj2_math_batched(batch):
    module = ablation()
    tree = ast.parse((ROOT/'flash_rt/models/hyvla/pipeline_thor.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name=='HyVLAThorBF16Pipeline')
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name=='merger_forward')
    scope = {'torch': torch, 'F': torch.nn.functional}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'production_merger', 'exec'), scope)
    def values(shape, offset):
        return ((torch.arange(np.prod(shape), dtype=torch.float64)+offset)%7-3).reshape(shape)/8
    weights = SimpleNamespace(_mg_proj1_w=values((3,2),1),_mg_proj1_b=values((3,),2),
        _mg_pred0_w=values((4,6),3),_mg_pred0_b=values((4,),4),
        _mg_pred2_w=values((3,4),5),_mg_pred2_b=values((3,),6),
        _mg_proj2_w=values((3,3),7),_mg_proj2_b=values((3,),8))
    owner = SimpleNamespace(W=weights)
    x = values((batch,16,2),9)
    expected = scope['merger_forward'](owner,x,grid=4)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,acc_events=True) as profile:
        actual = module.camera_merger(owner,x,grid=4)
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0)
    assert actual.abs().sum()>0 and owner.W is weights
    shapes=[event.input_shapes[0] for event in profile.events() if event.name=='aten::linear']
    assert shapes[:batch]==[[1,4,4,2]]*batch
    assert shapes[batch:]==[[batch,2,2,4,6],[batch,2,2,4,4],[batch,2,2,3]]


class FakeFrontend:
    def __init__(self):
        class Pipe:
            def merger_forward(self,x,grid=14): return x
        self.pipe=Pipe();self.numerical_profile='official_local_fp32'
        self._vit_graph_cache={};self._graph_cache={}


class FakeBackend:
    def __init__(self,events,close_error=False):
        self.frontend=FakeFrontend();self.events=events;self.close_error=close_error
    def close(self):
        self.events.append('close')
        if self.close_error: raise RuntimeError('close failed')


def config():
    return SimpleNamespace(model='hyvla',sampler='euler',num_steps=10,
                           options={'numerical_profile':'official_local_fp32','use_cuda_graph':True})


def test_failed_projection_precondition_never_constructs_full_backend():
    module=ablation();events=[]
    def forbidden(cfg): events.append('load');raise AssertionError('must not load')
    with pytest.raises(ValueError,match='precondition'):
        with module.scoped_backend(forbidden,config(),arm='camera_proj1',preflight={'operator_exact':False},
            state={},synchronize=lambda:None,expected_frontend_type=FakeFrontend): pass
    assert events==[]


def test_binding_is_instance_local_before_prepare_and_graphs_reset_before_restore_close():
    module=ablation();events=[];backend=FakeBackend(events);other=FakeFrontend();state={}
    class Graph:
        def __init__(self,name):self.name=name
        def reset(self):
            assert backend.frontend.pipe.merger_forward.__func__ is module.camera_merger
            events.append(self.name)
    original_linear=torch.nn.functional.linear
    with module.scoped_backend(lambda cfg:backend,config(),arm='camera_proj1',preflight={'operator_exact':True},
        state=state,synchronize=lambda:events.append('sync'),expected_frontend_type=FakeFrontend) as actual:
        assert actual is backend and backend.frontend.pipe.merger_forward.__func__ is module.camera_merger
        assert 'merger_forward' not in other.pipe.__dict__
        a,b,c=Graph('vit'),Graph('prefix'),Graph('velocity')
        backend.frontend._vit_graph_cache['one']={'graph':a}
        backend.frontend._graph_cache['one']={'prefix_graph':b,'velocity_graphs':{1.:c,.5:c}}
    assert events==['sync','vit','prefix','velocity','close']
    assert 'merger_forward' not in backend.frontend.pipe.__dict__
    assert not backend.frontend._graph_cache and not backend.frontend._vit_graph_cache
    assert state['cleanup']['passed'] is True and torch.nn.functional.linear is original_linear


@pytest.mark.parametrize('failure', ['body','sync','reset','close'])
def test_scope_failure_restores_binding_and_never_marks_cleanup_failure_success(failure):
    module=ablation();events=[];backend=FakeBackend(events,close_error=failure=='close');state={}
    class Graph:
        def reset(self):
            events.append('reset')
            if failure=='reset':raise RuntimeError('reset failed')
    def sync():
        events.append('sync')
        if failure=='sync':raise RuntimeError('sync failed')
    with pytest.raises(RuntimeError,match=failure):
        with module.scoped_backend(lambda cfg:backend,config(),arm='camera_proj1',preflight={'operator_exact':True},
            state=state,synchronize=sync,expected_frontend_type=FakeFrontend):
            backend.frontend._vit_graph_cache['one']={'graph':Graph()}
            if failure=='body':raise RuntimeError('body failed')
    assert 'merger_forward' not in backend.frontend.pipe.__dict__ and 'close' in events
    assert state['cleanup']['passed'] is (failure=='body')
    if failure!='body':assert state['failed_owners']


def test_first_publication_is_tagged_after_cleanup_and_exports_restored_on_writer_failure():
    module=ablation();events=[];backend=FakeBackend(events); published=[]
    vla=SimpleNamespace(load_backend=lambda cfg:backend)
    original_factory=vla.load_backend
    def writer(path,record):
        assert events[-1]=='close'
        assert record['experimental_merger_proj1']['cleanup']['passed'] is True
        published.append(record.copy());raise RuntimeError('writer failed')
    helper=SimpleNamespace(write_json=writer)
    def run(argv):
        with vla.load_backend(config()):pass
        helper.write_json(Path('/unused/result.json'),completed_record(True,'camera_proj1'))
        return 0
    helper.main=run
    with pytest.raises(RuntimeError,match='writer failed'):
        module.run_wrapped_validator(helper,vla,[],arm='camera_proj1',preflight={'operator_exact':True},
            source_revision='0'*40,synchronize=lambda:events.append('sync'),expected_frontend_type=FakeFrontend)
    assert vla.load_backend is original_factory and helper.write_json is writer
    assert len(published)==1 and published[0]['experimental_merger_proj1']['arm']=='camera_proj1'


def test_unexpected_frontend_is_rejected_but_its_constructed_backend_is_closed():
    module=ablation();events=[];backend=FakeBackend(events);backend.frontend=object();state={}
    with pytest.raises(ValueError,match='frontend'):
        with module.scoped_backend(lambda cfg:backend,config(),arm='camera_proj1',preflight={'operator_exact':True},
            state=state,synchronize=lambda:events.append('sync'),expected_frontend_type=FakeFrontend):pass
    assert 'close' in events and state['cleanup']['passed'] is True


def test_body_error_remains_primary_if_cleanup_also_fails():
    module=ablation();events=[];backend=FakeBackend(events,close_error=True);state={}
    with pytest.raises(RuntimeError,match='body failed') as error:
        with module.scoped_backend(lambda cfg:backend,config(),arm='camera_proj1',preflight={'operator_exact':True},
            state=state,synchronize=lambda:events.append('sync'),expected_frontend_type=FakeFrontend):
            raise RuntimeError('body failed')
    assert state['cleanup']['passed'] is False and state['failed_owners']
    assert any('cleanup also failed' in note for note in error.value.__notes__)


def completed_record(passed=True,arm='baseline',variant='baseline'):
    def metric(width):
        return {'shape':[40,width],'exact':passed,'allclose':passed,'atol':.01,'rtol':.01,
                'outside_tolerance':0 if passed else 1,'elements':40*width,
                'max_abs':0. if passed else .02,'mean_abs':0.,'rmse':0.,
                'max_relative_to_reference_floor_1e_12':0.,'cosine':1. if passed else .998}
    row={'sampler':'euler','scheduled_steps':10,'nfe':10,
         'timesteps':[1.+i*(-1./10) for i in range(10)],'numerical_profile':'official_local_fp32'}
    return {'schema_version':1,'status':'runtime_and_recorded_official_gate_passed' if passed else 'runtime_passed_oracle_mismatch',
        'runtime_checks_passed':True,'finite':True,'shape':[40,32],'public_shape':[40,20],'dtype':'float32',
        'action_domain':'normalized-padded-model-actions','profile':'flashrt_official_local_fp32_hooks_not_exact',
        'official_reference_profile':'recorded_official_bf16_weights_local_fp32_rope_attention',
        'config':{'sampler':'euler','num_steps':10,'max_steps':None,'epsilon':.008,'probe_horizon':.5,
                  'step_increment':2,'use_graph':True,'numerical_profile':'official_local_fp32'},
        'checkpoint':{'root':'/checkpoint','weights':{'path':'/checkpoint/model.safetensors','bytes':100,'mtime_ns':1}},
        'fixture':{'path':f'/fixture/{variant}.npz','bytes':100,'mtime_ns':2,'camera_order':['a','b','c'],
                   'manifest':{'variant':variant,'source_revision':'a'*40,'job_id':'1'}},
        'runtime':{'python':'3.11.14','executable':'/env/python','torch':'2.11.0+cu128','cuda':'12.8','numpy':'2.2.6'},
        'gpu':{'name':'NVIDIA A800-SXM4-80GB','capability':[8,0]},
        'model_sources':{'pipeline':{'path':'/new/pipeline.py','bytes':12,'mtime_ns':3}},
        'fixed_noise_replay':{'exact':True},'sampler_metadata_replay_equal':True,
        'sampler_metadata':[copy.deepcopy(row),copy.deepcopy(row)],
        'graph_eager_check':{'status':'passed' if arm=='camera_proj1' else 'not_requested',
                             'passed':True if arm=='camera_proj1' else None},
        'official_oracle':{'eligible_same_schedule':True,'same_action_gate_passed':passed,'public20_gate_passed':passed,
            'reference_profile':'recorded_official_bf16_weights_local_fp32_rope_attention',
            'gate':{'atol':.01,'rtol':.01,'minimum_cosine':.999,'legacy_gate_domain':'normalized_padded_40x32'},
            'actions32':metric(32),'actions20':metric(20)},
        'job_id':'7',
        'experimental_merger_proj1':{'arm':arm,'source_revision':'0'*40,
            'precondition':{'kind':'hy_camera_proj1_exact_frozen_precondition','operator_exact':True,
                            'job_id':'7','source_revision':'0'*40},'cleanup':{'passed':True},
                                     'baseline_prior':{'passed':True,'actions_exact':True}}}


def saved_arrays():
    return {'actions32':np.zeros((40,32),np.float32),'actions20':np.zeros((40,20),np.float32),
        'noise':np.ones((40,32),np.float32),'input_state':np.zeros(20,np.float32),'input_prompt':np.asarray('pick'),
        'oracle_actions32':np.zeros((40,32),np.float32),'oracle_actions20':np.zeros((40,20),np.float32)}


def test_baseline_compatibility_reports_source_difference_without_faking_it_equal():
    module=ablation();a=completed_record();b=copy.deepcopy(a)
    b['model_sources']['pipeline']['path']='/old/pipeline.py'
    report=module.compatible_prior(a,saved_arrays(),b,saved_arrays())
    assert report['passed'] and report['actions_exact'] and report['inputs_exact']
    assert report['model_sources_same'] is False and a['model_sources']!=b['model_sources']


@pytest.mark.parametrize('change', ['noise','actions','oracle','checkpoint','runtime','fixture'])
def test_baseline_incompatibility_blocks_candidate_interpretation(change):
    module=ablation();a=completed_record();b=copy.deepcopy(a);aa=saved_arrays();bb=saved_arrays()
    if change=='noise':aa['noise'][0,0]+=1
    if change=='actions':aa['actions32'][0,0]=.01;aa['actions20'][0,0]=.01
    if change=='oracle':aa['oracle_actions32'][0,0]=.01;aa['oracle_actions20'][0,0]=.01
    if change=='checkpoint':a['checkpoint']['weights']['mtime_ns']=9
    if change=='runtime':a['runtime']['torch']='different'
    if change=='fixture':a['fixture']['mtime_ns']=9
    with pytest.raises(ValueError):module.compatible_prior(a,aa,b,bb)


def test_exit3_only_means_verified_numerical_oracle_mismatch():
    module=ablation()
    assert module.classify_exit(0,completed_record())=='passed'
    assert module.classify_exit(3,completed_record(False))=='oracle_mismatch'
    for code,record in [(3,completed_record()),(0,completed_record(False)),(1,completed_record(False))]:
        with pytest.raises(ValueError):module.classify_exit(code,record)
    bad=completed_record(False);bad['runtime_checks_passed']=False
    with pytest.raises(ValueError):module.classify_exit(3,bad)
    bad=completed_record(False);bad['official_oracle']['gate']['atol']=.02
    with pytest.raises(ValueError):module.classify_exit(3,bad)


def test_aggregate_never_turns_partial_or_failed_candidate_gates_into_success():
    module=ablation()
    arms={'baseline':{},'camera_proj1':{}}
    for variant in module.VARIANTS:
        passed=variant not in ('history','state','noise')
        arms['baseline'][variant]=(0 if passed else 3,completed_record(passed,variant=variant))
        arms['camera_proj1'][variant]=(0,completed_record(True,'camera_proj1',variant=variant))
    summary=module.aggregate(arms)
    assert summary['exit_code']==0 and summary['baseline_gate_passes']==4 and summary['candidate_gate_passes']==7
    arms['camera_proj1']['history']=(3,completed_record(False,'camera_proj1',variant='history'))
    assert module.aggregate(arms)['exit_code']!=0
    del arms['camera_proj1']['noise']
    with pytest.raises(ValueError,match='complete|missing'):module.aggregate(arms)


@pytest.mark.parametrize('code', [0,1,3])
def test_actual_cli_entrypoint_propagates_main_returncode(code):
    tree=ast.parse(SCRIPT.read_text())
    entry=next(node for node in tree.body if isinstance(node,ast.If) and
        isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__')
    with pytest.raises(SystemExit) as error:
        exec(compile(ast.Module(body=[entry],type_ignores=[]),'actual_cli_entry','exec'),
             {'__name__':'__main__','main':lambda:code})
    assert error.value.code==code


@pytest.mark.parametrize('field,flag', [('checkpoint','--checkp'),('output','--out')])
def test_validator_abbreviations_cannot_override_admitted_paths(tmp_path,field,flag):
    module=ablation()
    helper=module._module(ROOT/'examples/vla_autotune/validate_hyvla.py','abbreviation_contract_validator')
    checkpoint,output=tmp_path/'A',tmp_path/'resultA'
    point=helper.parse_args(['--checkpoint',str(checkpoint),'--output',str(output),'--fixture','/fixture.npz',
        '--sampler','euler','--num-steps','10','--numerical-profile','official_local_fp32','--require-oracle',
        flag,str(tmp_path/'B')])
    assert getattr(point,field).resolve()==(tmp_path/'B').resolve()
    with pytest.raises(ValueError,match=field):
        module.validate_forwarded_point(point,checkpoint=checkpoint,output=output)


def test_factory_boundary_rechecks_checkpoint_before_model_construction(tmp_path):
    module=ablation();events=[]
    def forbidden(cfg):events.append('load');raise AssertionError('wrong model constructed')
    vla=SimpleNamespace(load_backend=forbidden);helper=SimpleNamespace(write_json=lambda *args:None)
    cfg=config();cfg.checkpoint=str(tmp_path/'B')
    def run(argv):
        with vla.load_backend(cfg):pass
        return 0
    helper.main=run
    with pytest.raises(ValueError,match='checkpoint'):
        module.run_wrapped_validator(helper,vla,[],arm='camera_proj1',preflight={'operator_exact':True},
            source_revision='0'*40,synchronize=lambda:None,expected_frontend_type=FakeFrontend,
            checkpoint_path=tmp_path/'A')
    assert events==[] and vla.load_backend is forbidden


@pytest.mark.parametrize('change', ['duplicate','cross_arm_fixture','checkpoint','runtime','proof','source'])
def test_seven_keys_do_not_hide_duplicate_or_cross_arm_execution(change):
    module=ablation();arms={'baseline':{},'camera_proj1':{}}
    for variant in module.VARIANTS:
        arms['baseline'][variant]=(0,completed_record(True,variant=variant))
        arms['camera_proj1'][variant]=(0,completed_record(True,'camera_proj1',variant=variant))
    record=arms['camera_proj1']['history'][1]
    if change=='duplicate':
        for arm in arms:
            for _,row in arms[arm].values():row['fixture']=copy.deepcopy(arms[arm]['baseline'][1]['fixture'])
    if change=='cross_arm_fixture':record['fixture']['path']='/fixture/baseline.npz'
    if change=='checkpoint':record['checkpoint']['weights']['mtime_ns']=100
    if change=='runtime':record['runtime']['torch']='wrong'
    if change=='proof':record['experimental_merger_proj1']['precondition']['job_id']='8'
    if change=='source':record['experimental_merger_proj1']['source_revision']='1'*40
    with pytest.raises(ValueError):module.aggregate(arms)


def test_actual_gpu_profile_confirmation_rejects_empty_or_annotation_only_evidence():
    module=ablation()
    for summary in ({'cuda_activity_observed':False,'cuda_kernel_names_in_region':{}},
                    {'cuda_activity_observed':True,'cuda_kernel_names_in_region':{'frozen_auto_sdpa_no_timing_acceptance':1}}):
        with pytest.raises(ValueError,match='CUDA|kernel|annotation|unverified'):
            module.validate_profile_confirmation(summary)
    assert module.validate_profile_confirmation({'cuda_activity_observed':True,
        'cuda_kernel_names_in_region':{'actual_kernel':1},'ignored_cuda_annotation_names':{'region':1}}) is None


def test_help_and_allocation_guard_precede_outputs_or_model_load(tmp_path):
    env={key:value for key,value in os.environ.items() if not key.startswith('SLURM_')}
    help_result=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert help_result.returncode==0
    output=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(SCRIPT),'--arm','preflight','--checkpoint','/missing',
        '--official-trace','/missing','--native-report','/missing','--source-revision','0'*40,
        '--output',str(output)],env=env,text=True,capture_output=True)
    assert result.returncode!=0 and 'allocated GPU srun' in result.stderr and not output.exists()
