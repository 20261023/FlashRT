import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/mean_launch.py'


@pytest.fixture
def launch():
    s=importlib.util.spec_from_file_location('hy_mean_launch_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def trace_fixture():
    return {'traceEvents':[
        {'cat':'user_annotation','name':'mean.probe.n0','ph':'X'},
        {'cat':'cpu_op','name':'aten::mean','ph':'X','args':{'External id':7}},
        {'cat':'kernel','name':'void at::native::reduce_kernel<512, 1, MeanOps<float>>(...)','ph':'X',
         'args':{'grid':[3,1,1],'block':[32,16,1],'External id':7,'stream':7}}]}


def test_trace_preserves_all_full_kernel_names_and_launches(launch):
    t=trace_fixture();second=copy.deepcopy(t['traceEvents'][-1]);second['name']='second full kernel';t['traceEvents'].append(second)
    r=launch.summarize_trace(t,'mean.probe.n0')
    assert r['cpu_mean_events']==1 and r['kernel_count']==2
    assert r['kernels'][0]['name']=='void at::native::reduce_kernel<512, 1, MeanOps<float>>(...)'
    assert r['kernels'][0]['grid']==[3,1,1] and r['kernels'][0]['block']==[32,16,1]
    assert r['kernels'][1]['name']=='second full kernel' and r['runtime_reduce_config_observed'] is False


def test_actual_chrome_gpu_projection_does_not_duplicate_the_cpu_range(launch):
    # Actual2.7 export315596 has one CPU range and its same-named GPU projection.
    t=trace_fixture();t['traceEvents'].append({'cat':'gpu_user_annotation','name':'mean.probe.n0','ph':'X','args':{'External id':1}})
    result=launch.summarize_trace(t,'mean.probe.n0')
    assert result['cpu_mean_events']==1 and result['kernel_count']==1
    assert result['kernels'][0]['grid']==[3,1,1]


@pytest.mark.parametrize('bad',['label','cpu','kernel','grid','block','duplicate_label','empty'])
def test_incomplete_trace_is_not_a_successful_launch_observation(launch,bad):
    t=trace_fixture()
    if bad=='label':t['traceEvents'][0]['name']='other'
    if bad=='cpu':t['traceEvents'][1]['name']='aten::sum'
    if bad=='kernel':t['traceEvents'].pop()
    if bad=='grid':del t['traceEvents'][-1]['args']['grid']
    if bad=='block':t['traceEvents'][-1]['args']['block']=[0,1,1]
    if bad=='duplicate_label':t['traceEvents'].append(copy.deepcopy(t['traceEvents'][0]))
    if bad=='empty':t={}
    with pytest.raises(ValueError,match='trace|mean|kernel|launch|label'):launch.summarize_trace(t,'mean.probe.n0')


@pytest.mark.parametrize('engine',['official','native'])
@pytest.mark.parametrize('bad',['none','dummy','dtype','profiled_value','square','finite'])
def test_complete_raw_capture_must_reproduce_same_runtime_before_and_during_profile(launch,engine,bad):
    from test_hyvla_expert_rms_handoff_example import history_fixture
    old,ro,new,rn=history_fixture(launch.handoff);raw={}
    for p in launch.rms.PROBES:
        for n in range(4):
            name=f'{p}.n{n}';base='own.'+name;ref=old if engine=='official' else new;vbase=base if engine=='official' else 'fixed_old.'+name
            raw[name+'.float_input']=old[base+'.stage.float_input'].clone();raw[name+'.square']=old[base+'.stage.square'].clone()
            raw[name+'.unprofiled_variance']=ref[vbase+'.stage.variance'].clone();raw[name+'.profiled_variance']=ref[vbase+'.stage.variance'].clone()
    key='baseline.s0.l1.n0'
    if bad=='dummy':del raw[key+'.profiled_variance'];raw['dummy']=torch.zeros(1)
    if bad=='dtype':raw[key+'.square']=raw[key+'.square'].bfloat16()
    if bad=='profiled_value':raw[key+'.profiled_variance']+=1
    if bad=='square':raw[key+'.square']+=1
    if bad=='finite':raw[key+'.float_input'].fill_(float('nan'))
    if bad=='none':assert launch.validate_raw(raw,old,new,engine=engine)=={'tensors':64,'profiled_means_exact':16,'unprofiled_means_exact':16}
    else:
        with pytest.raises(ValueError,match='raw|mean|role|profile'):launch.validate_raw(raw,old,new,engine=engine)


def test_live_layout_records_actual_noncontiguous_view_and_alignment(launch):
    base=torch.zeros(25);x=base[1:].reshape(2,3,4).transpose(0,1)
    r=launch.layout_info(x)
    assert r['shape']==[3,2,4] and r['live_stride']==[4,12,1] and r['storage_offset']==1
    assert r['data_ptr_mod16']==(base.data_ptr()+4)%16 and r['dtype']=='torch.float32'
    assert 'data_ptr' not in r


def test_help_and_allocation_guard_precede_history_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--engine' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--engine','native','--history-root','/missing','--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
