import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1];SCRIPT=ROOT/'examples/vla_autotune/hy_trace/mean_tree.py'


@pytest.fixture
def tree():
    s=importlib.util.spec_from_file_location('hy_mean_tree_test',SCRIPT)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


@pytest.mark.parametrize('width,positions,old_flags,old_value,new_value',[
    (128,[32,64,96],(False,False),131072.,131072.015625),
    (1024,[4,12],(True,False),16384.,16384.001953125)])
def test_hand_derived_roundoff_distinguishes_grouping_and_warp_trees(tree,width,positions,old_flags,old_value,new_value,monkeypatch):
    x=torch.zeros(1,width);x[0,0]=2**24;x[0,positions]=1;before=x.clone()
    def forbidden(*a,**k):raise AssertionError('hidden native mean or sum')
    monkeypatch.setattr(torch.Tensor,'mean',forbidden);monkeypatch.setattr(torch.Tensor,'sum',forbidden)
    a=tree.tree_mean(x,vectorized=old_flags[0],descending=old_flags[1]);b=tree.tree_mean(x,vectorized=True,descending=True)
    assert torch.equal(a,torch.tensor([[old_value]])) and torch.equal(b,torch.tensor([[new_value]]))
    assert a.dtype==b.dtype==torch.float32 and torch.equal(x,before)


@pytest.mark.parametrize('vectorized,descending',[(False,False),(False,True),(True,False),(True,True)])
def test_noncontiguous_logical_rows_preserve_values_and_keepdim(tree,vectorized,descending):
    x=torch.ones(1,41,16,128).transpose(1,2)
    out=tree.tree_mean(x,vectorized=vectorized,descending=descending)
    assert out.shape==(1,16,41,1) and torch.equal(out,torch.ones(1,16,41,1))


@pytest.mark.parametrize('bad',['dtype','width','stride','negative','nan','alignment','flag','overflow'])
def test_tree_rejects_unadmitted_inputs_instead_of_precision_or_native_fallback(tree,bad):
    x=torch.ones(1,128);v=True
    if bad=='dtype':x=x.double()
    if bad=='width':x=x[:,:64]
    if bad=='stride':x=torch.ones(128,2).t()
    if bad=='negative':x[0,0]=-1
    if bad=='nan':x[0,0]=float('nan')
    if bad=='alignment':x=torch.ones(129)[1:].reshape(1,128)
    if bad=='flag':v=1
    if bad=='overflow':x.fill_(3e38)
    with pytest.raises(ValueError,match='tree|finite|dtype|width|alignment|flag|FP32'):
        tree.tree_mean(x,vectorized=v,descending=False)


def raw_fixture(tree):
    raw={};capture={}
    for p in tree.rms.PROBES:
        for n in range(4):
            name=f'{p}.n{n}';shape=(1,41,1024) if n in (0,3) else (1,16 if n==1 else 4,41,128)
            square=torch.zeros(shape);mean=torch.zeros(shape[:-1]+(1,))
            capture[name+'.square']=square;capture[name+'.unprofiled_variance']=mean
            raw[name+'.square']=square.clone();raw[name+'.native_mean']=mean.clone()
            for arm in tree.ARMS:raw[name+'.'+arm]=mean.clone()
    return raw,capture


@pytest.mark.parametrize('bad',['none','dummy','dtype','finite','baseline'])
def test_all96_roles_and_unmodified_baseline_bound_but_arm_differences_are_outcomes(tree,bad):
    raw,capture=raw_fixture(tree);name='baseline.s0.l1.n0'
    if bad=='dummy':del raw[name+'.vector_ascending'];raw['dummy']=torch.zeros(1)
    if bad=='dtype':raw[name+'.scalar_ascending']=raw[name+'.scalar_ascending'].double()
    if bad=='finite':raw[name+'.vector_descending'].fill_(float('nan'))
    if bad=='baseline':raw[name+'.native_mean']+=1
    if bad=='none':
        raw[name+'.scalar_descending']+=1
        assert tree.validate_raw(raw,capture)['tensors']==96
    else:
        with pytest.raises(ValueError,match='raw|baseline|role'):tree.validate_raw(raw,capture)


def test_summary_uses_actual_four_arm_outcomes_not_expected_source_success(tree):
    rows={}
    for p in tree.rms.PROBES:
        for n in range(4):
            rows[f'{p}.n{n}']={'width':1024 if n in (0,3) else 128,'arms':{
                arm:{'old':{'exact':False},'native':{'exact':True}} for arm in tree.ARMS}}
    result=tree.summarize(rows)
    assert result['source_old_exact']==0 and result['source_native_exact']==16
    assert result['by_arm']['scalar_ascending']=={'old_exact':0,'native_exact':16}


def test_r30_capture_identity_precedes_candidate_use(tree):
    r={'kind':'hy_isolated_mean_launch_observation','status':'diagnostic_complete','engine':'native','job_id':'315622',
        'source_revision':'071ea36442cefb0b394f676cd562eb5a52f8e8eb','trace_count':16,
        'raw_proof':{'tensors':64,'profiled_means_exact':16,'unprofiled_means_exact':16},'model_constructed':False,'checkpoint_tensors_read':0}
    assert tree.admit_capture(r) is None
    r['job_id']='315596'
    with pytest.raises(ValueError,match='capture|R30'):tree.admit_capture(r)


def test_help_and_allocation_guard_precede_history_io(tmp_path):
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM_')}
    r=subprocess.run([sys.executable,str(SCRIPT),'--help'],env=env,text=True,capture_output=True)
    assert r.returncode==0 and '--capture-root' in r.stdout
    out=tmp_path/'unused'
    r=subprocess.run([sys.executable,str(SCRIPT),'--history-root','/missing','--capture-root','/missing','--source-revision','a'*40,'--output',str(out)],env=env,text=True,capture_output=True)
    assert r.returncode!=0 and 'allocated GPU srun' in r.stderr and not out.exists()
