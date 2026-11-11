import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def capture():
    path=ROOT/'examples/vla_autotune/hy_trace/eager_attention_capture.py'
    if not path.exists():pytest.fail('eager attention capture is missing')
    spec=importlib.util.spec_from_file_location('hy_eager_capture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Tower:
    def __init__(self,layers=2):
        self.layers=layers;self.config=SimpleNamespace(attention_implementation='eager');self.fail=False
    def eager_attention_forward(self,mask,batch,head_dim,q,k,v):
        # Real tiny eager mathematics; the observer must not alter any result.
        groups=q.shape[2]//v.shape[2]
        key=k.repeat_interleave(groups,dim=2).transpose(1,2)
        value=v.repeat_interleave(groups,dim=2).transpose(1,2)
        scores=(q.transpose(1,2)@key.transpose(-2,-1))*(head_dim**-.5)
        probabilities=torch.softmax(torch.where(mask[:,None],scores,-2.3819763e38),dim=-1).to(v.dtype)
        output=(probabilities@value).transpose(1,2).reshape(batch,q.shape[1],-1)
        return output,probabilities
    def forward(self,*,fill_kv_cache,inputs):
        result=None
        for _ in range(self.layers):
            result=self.eager_attention_forward(*inputs)
            if self.fail:raise RuntimeError('injected forward failure')
        return result


def inputs(prefix_rows=0):
    # Q is noncontiguous and not zero; metadata must describe the live view.
    q=torch.arange(8.).reshape(1,2,2,2).transpose(1,2)
    length=2+prefix_rows
    k=torch.full((1,length,1,2),2.);v=torch.arange(length*2,dtype=torch.bfloat16).reshape(1,length,1,2)
    return (torch.ones(1,2,length,dtype=torch.bool),1,2,q,k,v)


def test_capture_observes_phase_layer0_only_preserves_math_and_restores_class_bindings(capture):
    tower=Tower();args=inputs();saved={};meta={}
    calls=[(True,args),(False,inputs(2)),(False,inputs(2))]
    expected=[tower.forward(fill_kv_cache=fill,inputs=values) for fill,values in calls]
    with capture.observe_eager_attention(tower,saved,meta,num_layers=2,num_steps=2) as state:
        for index,(fill,values) in enumerate(calls):
            actual=tower.forward(fill_kv_cache=fill,inputs=values)
            torch.testing.assert_close(actual[0],expected[index][0],atol=0,rtol=0)
    assert state['complete'] is True and [x['attention_calls'] for x in state['phases']]==[2,2,2]
    assert set(saved)=={f'attention.{phase}.layer0.{name}' for phase in ('prefix','step0','step1')
        for name in ('q','k','v','mask','output','probabilities')}
    assert 'forward' not in tower.__dict__ and 'eager_attention_forward' not in tower.__dict__
    info=meta['attention.step0.layer0.q']
    assert info['live_stride']==list(args[3].stride()) and info['storage_offset']==args[3].storage_offset()
    args[3].add_(99)
    assert saved['attention.step0.layer0.q'].max()<99


@pytest.mark.parametrize('failure',['body','layer_count','phase_count','mode'])
def test_capture_failure_never_marks_complete_and_restores_both_bindings(capture,failure):
    tower=Tower(layers=1 if failure=='layer_count' else 2);saved={};meta={};state={}
    if failure=='body':tower.fail=True
    if failure=='mode':tower.config.attention_implementation='fa2'
    with pytest.raises((RuntimeError,ValueError),match='forward|layer|phase|eager'):
        with capture.observe_eager_attention(tower,saved,meta,num_layers=2,num_steps=2) as state:
            tower.forward(fill_kv_cache=True,inputs=inputs())
    assert not state.get('complete',False)
    assert 'forward' not in tower.__dict__ and 'eager_attention_forward' not in tower.__dict__


def test_existing_instance_bindings_are_restored_even_when_input_recording_fails(capture):
    tower=Tower();original=tower.eager_attention_forward;tower.eager_attention_forward=original
    saved={};meta={};args=list(inputs());args[0]=torch.ones(1,2,2)  # mask must remain bool
    with pytest.raises(ValueError,match='mask|dtype|attention'):
        with capture.observe_eager_attention(tower,saved,meta,num_layers=2,num_steps=1):
            tower.forward(fill_kv_cache=True,inputs=tuple(args))
    assert tower.eager_attention_forward is original and 'forward' not in tower.__dict__


def reference_pair(capture):
    tower=Tower();saved={};meta={}
    with capture.observe_eager_attention(tower,saved,meta,num_layers=2,num_steps=2):
        for fill in (True,False,False):tower.forward(fill_kv_cache=fill,inputs=inputs(0 if fill else 2))
    historical={name:torch.zeros(1) for name in capture.IDENTITY_KEYS}
    historical['prefix.k']=saved['attention.prefix.layer0.k'][None].clone()
    historical['prefix.v']=saved['attention.prefix.layer0.v'][None].clone()
    historical['prefix.attention_mask']=saved['attention.prefix.layer0.mask'].clone()
    historical['suffix.attention_mask']=saved['attention.step0.layer0.mask'].clone()
    for i,phase in enumerate(('prefix','step0','step1')):
        historical[f'query_norm.layer0.{i}']=saved[f'attention.{phase}.layer0.q'].transpose(1,2).clone()
    saved.update({name:historical[name].clone() for name in capture.IDENTITY_KEYS})
    for step in range(2):
        for field in ('latent_before','time','velocity'):
            key=f'step{step}.{field}'
            historical.setdefault(key,torch.tensor([float(step)]))
            saved[key]=historical[key].clone()
    return saved,historical


@pytest.mark.parametrize('bad',['none','actions','query','prefix','mask','dtype','latent','time','velocity'])
def test_recorded_attention_admission_checks_actual_history_not_only_case_labels(capture,bad):
    saved,history=reference_pair(capture)
    if bad=='actions':saved['actions'].add_(1)
    if bad=='query':saved['attention.step0.layer0.q'].add_(1)
    if bad=='prefix':saved['attention.step1.layer0.k'].add_(1)
    if bad=='mask':saved['attention.step1.layer0.mask'].logical_not_()
    if bad=='dtype':saved['prepared.state']=saved['prepared.state'].double()
    if bad in ('latent','time','velocity'):
        key={'latent':'latent_before','time':'time','velocity':'velocity'}[bad]
        saved[f'step1.{key}'].add_(.1)
    call=lambda:capture.admit_recorded_attention(saved,history,num_steps=2)
    if bad=='none':assert call()['exact'] is True
    else:
        with pytest.raises(ValueError,match='history|reference|query|prefix|mask|dtype'):call()
