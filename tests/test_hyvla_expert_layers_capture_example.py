import importlib.util
from pathlib import Path

import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def capture():
    path=ROOT/'examples/vla_autotune/hy_trace/expert_layers_capture.py'
    spec=importlib.util.spec_from_file_location('hy_layers_capture',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Half(torch.nn.Module):
    def forward(self,x):return x*.5


class Expert(torch.nn.Module):
    def __init__(self):
        super().__init__();self.layers=torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        self.norm=torch.nn.Identity();self.failure=None
        for layer in self.layers:
            layer.input_layernorm_v=torch.nn.Identity()
            layer.post_attention_layernorm_v=torch.nn.Identity()
            layer.mlp_v=Half()
    def forward(self,value):
        for index in ([1,0] if self.failure=='order' else [0,1]):
            layer=self.layers[index]
            layer.input_layernorm_v(value)
            if self.failure=='body':raise RuntimeError('injected expert body failure')
            residual=value+(index+1)
            normalized=layer.post_attention_layernorm_v(residual)
            mlp=layer.mlp_v(normalized)
            value=residual+mlp+(1 if self.failure=='chain' else 0)
        return self.norm(value[None])


def fixture_input():return torch.tensor([[1.,2.],[-1.,0.]],dtype=torch.bfloat16)


def test_records_actual_all_layer_boundaries_without_changing_math_or_hooks(capture):
    model=Expert();value=fixture_input();saved={};meta={}
    sentinel=model.layers[0].mlp_v.register_forward_hook(lambda *a:None)
    with capture.observe_expert_layers(model,saved,meta,num_layers=2,num_steps=2,rows=2,hidden=2) as state:
        for _ in range(2):
            actual=model(value)
            torch.testing.assert_close(actual,torch.tensor([[[7.5,9.75],[3.,5.25]]],dtype=torch.bfloat16),atol=0,rtol=0)
    assert state['complete'] and state['steps']==2 and state['events']==14
    assert len(saved)==14 and state['chain']['exact']
    torch.testing.assert_close(saved['expert.step0.layer1.input'],torch.tensor([[3.,4.5],[0.,1.5]],dtype=torch.bfloat16),atol=0,rtol=0)
    assert meta['expert.step0.layer0.input']['live_stride']==list(value.stride())
    value.add_(20);assert saved['expert.step0.layer0.input'].max()<20
    assert len(model.layers[0].mlp_v._forward_hooks)==1
    sentinel.remove()
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


@pytest.mark.parametrize('failure',['order','body','count','chain','dtype','shape'])
def test_invalid_boundary_capture_never_completes_and_removes_own_hooks(capture,failure):
    model=Expert();model.failure=failure
    value=fixture_input()
    if failure=='dtype':value=value.float()
    if failure=='shape':value=value[None]
    state={}
    with pytest.raises((ValueError,RuntimeError),match='expert|Expert|boundary|chain|BF16|step'):
        with capture.observe_expert_layers(model,{}, {},num_layers=2,num_steps=2,rows=2,hidden=2) as state:
            model(value)
    assert not state.get('complete',False)
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def captured(capture):
    model=Expert();saved={}
    with capture.observe_expert_layers(model,saved,{},num_layers=2,num_steps=1,rows=2,hidden=2):model(fixture_input())
    return saved


@pytest.mark.parametrize('bad',['none','residual','mlp_output','next_input','final_input','dtype','missing','suffix'])
def test_raw_chain_admission_checks_each_residual_and_actual_next_input(capture,bad):
    saved=captured(capture);first=fixture_input()[None].clone()
    name={'residual':'expert.step0.layer0.residual','mlp_output':'expert.step0.layer1.mlp_output',
          'next_input':'expert.step0.layer1.input','final_input':'expert.step0.final_input'}.get(bad)
    if name:saved[name].add_(1)
    if bad=='dtype':saved['expert.step0.layer1.input']=saved['expert.step0.layer1.input'].float()
    if bad=='missing':del saved['expert.step0.final_input']
    if bad=='suffix':first.add_(1)
    check=lambda:capture.verify_boundary_chain(saved,num_layers=2,num_steps=1,first_suffix=first)
    if bad=='none':assert check()['layer_boundaries']==2
    else:
        with pytest.raises(ValueError,match='chain|boundary|suffix|dtype'):check()
