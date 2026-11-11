"""Observe actual Expert hidden/residual boundaries, without replacing math."""
from __future__ import annotations

from contextlib import contextmanager


def verify_boundary_chain(tensors,*,num_layers=32,num_steps=10,first_suffix=None):
    import torch
    if any(type(n) is not int or n<1 for n in (num_layers,num_steps)):
        raise ValueError('Expert boundary chain requires positive layer/step counts')
    try:
        shape=tensors['expert.step0.layer0.input'].shape
        if len(shape)!=2 or min(shape)<1:raise ValueError('Expert boundary chain expects 2D hidden inputs')
        for step in range(num_steps):
            final=tensors[f'expert.step{step}.final_input']
            if final.shape!=(1,*shape) or final.dtype!=torch.bfloat16 or not torch.isfinite(final).all():
                raise ValueError('Expert final boundary shape/dtype/finite contract differs')
            for layer in range(num_layers):
                base=f'expert.step{step}.layer{layer}.'
                values=[tensors[base+field] for field in ('input','residual','mlp_output')]
                if any(v.shape!=shape or v.dtype!=torch.bfloat16 or not torch.isfinite(v).all() for v in values):
                    raise ValueError('Expert boundary shape/dtype/finite contract differs')
                expected=tensors[f'expert.step{step}.layer{layer+1}.input'] if layer+1<num_layers else final[0]
                if expected.dtype!=torch.bfloat16 or not torch.equal(values[1]+values[2],expected):
                    raise ValueError(f'Expert residual chain does not reproduce actual next input: step{step}.layer{layer}')
        if first_suffix is not None and (first_suffix.dtype!=torch.bfloat16 or first_suffix.shape!=(1,*shape)
                or not torch.equal(tensors['expert.step0.layer0.input'][None],first_suffix)):
            raise ValueError('Expert first boundary differs from recorded suffix entry')
    except KeyError as error:
        raise ValueError(f'Missing Expert boundary chain tensor: {error}') from error
    return {'exact':True,'layer_boundaries':num_layers*num_steps,'first_suffix_checked':first_suffix is not None}


@contextmanager
def observe_expert_layers(model,tensors,metadata,*,num_layers=32,num_steps=10,rows=41,hidden=1024):
    import torch
    if (any(type(n) is not int or n<1 for n in (num_layers,num_steps,rows,hidden))
            or len(model.layers)!=num_layers):
        raise ValueError('Expert boundary layer/step/geometry contract differs')
    state={'complete':False,'steps':0,'events':0,'num_layers':num_layers,'num_steps':num_steps,
           'rows':rows,'hidden':hidden}
    handles=[]
    fields=('input','residual','mlp_output')
    def record(field,layer,value):
        position=state['events']%(3*num_layers+1)
        final=position==3*num_layers
        expected_field='final_input' if final else fields[position%3]
        expected_layer=None if final else position//3
        shape=(1,rows,hidden) if final else (rows,hidden)
        if (field!=expected_field or layer!=expected_layer or state['steps']>=num_steps
                or value.shape!=shape or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all()):
            raise ValueError('Expert boundary event order or BF16 suffix geometry differs')
        name=f"expert.step{state['steps']}."+(field if final else f'layer{layer}.{field}')
        if name in tensors:raise ValueError('Duplicate Expert boundary record')
        metadata[name]={'shape':list(value.shape),'live_stride':list(value.stride()),
                        'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}
        tensors[name]=value.detach().cpu().contiguous().clone()
        state['events']+=1
        if final:state['steps']+=1
    try:
        for index,layer in enumerate(model.layers):
            handles.append(layer.input_layernorm_v.register_forward_pre_hook(lambda m,a,i=index:record('input',i,a[0])))
            handles.append(layer.post_attention_layernorm_v.register_forward_pre_hook(lambda m,a,i=index:record('residual',i,a[0])))
            handles.append(layer.mlp_v.register_forward_hook(lambda m,a,o,i=index:record('mlp_output',i,o)))
        handles.append(model.norm.register_forward_pre_hook(lambda m,a:record('final_input',None,a[0])))
        yield state
        if state['steps']!=num_steps or state['events']!=(3*num_layers+1)*num_steps:
            raise ValueError('Incomplete Expert boundary step count')
        state['chain']=verify_boundary_chain(tensors,num_layers=num_layers,num_steps=num_steps)
        state['complete']=True
    finally:
        for handle in reversed(handles):handle.remove()
