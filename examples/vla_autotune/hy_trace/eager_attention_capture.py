"""Instance-only observation of the recorded official eager attention method."""
from __future__ import annotations

from contextlib import contextmanager

IDENTITY_KEYS = ('prepared.images','prepared.state','prepared.tokens','prepared.token_mask',
    'prefix.embeddings','prefix.pad_mask','prefix.modality_mask','prefix.k','prefix.v',
    'suffix.state_embedding','step0.latent_before','actions')


def admit_recorded_attention(tensors, reference, *, num_steps=10):
    import torch
    trajectory = tuple(f'step{index}.{field}' for index in range(num_steps) for field in ('latent_before','time','velocity'))
    for name in (*IDENTITY_KEYS,*trajectory):
        if name not in tensors or name not in reference or tensors[name].dtype != reference[name].dtype or not torch.equal(tensors[name],reference[name]):
            raise ValueError(f'Historical reference identity/dtype differs: {name}')
    prefix = reference['prefix.k'].shape[2]
    phases = ('prefix',*(f'step{i}' for i in range(num_steps)))
    for index,phase in enumerate(phases):
        base = f'attention.{phase}.layer0.'
        q = tensors[base+'q'].transpose(1,2)
        expected = reference[f'query_norm.layer0.{index}']
        if q.dtype != expected.dtype or not torch.equal(q,expected):
            raise ValueError(f'Observed query differs from historical query-norm boundary: {phase}')
        mask_key = 'prefix.attention_mask' if index == 0 else 'suffix.attention_mask'
        if tensors[base+'mask'].dtype != reference[mask_key].dtype or not torch.equal(tensors[base+'mask'],reference[mask_key]):
            raise ValueError(f'Observed attention mask differs from historical mask: {phase}')
        for field in ('k','v'):
            observed = tensors[base+field][:,:prefix]
            expected = reference['prefix.'+field][0]
            if observed.dtype != expected.dtype or not torch.equal(observed,expected):
                raise ValueError(f'Observed attention prefix differs from historical cache: {phase}.{field}')
    return {'exact':True,'recorded_query_boundaries':len(phases),'prefix_rows':prefix,
            'scope':'Actual layer0 attention inputs; historical original-order prefix and masks'}


def _record(tensors, metadata, name, value):
    metadata[name] = {'shape':list(value.shape),'live_stride':list(value.stride()),
        'storage_offset':value.storage_offset(),'dtype':str(value.dtype)}
    tensors[name] = value.detach().cpu().contiguous().clone()
    metadata[name]['artifact_stride'] = list(tensors[name].stride())


@contextmanager
def observe_eager_attention(tower, tensors, metadata, *, num_layers=32, num_steps=10):
    import torch
    if (getattr(tower.config, 'attention_implementation', None) != 'eager'
            or type(num_layers) is not int or num_layers < 1 or type(num_steps) is not int or num_steps < 1):
        raise ValueError('Attention capture requires the recorded eager configuration and positive layer/step counts')
    missing = object()
    own_forward = tower.__dict__.get('forward',missing)
    own_eager = tower.__dict__.get('eager_attention_forward',missing)
    forward, eager = tower.forward, tower.eager_attention_forward
    state = {'complete':False,'phases':[],'num_layers':num_layers,'num_steps':num_steps}
    active = None
    def call_forward(*args, **kwargs):
        nonlocal active
        if active is not None:
            raise ValueError('Nested attention forward phase is unsupported')
        fill = kwargs.get('fill_kv_cache')
        if type(fill) is not bool:
            raise ValueError('Attention phase requires explicit fill_kv_cache')
        completed = state['phases']
        if fill:
            if completed:
                raise ValueError('Unexpected second prefix attention phase')
            label = 'prefix'
        else:
            if not completed or completed[0]['phase'] != 'prefix' or len(completed) > num_steps:
                raise ValueError('Unexpected suffix attention phase ordering/count')
            label = f'step{len(completed)-1}'
        active = {'phase':label,'attention_calls':0}
        try:
            result = forward(*args, **kwargs)
            phase = active
            completed.append(phase)
            if phase['attention_calls'] != num_layers:
                raise ValueError('Observed eager attention layer count differs from source contract')
            return result
        finally:
            active = None
    def call_eager(mask, batch_size, head_dim, q, k, v):
        if active is None:
            raise ValueError('Attention operator executed outside its observed forward phase')
        layer = active['attention_calls']
        active['attention_calls'] += 1
        if layer >= num_layers:
            raise ValueError('Unexpected extra eager attention layer')
        if (mask.dtype != torch.bool or q.dtype != torch.float32 or k.dtype != torch.float32
                or v.dtype != torch.bfloat16 or q.ndim != 4 or k.ndim != 4 or k.shape != v.shape
                or q.shape[0] != batch_size or k.shape[0] != batch_size or q.shape[-1] != head_dim
                or k.shape[-1] != head_dim or q.shape[2] % k.shape[2]
                or tuple(mask.shape) != (batch_size,q.shape[1],k.shape[1])):
            raise ValueError('Observed attention input shape/dtype/mask differs from recorded eager contract')
        name = f"attention.{active['phase']}.layer{layer}"
        if layer == 0:
            for field,value in (('q',q),('k',k),('v',v),('mask',mask)):
                _record(tensors,metadata,name+'.'+field,value)
        result = eager(mask,batch_size,head_dim,q,k,v)
        if layer == 0:
            output, probabilities = result
            if (tuple(output.shape) != (batch_size,q.shape[1],q.shape[2]*head_dim)
                    or tuple(probabilities.shape) != (batch_size,q.shape[2],q.shape[1],k.shape[1])
                    or output.dtype != torch.bfloat16 or probabilities.dtype != torch.bfloat16
                    or not torch.isfinite(output).all() or not torch.isfinite(probabilities).all()):
                raise ValueError('Observed eager attention output geometry/dtype/finite state differs')
            _record(tensors,metadata,name+'.output',output)
            _record(tensors,metadata,name+'.probabilities',probabilities)
        return result
    try:
        tower.forward, tower.eager_attention_forward = call_forward, call_eager
        yield state
        if len(state['phases']) != 1+num_steps:
            raise ValueError('Incomplete eager attention phase count')
        state['complete'] = True
    except BaseException as error:
        state['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        for name, own in (('forward',own_forward),('eager_attention_forward',own_eager)):
            if own is missing:
                tower.__dict__.pop(name,None)
            else:
                setattr(tower,name,own)
