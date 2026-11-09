"""Revision-bound official Hy observation hooks, never a model implementation.

The diagnostic driver extracts only these two definitions. No checkpoint scan,
fixture generation, checksum pass, download or model construction happens here.
The observed interface is official source af57e7507ec5964b52fdf6296741e553cbcd3288.
"""
from contextlib import contextmanager


def validate_load_report(missing, unexpected) -> dict:
    allowed = {'model.dual_tower.expert.lm_head.weight'}
    missing, unexpected = set(missing), set(unexpected)
    required_missing = missing - allowed
    if required_missing or unexpected:
        raise RuntimeError(f'Invalid reference load: missing inference tensors={sorted(required_missing)}, '
                           f'unexpected tensors={sorted(unexpected)}')
    return {'load_missing_keys': sorted(missing), 'load_unexpected_keys': sorted(unexpected),
            'ignored_unused_keys': sorted(missing & allowed), 'missing_inference_keys': sorted(required_missing)}


@contextmanager
def diagnostic_capture(policy, tensors):
    """Observe actual calls without replacing any mathematical operation."""
    import torch
    model = policy.model
    tower = model.dual_tower
    visual = tower.vlm.model.visual
    hooks, originals, counts = [], [], {}

    def record(name, tensor):
        tensors[name] = tensor.detach().cpu().contiguous().clone()

    def observe_module(name):
        def hook(module, args, output):
            count = counts.get(name, 0)
            counts[name] = count + 1
            record(f'{name}.{count}', output)
        return hook

    def observe_input(name):
        def hook(module, args):
            count = counts.get(name, 0)
            counts[name] = count + 1
            record(f'{name}.{count}', args[0])
            record(f'{name}_strides.{count}', torch.tensor(args[0].stride(), dtype=torch.int64))
        return hook

    def replace_method(owner, name, replacement):
        original = getattr(owner, name)
        originals.append((owner, name, original))
        setattr(owner, name, replacement(original))

    def prefix_observer(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            for name, tensor in zip(('embeddings', 'pad_mask', 'block_mask', 'modality_mask'), result[:4]):
                record(f'prefix.{name}', tensor)
            return result
        return call

    def tower_observer(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('fill_kv_cache'):
                record('prefix.attention_mask', kwargs['attention_mask'])
                record('prefix.position_ids', kwargs['position_ids'])
                for layer in (0, 15, 31):
                    record(f'prefix.layer{layer}.key', result[1][layer]['key_states'])
                    record(f'prefix.layer{layer}.value', result[1][layer]['value_states'])
                    record(f'prefix.layer{layer}.hidden', result[3][layer])
            elif 'suffix.embeddings' not in tensors:
                record('suffix.embeddings', kwargs['inputs_embeds'][1])
                record('suffix.attention_mask', kwargs['attention_mask'])
                record('suffix.position_ids', kwargs['position_ids'])
            return result
        return call

    def step_observer(original):
        def call(state, pad_mask, cache, x_t, timestep):
            step = counts.get('step', 0)
            counts['step'] = step + 1
            record(f'step{step}.latent_before', x_t)
            record(f'step{step}.time', timestep)
            result = original(state, pad_mask, cache, x_t, timestep)
            record(f'step{step}.velocity', result[0])
            return result
        return call

    def position_observer(original):
        def call(*args, **kwargs):
            result = original(*args, **kwargs)
            record('vision.position_embedding', result)
            return result
        return call

    try:
        replace_method(model, 'embed_prefix', prefix_observer)
        replace_method(model, 'denoise_step', step_observer)
        replace_method(tower, 'forward', tower_observer)
        replace_method(visual.vision_tower, 'rescale_positional_embedding', position_observer)
        for name, module in (
            ('vision.patch_embed', visual.vision_tower.patch_embed),
            ('vision.block3', visual.vision_tower.blocks[3]),
            ('vision.block26', visual.vision_tower.blocks[26]),
            ('vision.projector', visual.merger), ('vision.proj1', visual.merger.proj1),
            ('vision.pool_scores', visual.merger.pooler.predictor), ('vision.pool_output', visual.merger.pooler),
            ('query_norm.layer0', tower.vlm.language_model.model.layers[0].self_attn.query_layernorm),
        ):
            hooks.append(module.register_forward_hook(observe_module(name)))
        for name, module in (('vision.proj1_input', visual.merger.proj1),
                             ('vision.pool_input', visual.merger.pooler),
                             ('vision.pool_predictor_input', visual.merger.pooler.predictor)):
            hooks.append(module.register_forward_pre_hook(observe_input(name)))
        record('language.rope_inv_freq', tower.vlm.language_model.model.rotary_emb.inv_freq)
        record('vision.proj1_weight', visual.merger.proj1.weight)
        record('vision.proj1_bias', visual.merger.proj1.bias)
        yield
    finally:
        for hook in hooks:
            hook.remove()
        for owner, name, original in reversed(originals):
            setattr(owner, name, original)
