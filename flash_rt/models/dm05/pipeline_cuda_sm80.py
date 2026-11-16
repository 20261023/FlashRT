"""Project-owned DM0.5 BF16 vision/prefix/raw-velocity/Euler pipeline, SM80.

Equations follow dexmal/opendm@03f799120d425ead51d9482bb1da950b6dabade9
(Apache-2.0), opendm/model/dm05/dm05_arch.py. This is a new inference schedule,
not a call to the author's inference_action or to ApxInf. In particular it
matches the measured author's HOT BF16 suffix graph, not its FP32 eager I/O.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .attention_cuda_sm80 import Dm05AttentionSm80
from .weights import ACTION, PREFIX, VISION, load_weights


def rms(x, weight):
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
    return (value * (1.0 + weight.float())).to(x.dtype)


def adaptive_rms(x, modulation):
    scale, shift, gate = modulation.chunk(3, dim=-1)
    value = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    return (value * (1.0 + scale.float()) + shift.float()).to(x.dtype), gate


def rotate(x, cos, sin):
    half = torch.cat((-x[..., 128:], x[..., :128]), dim=-1)
    # Keep all three BF16 rounding points of the author's uncompiled RoPE.
    return x * cos + half * sin


class Dm05PipelineSm80:
    def __init__(self, checkpoint, *, native):
        if tuple(torch.cuda.get_device_capability()) != (8, 0):
            raise ValueError('DM05 cuda_sm80 requires compute capability 8.0')
        self.native = native
        self.weights, self.identity = load_weights(checkpoint)
        self.buffers = {}
        self.zero_bias = {}
        self.native_gelu_enqueues = 0
        if native:
            from flash_rt import flash_rt_gemm_plan, flash_rt_kernels
            self.runner = flash_rt_gemm_plan.CachedGemmRunner()
            self.fvk = flash_rt_kernels
        # The Gemma multimodal projector stores K,N instead of nn.Linear N,K.
        projector = self.weights['model.vlm.model.multi_modal_projector.mm_input_projection_weight']
        self.weights['_projector.weight'] = projector.T.contiguous()
        self.embed_scale = torch.tensor(math.sqrt(2560), device='cuda', dtype=torch.bfloat16)
        self.inv_freq = []
        for theta, factor in ((10000.0, 1.0), (1000000.0, 8.0)):
            self.inv_freq.append(1.0 / theta ** (
                torch.arange(0, 256, 2, dtype=torch.float32, device='cuda') / 256) / factor)
        self.modulations = self._time_modulations()

    def linear(self, x, name):
        weight = self.weights[f'{name}.weight']
        bias = self.weights.get(f'{name}.bias')
        shape = (*x.shape[:-1], weight.shape[0])
        activation = x.contiguous().reshape(-1, weight.shape[1])
        key = (name, activation.shape[0])
        if key not in self.buffers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'DM05 linear {name} was not prepared for {key[1]} rows')
            self.buffers[key] = torch.empty((activation.shape[0], weight.shape[0]),
                                           dtype=torch.bfloat16, device='cuda')
        out = self.buffers[key]
        if self.native:
            if bias is None:
                if weight.shape[0] not in self.zero_bias:
                    if torch.cuda.is_current_stream_capturing():
                        raise RuntimeError('DM05 zero bias must be prepared before capture')
                    self.zero_bias[weight.shape[0]] = torch.zeros(
                        weight.shape[0], device='cuda', dtype=torch.bfloat16)
                bias = self.zero_bias[weight.shape[0]]
            stream = torch.cuda.current_stream()
            self.runner.bf16_nt_bias(
                activation.data_ptr(), weight.data_ptr(), out.data_ptr(), bias.data_ptr(),
                activation.shape[0], weight.shape[0], weight.shape[1], stream.cuda_stream)
            for tensor in (activation, weight, bias, out):
                tensor.record_stream(stream)
        elif bias is None:
            torch.mm(activation, weight.T, out=out)
        else:
            torch.addmm(bias, activation, weight.T, out=out)
        return out.view(shape)

    def gelu(self, x):
        if not self.native:
            return F.gelu(x, approximate='tanh')
        width = x.shape[-1]
        if width not in self.zero_bias:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('DM05 GELU zero bias must be prepared before capture')
            self.zero_bias[width] = torch.zeros(width, dtype=torch.bfloat16, device='cuda')
        self.fvk.bias_gelu_bf16_strict(
            x.data_ptr(), self.zero_bias[width].data_ptr(), x.numel() // width, width,
            torch.cuda.current_stream().cuda_stream)
        self.native_gelu_enqueues += 1
        return x

    def layer_norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.weights[f'{name}.weight'],
                            self.weights[f'{name}.bias'], 1e-6)

    def mlp(self, x, base):
        gate = self.gelu(self.linear(x, f'{base}.gate_proj'))
        up = self.linear(x, f'{base}.up_proj')
        return self.linear(gate * up, f'{base}.down_proj')

    def _time_modulations(self):
        fraction = torch.linspace(0, 1, 512, dtype=torch.float32, device='cuda')
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        frequency = 1.0 / period * 2 * math.pi
        result = []
        time_value = 1.0
        for _ in range(10):
            time = torch.full((1, 1), time_value, dtype=torch.bfloat16, device='cuda')
            angle = frequency[None, :] * time.float()
            embedding = torch.cat((angle.sin(), angle.cos()), dim=-1).to(torch.bfloat16)
            cond = F.silu(self.linear(embedding, 'model.time_mlp_in'))
            cond = F.silu(self.linear(cond, 'model.time_mlp_out'))
            # Linear output buffers are reused across times; retain copies.
            inputs = tuple(self.linear(cond, f'{ACTION}.input_time_modulators.{i}').clone()[:, None]
                           for i in range(34))
            mlps = tuple(self.linear(cond, f'{ACTION}.mlp_time_modulators.{i}').clone()[:, None]
                         for i in range(34))
            final = self.linear(cond, f'{ACTION}.final_time_modulator').clone()[:, None]
            result.append((inputs, mlps, final))
            time_value += -0.1
        return result

    def rope(self, positions):
        result = []
        for inv in self.inv_freq:
            angle = (inv[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
            angle = torch.cat((angle, angle), dim=-1)
            result.append((angle.cos().to(torch.bfloat16)[:, :, None, :],
                           angle.sin().to(torch.bfloat16)[:, :, None, :]))
        return result

    def qkv(self, x, base, rotary):
        q = self.linear(x, f'{base}.q_proj').view(1, x.shape[1], 8, 256)
        k = self.linear(x, f'{base}.k_proj').view(1, x.shape[1], 4, 256)
        v = self.linear(x, f'{base}.v_proj').view(1, x.shape[1], 4, 256)
        q = rms(q, self.weights[f'{base}.q_norm.weight'])
        k = rms(k, self.weights[f'{base}.k_norm.weight'])
        return rotate(q, *rotary), rotate(k, *rotary), v

    def vision(self, pixels, attention):
        patch = f'{VISION}.embeddings.patch_embedding'
        hidden = F.conv2d(pixels, self.weights[f'{patch}.weight'],
                          self.weights[f'{patch}.bias'], stride=14)
        hidden = hidden.flatten(2).transpose(1, 2).contiguous()
        hidden = hidden + self.weights[f'{VISION}.embeddings.position_embedding.weight']
        for i in range(27):
            base = f'{VISION}.encoder.layers.{i}'
            x = self.layer_norm(hidden, f'{base}.layer_norm1')
            q, k, v = (self.linear(x, f'{base}.self_attn.{name}_proj').view(3, 1024, 16, 72)
                       for name in ('q', 'k', 'v'))
            attended = attention.attend('vision', i, q, k, v).reshape(3, 1024, 1152)
            hidden = hidden + self.linear(attended, f'{base}.self_attn.out_proj')
            x = self.layer_norm(hidden, f'{base}.layer_norm2')
            x = self.gelu(self.linear(x, f'{base}.mlp.fc1'))
            hidden = hidden + self.linear(x, f'{base}.mlp.fc2')
        hidden = self.layer_norm(hidden, f'{VISION}.post_layernorm')
        grid = hidden.transpose(1, 2).reshape(3, 1152, 32, 32)
        pooled = F.avg_pool2d(grid, 2, stride=2).flatten(2).transpose(1, 2).contiguous()
        pooled = rms(pooled, self.weights[
            'model.vlm.model.multi_modal_projector.mm_soft_emb_norm.weight'])
        return self.linear(pooled, '_projector').reshape(768, 2560)

    def prefill(self, profile):
        features = self.vision(profile.pixels, profile.attention)
        hidden = F.embedding(profile.ids, self.weights[f'{PREFIX}.embed_tokens.weight']) * self.embed_scale
        hidden = hidden.reshape(-1, 2560).index_copy(0, profile.image_rows, features)[None]
        for i in range(34):
            base = f'{PREFIX}.layers.{i}'
            x = rms(hidden, self.weights[f'{base}.input_layernorm.weight'])
            q, k, v = self.qkv(x, f'{base}.self_attn', profile.prefix_rope[(i + 1) % 6 == 0])
            profile.keys[i][:, :profile.length].copy_(k)
            profile.values[i][:, :profile.length].copy_(v)
            x = profile.attention.attend('prefix', i, q, k, v).reshape(1, profile.length, 2048)
            x = self.linear(x, f'{base}.self_attn.o_proj')
            hidden = hidden + rms(x, self.weights[f'{base}.post_attention_layernorm.weight'])
            x = rms(hidden, self.weights[f'{base}.pre_feedforward_layernorm.weight'])
            hidden = hidden + rms(self.mlp(x, f'{base}.mlp'),
                                  self.weights[f'{base}.post_feedforward_layernorm.weight'])

    def velocity(self, state, profile, step):
        hidden = self.linear(state, 'model.action_in_proj')
        inputs, mlps, final = self.modulations[step]
        for i in range(34):
            base = f'{ACTION}.layers.{i}'
            x, attn_gate = adaptive_rms(hidden, inputs[i])
            q, k, v = self.qkv(x, f'{base}.self_attn', profile.suffix_rope[(i + 1) % 6 == 0])
            profile.keys[i][:, profile.length:].copy_(k)
            profile.values[i][:, profile.length:].copy_(v)
            x = profile.attention.attend('suffix', i, q, profile.keys[i], profile.values[i]).reshape(1, 50, 2048)
            x = self.linear(x, f'{base}.self_attn.o_proj')
            x = rms(x, self.weights[f'{base}.post_attention_layernorm.weight'])
            hidden = hidden + x * attn_gate
            x, mlp_gate = adaptive_rms(hidden, mlps[i])
            x = rms(self.mlp(x, f'{base}.mlp'), self.weights[f'{base}.post_feedforward_layernorm.weight'])
            hidden = hidden + x * mlp_gate
        hidden, _ = adaptive_rms(hidden, final)
        return self.linear(hidden, 'model.action_out_proj')

    def forward(self, profile):
        self.prefill(profile)
        state = profile.noise
        for step in range(10):
            state = state * profile.action_mask
            state = state + self.velocity(state, profile, step) * -0.1
        profile.output.copy_(state)


class Dm05ProfileSm80:
    """One bounded exact-prefix graph and its address-stable request inputs."""

    def __init__(self, pipeline, length):
        self.length = length
        self.graph = None
        self.replays = 0
        self.ids = torch.zeros((1, length), dtype=torch.long, device='cuda')
        self.image_rows = torch.zeros(768, dtype=torch.long, device='cuda')
        self.pixels = torch.zeros((3, 3, 448, 448), dtype=torch.bfloat16, device='cuda')
        self.noise = torch.zeros((1, 50, 32), dtype=torch.bfloat16, device='cuda')
        self.output = torch.empty_like(self.noise)
        self.action_mask = torch.zeros((1, 1, 32), dtype=torch.bfloat16, device='cuda')
        self.action_mask[..., :14] = 1
        self.keys = [torch.empty(1, length + 50, 4, 256, dtype=torch.bfloat16, device='cuda')
                     for _ in range(34)]
        self.values = [torch.empty_like(key) for key in self.keys]
        self.attention = Dm05AttentionSm80(length, native=pipeline.native)
        self.attention.prefix_mask = torch.ones(1, 1, length, length, dtype=torch.bool, device='cuda')
        self.prefix_rope = pipeline.rope(torch.arange(length, device='cuda')[None])
        self.suffix_rope = pipeline.rope(torch.arange(length, length + 50, device='cuda')[None])

    def capture(self, pipeline):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                pipeline.forward(self)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            pipeline.forward(self)
        self.graph = graph

    def run(self, pipeline):
        if self.graph is None:
            pipeline.forward(self)
        else:
            self.graph.replay()
            self.replays += 1
