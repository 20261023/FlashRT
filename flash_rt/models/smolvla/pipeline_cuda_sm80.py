"""Project graph ownership around unchanged upstream SmolVLA Torch operators."""

from dataclasses import dataclass
import uuid

import torch


@dataclass
class SmolVLAProfile:
    batch: dict
    noise: torch.Tensor
    output: torch.Tensor | None = None
    graph: object = None
    capture_id: str | None = None
    captured_nfe: int | None = None
    replays: int = 0


class SmolVLAPipelineCudaSm80:
    def __init__(self, policy):
        self.policy = policy
        self.nfe = 0
        original = policy.model.denoise_step
        def observed(*args, **kwargs):
            self.nfe += 1
            return original(*args, **kwargs)
        policy.model.denoise_step = observed

    def run_static(self, profile):
        self.nfe = 0
        output = self.policy.predict_action_chunk(profile.batch, noise=profile.noise)
        if self.nfe != self.policy.config.num_steps:
            raise RuntimeError(f'SmolVLA actual route NFE {self.nfe} differs from configured steps')
        return output

    def capture(self, profile):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run_static(profile)
        torch.cuda.current_stream().wait_stream(stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                output = self.run_static(profile)
            torch.cuda.current_stream().wait_stream(stream)
        except Exception:
            graph.reset()
            raise
        profile.graph, profile.output = graph, output
        profile.capture_id = uuid.uuid4().hex
        profile.captured_nfe = self.nfe

    def run(self, profile):
        if profile.graph is not None:
            profile.graph.replay()
            profile.replays += 1
            return profile.output, profile.captured_nfe
        output = self.run_static(profile)
        return output, self.nfe
