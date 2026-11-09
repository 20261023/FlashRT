"""Observation-oriented backend boundary, independent of robot transport."""

from __future__ import annotations

import inspect
from threading import RLock

import numpy as np

from .config import BackendConfig


class VLABackend:
    """Own one stateful frontend; do not share its buffers across parallel calls.

    Input is a frontend observation, not a universal robot schema. In particular,
    this class does not normalize state, resize camera histories or reinterpret
    normalized/padded actions as physical robot commands.
    """

    def __init__(self, config: BackendConfig, frontend, *, hardware: str,
                 precision: str, action_domain: str, action_decoder=None):
        self.config = config
        self.frontend = frontend
        self.hardware = hardware
        self.precision = precision
        self.action_domain = action_domain
        self.action_decoder = action_decoder
        self._lock = RLock()
        self._prompt = None
        self._prompt_state = None
        self._closed = False
        self._retired_pi_pipelines = {}
        setter = getattr(frontend, 'set_prompt', None)
        self._has_prompt_setter = callable(setter)
        self._prompt_accepts_state = (
            self._has_prompt_setter and 'state' in inspect.signature(setter).parameters)
        self._infer_accepts_noise = (
            'noise' in inspect.signature(frontend.infer).parameters)

    def _prepare(self, observation, prompt):
        if self._closed:
            raise RuntimeError('Backend is closed; load a new backend')
        if not isinstance(observation, dict):
            raise TypeError('observation must be a frontend observation dict')
        has_context = self.config.model == 'fastwam' and observation.get('context') is not None
        if has_context:
            if prompt is not None or observation.get('prompt') not in (None, ''):
                raise ValueError('FastWAM prompt and prepared context are mutually exclusive')
            # FastWAM can consume caller-prepared UMT5 embeddings without a
            # tokenizer/text encoder. An empty label must not trigger encoding.
            selected = ''
        else:
            selected = prompt if prompt is not None else observation.get('prompt')
            if selected is None:
                selected = self._prompt
        if selected is None:
            raise ValueError('prompt is required on the first request')
        if not isinstance(selected, str):
            raise TypeError('prompt must be text; token-id inputs use the direct frontend')
        state = observation.get('state')
        # Only Pi0.5 tokenizes state into the prompt. Hy-VLA accepts a state
        # kwarg for API compatibility but processes it in the action tower.
        state_changed = False
        if self.config.model == 'pi05' and self._prompt_accepts_state:
            state_changed = not self._same_state(self._prompt_state, state)
        if selected != self._prompt or state_changed:
            kwargs = {'state': state} if self._prompt_accepts_state else {}
            previous = getattr(self.frontend, 'pipeline', None)
            setter_failed = True
            try:
                if self._has_prompt_setter:
                    self.frontend.set_prompt(selected, **kwargs)
                setter_failed = False
            finally:
                if (self.config.model == 'pi0' and previous is not None
                        and previous is not getattr(self.frontend, 'pipeline', None)):
                    # The setter may replace the pipeline before an embedding
                    # upload fails. Keep ownership even if cleanup also fails.
                    self._retired_pi_pipelines[id(previous)] = previous
                    if setter_failed:
                        try:
                            self._release_retired_pi_pipelines()
                        except Exception:
                            # Preserve the original setter exception. close()
                            # retries the retained, possibly partly freed set.
                            pass
                    else:
                        self._release_retired_pi_pipelines()
            self._prompt = selected
            if self.config.model == 'pi05' and self._prompt_accepts_state:
                self._prompt_state = None if state is None else np.asarray(state).copy()
        if hasattr(self.frontend, 'calibrate_with_real_data') and not getattr(
                self.frontend, 'calibrated', False):
            self.frontend.calibrate_with_real_data([observation])

    @staticmethod
    def _same_state(left, right):
        if left is None or right is None:
            return left is right
        return np.array_equal(np.asarray(left), np.asarray(right))

    def _release_retired_pi_pipelines(self):
        if self._retired_pi_pipelines:
            from .resources import release_pi_pipeline_graphs
            release_pi_pipeline_graphs(self._retired_pi_pipelines.values())
            self._retired_pi_pipelines.clear()

    def prepare(self, observation: dict, *, prompt: str | None = None):
        """Prepare prompt/calibration/graphs using a representative observation.

        This can execute model warmup. It is an explicit runtime operation,
        unlike :func:`describe_backend`, which only describes a candidate.
        """
        with self._lock:
            self._prepare(observation, prompt)
            if not hasattr(self.frontend, 'calibrate_with_real_data'):
                # Hy-VLA/FastWAM build their execution state on the first
                # request rather than through Pi's calibration entry point.
                # An explicit prepare must actually warm that path; internal
                # preparation in infer must NOT generate an extra request.
                self._invoke(observation)
        return self

    def _invoke(self, observation, noise=None):
        obs = dict(observation)
        # Some frontends also read prompt from the observation. Never let its
        # old label override the effective prompt chosen at this boundary.
        if self.config.model == 'fastwam' and obs.get('context') is not None:
            obs.pop('prompt', None)
        else:
            obs['prompt'] = self._prompt
        supplied_noise = noise if noise is not None else obs.pop('noise', None)
        if self._infer_accepts_noise:
            kwargs = {} if supplied_noise is None else {'noise': supplied_noise}
            return self.frontend.infer(obs, **kwargs)
        if supplied_noise is not None:
            obs['noise'] = supplied_noise
        return self.frontend.infer(obs)

    def infer(self, observation: dict, *, prompt: str | None = None, noise=None):
        """Run one complete request, retaining model and sampler metadata."""
        with self._lock:
            if not isinstance(observation, dict):
                raise TypeError('observation must be a frontend observation dict')
            if noise is not None and 'noise' in observation:
                raise ValueError('Supply noise either in observation or as a keyword, not both')
            self._prepare(observation, prompt)
            result = self._invoke(observation, noise)
            if not isinstance(result, dict) or 'actions' not in result:
                raise RuntimeError('Frontend did not return an actions dictionary')
            result = dict(result)
            if self.action_decoder is not None:
                if 'raw_actions' not in result:
                    raise RuntimeError('Checkpoint action processor requires normalized raw_actions')
                result['actions'] = self.action_decoder(result['raw_actions'])
            if self.config.sampler == 'probeflow' and 'sampler_metadata' not in result:
                raise RuntimeError('Requested ProbeFlow but frontend returned no sampler metadata')
            if self.config.sampler == 'euler' and 'sampler_metadata' not in result:
                steps = getattr(self.frontend, '_num_steps',
                                getattr(self.frontend, 'num_steps', None))
                if steps is not None:
                    result['sampler_metadata'] = {
                        'sampler': 'euler', 'scheduled_steps': int(steps),
                        'nfe': int(steps),
                    }
            metadata = result.get('sampler_metadata')
            if metadata is not None and (
                    not isinstance(metadata, dict)
                    or metadata.get('sampler') != self.config.sampler):
                raise RuntimeError('Frontend sampler metadata does not match the requested policy')
            result['backend_metadata'] = {
                'model': self.config.model, 'backend': self.config.backend,
                'sampler': self.config.sampler, 'hardware': self.hardware,
                'precision': self.precision, 'framework': self.config.framework,
                'action_domain': self.action_domain,
            }
            if self.action_decoder is not None:
                result['backend_metadata']['action_processor'] = {
                    'mode': self.action_decoder.mode, 'dimension': self.action_decoder.dimension,
                    'source': self.action_decoder.source}
            coverage = result.get('execution_metadata',
                                  getattr(self.frontend, 'execution_coverage', None))
            if coverage is not None:
                result['backend_metadata']['execution_coverage'] = coverage
            return result

    def close(self):
        """Release owned native graphs/frontend; never flush global caches."""
        with self._lock:
            if not self._closed:
                self._release_retired_pi_pipelines()
                close = getattr(self.frontend, 'close', None)
                if callable(close):
                    close()
                if self.config.model in ('pi0', 'pi05'):
                    from .resources import release_pi_frontend_graphs
                    release_pi_frontend_graphs(self.frontend)
                self.frontend = None
                self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError('Backend is closed')
        return self

    def __exit__(self, *exc):
        self.close()
