"""First execution backend for model-level VLA optimization.

``describe_backend`` is CPU-only. ``load_backend`` constructs a real frontend;
``prepare``/``infer`` execute it. Runtime verification is a separate operation.
"""

from .config import BackendConfig
from .backend import VLABackend
from .registry import describe_backend, load_backend

__all__ = ['BackendConfig', 'VLABackend', 'describe_backend', 'load_backend']
