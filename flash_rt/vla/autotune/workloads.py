"""Explicit, file-backed requests; no implicit camera/robot transformations."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Request:
    request_id: str
    observation: dict
    prompt: str | None
    noise: np.ndarray


class NPZWorkload:
    def __init__(self, manifest, base_dir):
        if manifest.get('schema_version') != 1:
            raise ValueError('workload schema_version must be 1')
        self.workload_id = manifest['workload_id']
        self.model = manifest['model']
        self.specs = manifest['requests']
        if not self.workload_id or not self.specs:
            raise ValueError('workload requires an ID and at least one request')
        ids = [spec['request_id'] for spec in self.specs]
        if len(ids) != len(set(ids)):
            raise ValueError('request IDs must be unique within a workload')
        self.manifest = manifest
        self.base_dir = Path(base_dir)
        self._arrays = {}
        for spec in self.specs:
            path = Path(spec['npz_path'])
            path = path if path.is_absolute() else self.base_dir / path
            with np.load(path, allow_pickle=False) as data:
                self._arrays[spec['request_id']] = {key: data[key].copy() for key in data.files}
            if not isinstance(spec['fields'], dict):
                raise TypeError('request fields must map frontend keys to NPZ array names')
            if spec['noise_key'] not in self._arrays[spec['request_id']]:
                raise ValueError('explicit initial noise is required')

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        return cls(json.loads(path.read_text()), path.parent)

    def __len__(self):
        return len(self.specs)

    def request(self, index):
        spec = self.specs[index % len(self.specs)]
        arrays = self._arrays[spec['request_id']]
        def array(name):
            return arrays[name].copy()
        observation = {
            key: [array(name) for name in source] if isinstance(source, list) else array(source)
            for key, source in spec['fields'].items()
        }
        return Request(spec['request_id'], observation, spec.get('prompt'), array(spec['noise_key']))
