"""Admit pinned read-only LeRobot sources; importing this module is CPU-only."""

import hashlib
from pathlib import Path
import subprocess


REVISION = 'b8ad81bf397d59dda69ccfc7e74e847f0a9d4fbf'
SOURCE_URL = 'https://github.com/huggingface/lerobot'
POLICY_REVISION = '6721902bc4d61e50a3bfdb11dfb4cb626f05d102'
POLICY_SHA256 = '71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f'
SOURCE_HASHES = {
    'configuration_smolvla.py': '2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0',
    'modeling_smolvla.py': '5daaa297954acf9ae89397b571322f07bf30798e72f566e61a09393342cb1f99',
    'processor_smolvla.py': 'd34594291b2ecf7801f8c27a940a998c52a7e308cbc89f2403c9cbecfb368f22',
    'smolvlm_with_expert.py': 'f7542fa2bf904f9ef64d26843809f913a5e93c4dfa538dda06db0b288391ab4d',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def admit_source(source):
    root = Path(source).resolve(strict=True)
    marker = root / 'AUTHOR_GIT_COMMIT'
    if marker.is_file():
        revision = marker.read_text().strip()
    else:
        process = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                 text=True, capture_output=True, check=True)
        revision = process.stdout.strip()
        status = subprocess.check_output(
            ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'], text=True)
        if status.strip():
            raise ValueError('SmolVLA requires a clean pinned author source; tracked changes found')
    if revision != REVISION:
        raise ValueError(f'SmolVLA requires author revision {REVISION}')
    package = root / 'src/lerobot/policies/smolvla'
    for filename, expected in SOURCE_HASHES.items():
        if sha256(package / filename) != expected:
            raise ValueError(f'SmolVLA pinned author bytes differ: {filename}')
    return root
