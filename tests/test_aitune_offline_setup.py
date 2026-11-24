"""CPU checks for the explicit CUDA12 metapackage adapter and clean venv."""

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'examples/vla_autotune/external/aitune_offline.py'


def _module():
    assert SCRIPT.is_file(), 'AITune clean offline setup is missing'
    spec = importlib.util.spec_from_file_location('aitune_offline_setup', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cuda12_metapackage_is_explicitly_local_version_not_official_bytes():
    module = _module()
    original = (
        '# SPDX-License-Identifier: Apache-2.0\n'
        'setup(name="tensorrt", version="10.15.1.29",\n'
        '      install_requires=["tensorrt_cu13==10.15.1.29"])\n')
    text = module.cuda12_metapackage_text(original)
    call = ast.parse(text).body[0].value
    fields = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
    assert fields == {'name': 'tensorrt', 'version': '10.15.1.29+flashrtcu12.1',
                      'install_requires': ['tensorrt_cu12==10.15.1.29']}
    assert text.splitlines()[0] == '# SPDX-License-Identifier: Apache-2.0'
    with pytest.raises(ValueError, match='source|version'):
        module.cuda12_metapackage_text(original.replace('10.15.1.29', '11.0.0'))


def test_clean_venv_does_not_inherit_user_packages_or_overwrite(tmp_path):
    module = _module()
    destination = tmp_path / 'venv'
    module.create_clean_venv(sys.executable, destination, with_pip=False)
    output = subprocess.check_output(
        [destination / 'bin/python', '-c',
         'import json,site; print(json.dumps({"user_site":site.ENABLE_USER_SITE}))'],
        text=True)
    assert '"user_site": false' in output
    assert 'include-system-site-packages = false' in (destination / 'pyvenv.cfg').read_text()
    with pytest.raises(FileExistsError):
        module.create_clean_venv(sys.executable, destination, with_pip=False)


def test_hash_lock_constraints_allow_editable_source_without_hash_mode(tmp_path):
    module = _module()
    lock = tmp_path / 'locked.txt'
    lock.write_text('# installed and hash-checked first\n'
                    'torch==2.11.0+cu128 \\\n'
                    '    --hash=sha256:' + 'a' * 64 + '\n'
                    'tensorrt==10.15.1.29+flashrtcu12.1 --hash=sha256:' + 'b' * 64 + '\n')
    assert module.lock_constraints(lock) == (
        'torch==2.11.0+cu128\n'
        'tensorrt==10.15.1.29+flashrtcu12.1\n')
