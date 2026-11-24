"""CPU regression coverage for external setup safety and reproducibility."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'examples/vla_autotune/external/setup.py'


def setup_module():
    assert SCRIPT.is_file(), 'isolated external-tool setup is not implemented'


def _setup():
    spec = importlib.util.spec_from_file_location('external_tool_setup', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], text=True, check=True,
                          capture_output=True).stdout.strip()


def _bundle(tmp_path):
    repo = tmp_path / 'source'
    repo.mkdir()
    _git(repo, 'init', '-q')
    (repo / 'one.py').write_text('VALUE = 1\n')
    (repo / 'two.py').write_text('VALUE = 2\n')
    _git(repo, 'add', '.')
    _git(repo, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
         'commit', '-qm', 'fixture')
    pin = _git(repo, 'rev-parse', 'HEAD')
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    files = {}
    for name, value in [('one.py', 10), ('two.py', 20)]:
        path = repo / name
        before = path.read_bytes()
        after = f'VALUE = {value}\n'.encode()
        files[name] = {'before': hashlib.sha256(before).hexdigest(),
                       'after': hashlib.sha256(after).hexdigest()}
        path.write_bytes(after)
    patch = subprocess.run(['git', '-C', str(repo), 'diff', '--binary'],
                           check=True, capture_output=True).stdout
    (bundle / 'enable.patch').write_bytes(patch)
    for name, value in [('one.py', 1), ('two.py', 2)]:
        (repo / name).write_text(f'VALUE = {value}\n')
    manifest = {'commit': pin, 'files': files,
                'patches': [{'path': 'enable.patch',
                             'sha256': hashlib.sha256(patch).hexdigest()}]}
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    return repo, bundle


def test_patch_bundle_checks_every_preimage_before_writing(tmp_path):
    setup = _setup()
    repo, bundle = _bundle(tmp_path)
    (repo / 'two.py').write_text('USER_WORK = True\n')
    with pytest.raises(ValueError, match='drift|modified|preimage'):
        setup.apply_patch_bundle(repo, bundle)
    assert (repo / 'one.py').read_text() == 'VALUE = 1\n'
    assert (repo / 'two.py').read_text() == 'USER_WORK = True\n'


def test_patch_bundle_is_idempotent_but_rejects_subsequent_edits(tmp_path):
    setup = _setup()
    repo, bundle = _bundle(tmp_path)
    result = setup.apply_patch_bundle(repo, bundle)
    assert result['state'] == 'applied'
    assert (repo / 'one.py').read_text() == 'VALUE = 10\n'
    assert setup.apply_patch_bundle(repo, bundle)['state'] == 'already_applied'
    (repo / 'one.py').write_text('VALUE = 99\n')
    with pytest.raises(ValueError, match='drift|modified|preimage'):
        setup.apply_patch_bundle(repo, bundle)


def test_patch_bundle_rejects_tampering_and_wrong_revision(tmp_path):
    setup = _setup()
    repo, bundle = _bundle(tmp_path)
    (bundle / 'enable.patch').write_text('not the reviewed patch\n')
    with pytest.raises(ValueError, match='hash|sha256'):
        setup.apply_patch_bundle(repo, bundle)
    assert (repo / 'one.py').read_text() == 'VALUE = 1\n'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    manifest['commit'] = 'a' * 40
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='revision|commit'):
        setup.apply_patch_bundle(repo, bundle)


def test_patch_bundle_rejects_symlink_escape(tmp_path):
    setup = _setup()
    repo, bundle = _bundle(tmp_path)
    outside = tmp_path / 'private.py'
    outside.write_text('VALUE = 1\n')
    (repo / 'one.py').unlink()
    (repo / 'one.py').symlink_to(outside)
    with pytest.raises(ValueError, match='symlink|escape'):
        setup.apply_patch_bundle(repo, bundle)
    assert outside.read_text() == 'VALUE = 1\n'


def test_runtime_constraints_preserve_exact_torch_triton():
    setup = _setup()
    text = setup.constraints_text({'torch': '2.11.0+cu128', 'triton': '3.6.0',
                                   'numpy': '1.26.4'})
    assert text.splitlines() == ['torch==2.11.0+cu128', 'triton==3.6.0',
                                 'numpy==1.26.4']
    with pytest.raises(ValueError, match='torch|triton'):
        setup.constraints_text({'torch': None, 'triton': '3.6.0', 'numpy': '1.26.4'})


def test_auth_probe_does_not_infer_or_serialize_other_credentials():
    setup = _setup()
    result = setup.provider_presence({
        'ANTHROPIC_API_KEY': '', 'ANTHROPIC_AUTH_TOKEN': 'private-sentinel',
        'OPENAI_API_KEY': 'private-openai-sentinel', 'CLAUDE_CODE_OAUTH_TOKEN': '',
    })
    assert result['anthropic_api_key_present'] is False
    assert result['claude_oauth_present'] is False
    assert result['openai_api_key_present'] is True
    assert result['authentication_verified'] is False
    assert 'sentinel' not in json.dumps(result)


def test_existing_install_destination_is_never_overwritten(tmp_path):
    setup = _setup()
    root = tmp_path / 'existing'
    root.mkdir()
    (root / 'user-file').write_text('keep')
    with pytest.raises(FileExistsError):
        setup.install('kernelagent', root, '/no/such/python')
    assert list(root.iterdir()) == [root / 'user-file']


def test_offline_install_sets_no_index_for_build_subprocesses(tmp_path):
    setup = _setup()
    wheels = tmp_path / 'wheelhouse'
    wheels.mkdir()
    env = setup._install_env(tmp_path, wheelhouse=wheels)
    assert env['PIP_NO_INDEX'] == '1'
    assert env['PIP_FIND_LINKS'] == str(wheels)
    assert env['PIP_CONFIG_FILE'] == setup.os.devnull
    with pytest.raises(ValueError, match='wheelhouse'):
        setup._install_env(tmp_path, wheelhouse=tmp_path / 'missing')


def test_source_license_records_and_patch_bytes_have_valid_digests():
    assets = SCRIPT.parent
    records = json.loads((assets / 'source-copies.json').read_text())
    for source in records['sources']:
        for row in source['license_files']:
            content = (ROOT / row['local_path']).read_bytes()
            assert hashlib.sha256(content).hexdigest() == row['sha256']
    manifest = json.loads((assets / 'forgekernel-sm80/manifest.json').read_text())
    for row in manifest['patches']:
        content = (assets / 'forgekernel-sm80' / row['path']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == row['sha256']


def test_aitune_install_uses_the_pinned_torch_extra(tmp_path):
    setup = _setup()
    (tmp_path / 'pyproject.toml').write_text(
        '[project.optional-dependencies]\ntorch211 = ["torch>=2.11,<2.12"]\n')
    assert setup.aitune_requirement(tmp_path, '2.11.0+cu128') == f'{tmp_path}[torch211]'
    with pytest.raises(ValueError, match='extra|Torch'):
        setup.aitune_requirement(tmp_path, '2.15.0')


def test_installer_and_patch_bundle_match_registered_adapter_pins():
    from flash_rt.vla.autotune.tools.external import aitune, forgekernel, kernelagent
    setup = _setup()
    for name, adapter in [('forgekernel', forgekernel), ('kernelagent', kernelagent),
                          ('aitune', aitune)]:
        assert setup.TOOLS[name]['tool_id'] == adapter.TOOL_ID
        assert setup.TOOLS[name]['commit'] == adapter.UPSTREAM_COMMIT
    manifest = json.loads((SCRIPT.parent / 'forgekernel-sm80/manifest.json').read_text())
    assert manifest['commit'] == forgekernel.UPSTREAM_COMMIT
