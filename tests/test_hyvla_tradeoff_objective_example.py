"""Exercise the shared worker with Hy's explicit performance/quality budget."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from flash_rt.vla.autotune.quality import validate_quality_contract
from flash_rt.vla.autotune.worker import _quality

ROOT=Path(__file__).resolve().parents[1]
CONTRACT=ROOT/'examples/vla_autotune/hyvla_tradeoff_quality.json'


def evaluate(tmp_path,actual):
    gate=json.loads(CONTRACT.read_text());expected=np.ones((40,32),dtype=np.float32)
    path=tmp_path/'reference.npz';np.savez(path,request_ids=np.array(['case']),actions_00000=expected)
    result=_quality([actual],['case'],path,gate)
    validate_quality_contract(result,gate)
    return result,gate


def test_Hy_tradeoff_accepts_bounded_nonexact_public_outputs(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,:20]*=1.04
    result,_=evaluate(tmp_path,actual)
    assert result['status']=='passed' and 0.<result['metrics']['max_relative_l2']<.05


def test_untrained_padding_no_longer_vetoes_public_action_quality(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,20:]=1000.
    result,_=evaluate(tmp_path,actual)
    assert result['status']=='passed' and result['metrics']['max_relative_l2']==0.


def test_twentieth_public_channel_still_participates_in_the_budget(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,19]+=1.
    result,_=evaluate(tmp_path,actual)
    assert result['status']=='failed' and result['metrics']['max_relative_l2']>.05


def test_public_scale_error_beyond_common_five_percent_budget_rejects(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,:20]*=1.06
    result,_=evaluate(tmp_path,actual)
    assert result['status']=='failed' and result['metrics']['max_relative_l2']>.05


def test_cosine_budget_still_rejects_direction_error_below_L2_limit(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,:20]+=np.tile([.048,-.048],10)
    result,_=evaluate(tmp_path,actual)
    assert result['status']=='failed' and result['metrics']['max_relative_l2']<.05 and result['metrics']['min_cosine']<.999


def test_new_objective_does_not_relabel_the_old_padded_oracle_gate(tmp_path):
    actual=np.ones((40,32),dtype=np.float32);actual[:,20:]=1000.
    result,gate=evaluate(tmp_path,actual)
    spec=importlib.util.spec_from_file_location('legacy_Hy_goal_gate',ROOT/'examples/vla_autotune/validate_hyvla.py')
    legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
    arrays={'expected_actions32':np.ones((40,32),dtype=np.float32),'expected_actions20':np.ones((40,20),dtype=np.float32)}
    historical=legacy.oracle_comparison(actual,arrays,sampler='euler',num_steps=10)
    assert result['status']=='passed' and historical['same_action_gate_passed'] is False
    assert historical['public20_gate_passed'] is True
    changed={**result,'scope':'historical-padded-official-oracle'}
    with pytest.raises(ValueError,match='contract'):validate_quality_contract(changed,gate)
