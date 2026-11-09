"""WP-B item 4: prepare_iteration tolerates one float32 ulp on the trade value/share
columns only (cross-host sigmoid/exp), and stays exact everywhere else."""

from __future__ import annotations

import math

import numpy as np
import torch

from anthropocene_rl.autonomous_v7 import labor_a_ppo as ap
from anthropocene_rl.autonomous_v7 import labor_advantage as la

# The observed cross-host pair: sigmoid(-9.261749) as stored by one host and re-encoded on
# another (adjacent float32 values).
STORED = 9.497999417e-05
REENCODED = 9.498000145e-05


def _row(share: float = 0.25, value: float = 0.1) -> torch.Tensor:
    vector = np.zeros((1, la.ACTION_FEATURE_WIDTH))
    vector[0, ap._SHARE_COLUMNS] = share
    vector[0, ap._VALUE_COLUMNS] = value
    vector[0, 0] = 0.5
    return torch.from_numpy(vector)


def test_observed_values_are_one_float32_ulp_apart():
    a, b = np.float32(STORED), np.float32(REENCODED)
    assert np.nextafter(a, np.float32(1.0)) == b
    host = float(torch.sigmoid(torch.tensor(-9.261749, dtype=torch.float32)))
    assert host in (float(a), float(b))


def test_one_ulp_on_trade_share_is_accepted():
    stored = _row(share=float(np.float32(STORED)))
    reencoded = _row(share=float(np.float32(REENCODED)))
    assert not torch.equal(stored, reencoded)
    assert ap.features_match(reencoded, stored)
    assert ap.features_match(stored, reencoded)


def test_two_ulps_on_trade_share_are_refused():
    a = np.float32(STORED)
    two = np.nextafter(np.nextafter(a, np.float32(1.0)), np.float32(1.0))
    assert not ap.features_match(_row(share=float(two)), _row(share=float(a)))


def test_one_ulp_of_trade_value_is_accepted_and_more_is_refused():
    v = np.float32(1.7)
    up = np.nextafter(v, np.float32(10.0))
    scale = math.log(4.0)
    feature = lambda x: math.log(float(x)) / scale  # noqa: E731
    assert ap.features_match(_row(value=feature(up)), _row(value=feature(v)))
    far = np.nextafter(np.nextafter(np.nextafter(up, np.float32(10)), np.float32(10)), np.float32(10))
    assert not ap.features_match(_row(value=feature(far)), _row(value=feature(v)))


def test_other_columns_stay_exact():
    stored = _row()
    moved = stored.clone()
    moved[0, 0] = float(np.nextafter(np.float32(0.5), np.float32(1.0)))
    assert not ap.features_match(moved, stored)
    assert ap.features_match(stored.clone(), stored)
    assert not ap.features_match(stored[:, :-1], stored)
    nan = stored.clone()
    nan[0, ap._SHARE_COLUMNS] = float("nan")
    assert not ap.features_match(nan, stored)


def test_tolerant_blocks_are_the_trade_value_and_share_blocks():
    assert len(ap._SHARE_COLUMNS) == 1 and ap._SHARE_COLUMNS[0] == la.ACTION_FEATURE_WIDTH - 1
    assert len(ap._VALUE_COLUMNS) == la.TRADE_GROUP_COUNT
    assert ap._VALUE_COLUMNS[-1] == la.ACTION_FEATURE_WIDTH - 2
