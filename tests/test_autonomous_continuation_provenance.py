"""A JSON report must reconstruct the checkpoint's integer-keyed historical ledger."""

import copy
import json

import pytest

from anthropocene_rl.autonomous_v4.provenance import continuation_checkpoint_provenance


def test_typed_provenance_roundtrip_without_weakening_other_identity_fields():
    campaign = dict(
        config=dict(
            version="autonomous-collapse-continuation-v4",
            seeds=[29, 43],
            parent_checkpoint_sha256={"29": "parent29", "43": "parent43"},
        ),
        provenance=dict(
            cumulative_accounting=dict(prior_by_seed={17: 32768, 29: 8192, 43: 8192}),
            bank_sha256="original",
        ),
    )
    expected = dict(campaign["provenance"], training_seed=29, parent_checkpoint_sha256="parent29")
    serialized = json.loads(json.dumps(campaign))
    before = copy.deepcopy(serialized)
    assert continuation_checkpoint_provenance(serialized, 29) == expected
    assert serialized == before
    serialized["provenance"]["bank_sha256"] = "changed"
    assert continuation_checkpoint_provenance(serialized, 29) != expected
    serialized["provenance"]["cumulative_accounting"]["prior_by_seed"]["43"] += 1
    with pytest.raises(ValueError):
        continuation_checkpoint_provenance(serialized, 29)
