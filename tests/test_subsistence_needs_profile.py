import base64

import pytest

from anthropocene_rl.autonomous_v6.contracts import HISTORY_NAMES
from anthropocene_rl.autonomous_v6.needs_profiles import (
    PROFILE,
    derive_profile_snapshot,
    validate_needs_profile_schema,
)


def test_profile_requires_native_capability_and_preserves_source():
    schema = dict(
        actionVersion=4,
        observationVersion=6,
        environmentVersion=6,
        replayVersion=6,
        historyWidth=301,
        featureWidth=1062,
        saveEnvelope="ANTHROPOCENE-13",
        controlTransform="clip((latent-.05)/.90,0,1)",
        herdRetention="post-growth",
        historyFeatures=list(HISTORY_NAMES),
    )
    with pytest.raises(ValueError, match="does not support"):
        validate_needs_profile_schema(schema)
    schema.update(needsProfiles=["legacy", PROFILE], additionalSaveEnvelopes=["ANTHROPOCENE-14"])

    class Transport:
        def schema(self):
            return schema

        def _request(self, payload):
            assert base64.b64decode(payload["snapshot"]) == b"source"
            assert payload["focal"] == 42 and payload["transform"] == "subsistence_needs_v13"
            return dict(snapshot=base64.b64encode(b"derived").decode("ascii"))

    source = b"source"
    derived, receipt = derive_profile_snapshot(Transport(), source, 42)
    assert source == b"source" and derived == b"derived"
    assert receipt["source_sha256"] != receipt["derived_sha256"]
    assert receipt["gear_baseline"] == 0.8 and receipt["save_envelope"] == "ANTHROPOCENE-14"
