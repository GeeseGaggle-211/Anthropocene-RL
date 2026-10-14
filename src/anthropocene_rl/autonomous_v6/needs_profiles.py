"""Explicit physics profile: retain the Native6 tensor/action layout and old bank."""

import base64
import hashlib

from anthropocene_rl.autonomous_v6.contracts import validate_schema

PROFILE = "subsistence-needs-v13"
GEAR_BASELINE = 0.80
SAVE_ENVELOPE = "ANTHROPOCENE-14"


def validate_needs_profile_schema(schema):
    validate_schema(schema)
    if PROFILE not in schema.get("needsProfiles", []) or SAVE_ENVELOPE not in schema.get(
        "additionalSaveEnvelopes", []
    ):
        raise ValueError("Native bridge does not support the explicit subsistence needs profile")


def derive_profile_snapshot(transport, snapshot, focal):
    """Only the job-needs rule changes; native save/version/hash owns semantics."""
    validate_needs_profile_schema(transport.schema())
    result = transport._request(
        dict(
            op="autonomous6_prepare_probe",
            snapshot=base64.b64encode(snapshot).decode("ascii"),
            focal=focal,
            transform="subsistence_needs_v13",
        )
    )
    derived = base64.b64decode(result["snapshot"], validate=True)
    return derived, dict(
        profile=PROFILE,
        gear_baseline=GEAR_BASELINE,
        save_envelope=SAVE_ENVELOPE,
        source_sha256=hashlib.sha256(snapshot).hexdigest(),
        derived_sha256=hashlib.sha256(derived).hexdigest(),
        changed_rule="Gatherer job gear baseline; hunter goods cohort remains excluded",
    )
