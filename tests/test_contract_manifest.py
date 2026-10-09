from pathlib import Path

from anthropocene_rl.config import load_yaml
from anthropocene_rl.contracts import (
    ACTION_SCHEMA_VERSION,
    ENVIRONMENT_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    SIMULATOR_COMMIT,
    TENSOR_FEATURE_COUNT,
)


def test_native_contract_pin() -> None:
    assert SIMULATOR_COMMIT == "81bae49d71b23c32d956991601546d3cb2064ad7"
    assert (OBSERVATION_SCHEMA_VERSION, ACTION_SCHEMA_VERSION, ENVIRONMENT_SCHEMA_VERSION) == (
        1,
        1,
        1,
    )
    assert TENSOR_FEATURE_COUNT == 512


def test_manifest_matches_native_contract_pin() -> None:
    manifest = load_yaml(Path("configs/simulator/elder-v1.yaml"))
    assert manifest["commit"] == SIMULATOR_COMMIT
    assert manifest["observation_schema_version"] == OBSERVATION_SCHEMA_VERSION
    assert manifest["action_schema_version"] == ACTION_SCHEMA_VERSION
    assert manifest["environment_schema_version"] == ENVIRONMENT_SCHEMA_VERSION
    assert manifest["tensor_feature_count"] == TENSOR_FEATURE_COUNT
