from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from anthropocene_rl.autonomous_v7 import economic_diagnostics as diagnostics


def transition(seed: int = 70, arm: str = "base", world: int = 2000, turn: int = 1) -> dict:
    identity = f"run:{arm}:{seed}:{world}:{turn}:7"
    head_widths = {
        "strategic": 5,
        "service": 6,
        "clothingRoute": 1,
        "shelterRoute": 1,
        "movement": 1,
        "intentTile": 1,
        "fission": 1,
        "commit": 1,
        "tradeValue": 7,
        "tradeGate": 7,
        "tradeShare": 1,
    }
    requested = {name: [0.0] * width for name, width in head_widths.items()}
    accepted = {name: [0.0] * width for name, width in head_widths.items()}
    realised = {name: [0.0] * width for name, width in head_widths.items()}
    requested["strategic"][0] = 0.8
    accepted["strategic"][0] = realised["strategic"][0] = 0.7
    active_mask = {name: [False] * width for name, width in head_widths.items()}
    active_mask["strategic"][0] = True
    available_mask = {name: list(values) for name, values in active_mask.items()}
    economic = {
        "schemaVersion": 1,
        "transitionId": identity,
        "trajectoryId": f"trajectory:{arm}:{seed}:{world}",
        "decisionTurn": turn,
        "campId": 7,
        "societyId": 3,
        "years": 20.0,
        "domains": {
            "population": {"endingLinked": 100.0, "qol": 0.7},
            "cultivation": {
                "funnels": [
                    {
                        "metric": "fieldFood",
                        "unit": "food_units",
                        "nativeDemand": 10.0,
                        "actorCappedRequest": 8.0,
                        "compilerGrant": 6.0,
                        "physicalReceipt": 5.0,
                        "finalService": 5.0,
                        "active": True,
                        "constraintReasons": ["labour"],
                    }
                ],
                "constraintReasons": ["labour"],
            },
            "herding": {
                "requestedAnimals": 4.0,
                "grantedAnimals": 3.0,
                "realisedAnimals": 2.0,
            },
            "manufacturing": {},
            "trade": {"realisedUtility": 1.0},
        },
        "units": {
            "domains.population.endingLinked": "people",
            "domains.population.qol": "qol",
            "goods[0].output": "good_units",
        },
        "unavailable": {},
        "goods": [
            {
                "good": "Cloth",
                "unit": "good_units",
                "usefulDemand": 10.0,
                "plannedOutputCeiling": 8.0,
                "output": 6.0,
                "workerIntervals": 3.0,
                "workerYears": 1.0,
                "equipmentCoveredWork": 1.0,
                "stone": 0.0,
                "clay": 0.0,
                "wood": 0.0,
                "plantFood": 2.0,
                "animalFood": 0.0,
                "animalFabric": 0.0,
                "cloth": 0.0,
                "housingMaterial": 0.0,
                "tools": 0.0,
            }
        ],
        "materials": [
            {
                "material": "Wood",
                "unit": "material_units",
                "requested": 5.0,
                "granted": 4.0,
                "received": 3.0,
                "workerIntervals": 2.0,
                "workerYears": 0.5,
                "extractionRate": 8.0,
            }
        ],
        "trades": [],
        "conservation": {"populationResidual": 0.0, "foodResidual": 0.0},
    }
    candidate_bonus = 2.0 if arm == "candidate" else 0.0
    return {
        "schemaVersion": 1,
        "transitionId": identity,
        "trajectoryId": economic["trajectoryId"],
        "decisionTurn": turn,
        "campId": 7,
        "societyId": 3,
        "rollout": 0,
        "runId": f"run:{arm}:{seed}",
        "arm": arm,
        "optimizerSeed": seed,
        "worldSeed": world,
        "policyVersion": 1,
        "evidenceClass": "prospective_diagnostic_training_trace",
        "years": 20.0,
        "gamma": 0.95,
        "gaeLambda": 0.95,
        "terminated": False,
        "truncated": False,
        "economicDiagnostics": economic,
        "actionDiagnostics": {
            "schemaVersion": 1,
            "trajectoryId": economic["trajectoryId"],
            "decisionTurn": turn,
            "campId": 7,
            "requested": requested,
            "accepted": accepted,
            "realised": realised,
            "requestedMask": available_mask,
            "acceptedMask": available_mask,
            "realisedMask": available_mask,
            "activeMask": active_mask,
        },
        "objectives": {
            "welfare": {
                "reward": 10.0 + candidate_bonus,
                "shapedReward": 10.0 + candidate_bonus,
                "prediction": 8.0,
                "bootstrap": 4.0,
                "tdResidual": -2.0,
                "target": 10.0,
                "advantage": -1.0,
                "referenceScale": 10.0,
                "scaledAdvantage": -0.1,
                "clippedAdvantage": -0.1,
                "independentReturn": 12.0,
            }
        },
        "finalPolicyAdvantage": -0.5,
        "cost": {"prediction": 0.2, "bootstrap": 0.1, "target": 0.3, "advantage": -0.1},
        "sampledAction": {"strategic": [0.8, 0.0, 0.0, 0.0, 0.0]},
        "personality": [1.0, 0.0, 0.0, 0.0],
        "capabilityRegime": "mixed",
        "provenance": {"binarySha256": "a" * 64, "checkpointSha256": "b" * 64},
        "unavailable": {},
    }


def update_pair(rows: list[dict]) -> list[dict]:
    identities = [row["transitionId"] for row in rows]
    digest = hashlib.sha256(diagnostics.canonical_bytes(identities)).hexdigest()
    common = {
        "schemaVersion": 1,
        "evidenceClass": "prospective_diagnostic_training_trace",
        "runId": rows[0]["runId"],
        "arm": rows[0]["arm"],
        "optimizerSeed": rows[0]["optimizerSeed"],
        "rollout": 0,
        "epoch": 0,
        "minibatch": 0,
        "batchTransitionIds": identities,
        "batchRowDigest": digest,
    }
    return [
        {
            **common,
            "event": "mutation-start",
            "policyVersionBefore": 0,
            "beforeHashes": {"actor": "a" * 64, "critic": "b" * 64, "cost": "c" * 64},
        },
        {
            **common,
            "event": "mutation-complete",
            "policyVersionAfter": 1,
            "afterHashes": {"actor": "d" * 64, "critic": "e" * 64, "cost": "f" * 64},
            "ppoRatioMean": 1.01,
            "ppoClipFraction": 0.1,
            "approxKl": 0.01,
            "losses": {"value": 2.0, "value_by": {"welfare": 2.0}},
            "gradientMetrics": {
                "actorGradientNormBeforeClipping": 0.5,
                "actorClippingScale": 1.0,
                "criticGradientNormBeforeClipping": 1.5,
                "criticClippingScale": 2.0 / 3.0,
                "costGradientNormBeforeClipping": 0.25,
                "costClippingScale": 1.0,
            },
            "parameterUpdateNorms": {"actor": 0.02, "critic": 0.03, "cost": 0.01},
        },
    ]


def write_artifact(
    root: Path, rows: list[dict], *, baseline: str = "base", updates: list[dict] | None = None
) -> None:
    root.mkdir(parents=True)
    manifest = {
        "schemaVersion": 1,
        "files": {"transitions": "transitions.jsonl", "updates": "updates.jsonl"},
        "baselineArm": baseline,
        "sliceBoundaries": {
            "population": [0, 50, 100, 150, 200, 1000],
            "qol": [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            "turn": [0, 5, 10, 20, 40, 1000],
        },
        "tradeCreditMaxLag": 1,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    transition_bytes = b"".join(diagnostics.canonical_bytes(row) + b"\n" for row in rows)
    (root / "transitions.jsonl").write_bytes(transition_bytes)
    update_rows = updates or []
    update_bytes = b"".join(diagnostics.canonical_bytes(row) + b"\n" for row in update_rows)
    (root / "updates.jsonl").write_bytes(update_bytes)
    done = {
        "schemaVersion": 1,
        "transitions": {"rows": len(rows), "sha256": hashlib.sha256(transition_bytes).hexdigest()},
        "updates": {
            "rows": len(update_rows),
            "sha256": hashlib.sha256(update_bytes).hexdigest(),
        },
        "merkleRoot": diagnostics.merkle_root([*rows, *update_rows]),
    }
    (root / "done.json").write_text(json.dumps(done), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_complete_trace_writes_deterministic_physical_and_critic_reports(tmp_path: Path) -> None:
    rows = [
        transition(seed, arm, world, turn)
        for seed in (70, 71, 72)
        for arm in ("base", "candidate")
        for world in (2000, 2001)
        for turn in (1, 2)
    ]
    artifact = tmp_path / "trace"
    write_artifact(artifact, rows)
    output = tmp_path / "report"
    report = diagnostics.aggregate_trace_directories([artifact], output)

    assert report["transitions"] == 24
    assert not report["historicalAttributionAvailable"]
    assert report["conservationFailures"] == []
    assert all(
        (output / name).is_file()
        for name in (
            "field_dictionary.json",
            "decision_rows.csv",
            "optimizer_updates.csv",
            "optimizer_mutations.csv",
            "optimizer_update_summary.csv",
            "domain_summary.csv",
            "critic_calibration.csv",
            "action_alignment.csv",
            "matched_fork_effects.csv",
            "paired_arm_contrasts.csv",
            "trade_credit_latency.csv",
            "report.json",
            "report.md",
        )
    )
    domains = read_csv(output / "domain_summary.csv")
    field = next(
        row for row in domains if row["domain"] == "cultivation" and row["metric"] == "fieldFood"
    )
    assert float(field["nativeDemandToActorCappedRequestRatioOfSumsLoss"]) == pytest.approx(0.2)
    assert float(field["actorCappedRequestToCompilerGrantRatioOfSumsLoss"]) == pytest.approx(0.25)
    cultivation_slices = {
        row["slice"]
        for row in domains
        if row["domain"] == "cultivation" and row["metric"] == "fieldFood"
    }
    assert cultivation_slices == {
        "overall",
        "optimizer_seed",
        "world_seed",
        "capability_regime",
        "personality_band",
        "population_quintile",
        "qol_quintile",
        "turn_band",
        "intervention_reason",
    }
    calibration = read_csv(output / "critic_calibration.csv")
    overall = next(
        row
        for row in calibration
        if row["targetClass"] == "lambda_target" and row["slice"] == "overall"
    )
    assert float(overall["signedBias"]) == pytest.approx(-2.0)
    independent = next(
        row
        for row in calibration
        if row["targetClass"] == "independent_return" and row["slice"] == "overall"
    )
    assert float(independent["signedBias"]) == pytest.approx(-4.0)
    assert any(
        row["slice"] == "intervention_reason" and row["sliceValue"] == "labour"
        for row in calibration
    )
    alignment = read_csv(output / "action_alignment.csv")
    assert all(row["head"] == "strategic[0]" and row["count"] == "12" for row in alignment)
    assert all(row["activeOnly"] == "True" for row in alignment)
    assert all(row["interventionRate"] == "1.0" for row in alignment)
    assert all(row["interventionDenominator"] == "12" for row in alignment)
    assert {
        "descriptive_action_to_final_policy_advantage",
        "descriptive_action_to_objective_advantage",
        "descriptive_action_to_economic_physicalReceipt",
    }.issubset({row["relationship"] for row in alignment})
    contrasts = read_csv(output / "paired_arm_contrasts.csv")
    across = next(row for row in contrasts if row["level"] == "across_optimizer_seeds")
    assert float(across["difference"]) == pytest.approx(2.0)
    assert read_csv(output / "trade_credit_latency.csv")
    dictionary = json.loads((output / "field_dictionary.json").read_text())
    assert any(
        field["field"] == "objectives.welfare.tdResidual" and field["signRange"] == "signed"
        for field in dictionary["fields"]
    )
    goods_output = next(
        field
        for field in dictionary["fields"]
        if field["field"] == "economicDiagnostics.goods[0].output"
    )
    assert goods_output["units"] == ["good_units"]


def test_action_alignment_filters_inactive_heads_and_treats_routes_as_categories() -> None:
    rows = []
    for index in range(6):
        row = transition(turn=index + 1)
        category = index % 2
        row["sampledAction"]["clothingRoute"] = category
        diagnostic = row["actionDiagnostics"]
        diagnostic["activeMask"]["clothingRoute"][0] = True
        for phase in ("requestedMask", "acceptedMask", "realisedMask"):
            diagnostic[phase]["clothingRoute"][0] = True
        for phase in ("requested", "accepted", "realised"):
            diagnostic[phase]["clothingRoute"][0] = category
        row["finalPolicyAdvantage"] = -1.0 if category == 0 else 1.0
        rows.append(row)
    rows[0]["actionDiagnostics"]["activeMask"]["strategic"][0] = False

    alignment = diagnostics.action_alignment(rows)
    strategic = next(
        row
        for row in alignment
        if row["head"] == "strategic[0]"
        and row["relationship"] == "descriptive_action_to_final_policy_advantage"
    )
    assert strategic["count"] == 5
    assert strategic["inactiveRowsExcluded"] == 1
    routes = [
        row
        for row in alignment
        if row["head"] == "clothingRoute"
        and row["relationship"] == "descriptive_action_to_final_policy_advantage"
    ]
    assert {row["category"] for row in routes} == {0, 1}
    assert all(row["associationMethod"] == "category-mean-difference" for row in routes)
    assert all(row["pearson"] is None for row in routes)
    assert {row["meanDifference"] for row in routes} == {-1.0, 1.0}


def test_optimizer_boundaries_join_to_exact_preupdate_rows(tmp_path: Path) -> None:
    rows = [transition(70, "base", 2000, turn) for turn in (1, 2)]
    artifact = tmp_path / "trace"
    write_artifact(artifact, rows, updates=update_pair(rows))
    output = tmp_path / "report"
    report = diagnostics.aggregate_trace_directories([artifact], output)

    assert report["counts"]["optimizerMutations"] == 1
    mutations = read_csv(output / "optimizer_mutations.csv")
    assert len(mutations) == 1
    assert mutations[0]["criticHashChanged"] == "True"
    assert float(mutations[0]["parameterUpdateNorms.critic"]) == pytest.approx(0.03)
    assert float(mutations[0]["batch.welfare.tdResidualAbsMean"]) == pytest.approx(2.0)
    rollup = read_csv(output / "optimizer_update_summary.csv")
    overall = next(row for row in rollup if row["slice"] == "overall")
    assert float(overall["gradientMetrics.criticClippingScale.mean"]) == pytest.approx(2 / 3)
    assert overall["zeroCriticUpdateRate"] == "0.0"


def test_optimizer_batch_digest_or_membership_mismatch_is_rejected(tmp_path: Path) -> None:
    rows = [transition()]
    updates = update_pair(rows)
    updates[1]["batchRowDigest"] = "0" * 64
    artifact = tmp_path / "trace"
    write_artifact(artifact, rows, updates=updates)
    with pytest.raises(ValueError, match="batch-row digest"):
        diagnostics.load_traces([artifact])


def test_input_order_does_not_change_report_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    write_artifact(first, [transition(70, "base", 2000, 1)])
    write_artifact(second, [transition(71, "candidate", 2001, 1)])
    out_a, out_b = tmp_path / "out-a", tmp_path / "out-b"
    diagnostics.aggregate_trace_directories([first, second], out_a)
    diagnostics.aggregate_trace_directories([second, first], out_b)
    for name in (
        "field_dictionary.json",
        "decision_rows.csv",
        "domain_summary.csv",
        "critic_calibration.csv",
        "action_alignment.csv",
        "report.json",
        "report.md",
    ):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes(), name


def test_cross_artifact_slice_manifest_presence_must_match(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    write_artifact(first, [transition(70, "base", 2000, 1)])
    write_artifact(second, [transition(71, "base", 2001, 1)])
    manifest = json.loads((second / "manifest.json").read_text())
    manifest.pop("sliceBoundaries")
    (second / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="presence of sliceBoundaries"):
        diagnostics.aggregate_trace_directories([first, second], tmp_path / "report")


def test_signed_training_values_are_valid_but_physical_negatives_and_sealed_seeds_fail(
    tmp_path: Path,
) -> None:
    row = transition()
    diagnostics.validate_transition(row)  # negative TD/advantages are valid
    bad = json.loads(json.dumps(row))
    bad["economicDiagnostics"]["goods"][0]["output"] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        diagnostics.validate_transition(bad)
    sealed = json.loads(json.dumps(row))
    sealed["optimizerSeed"] = 3000
    with pytest.raises(ValueError, match="confirmation seed"):
        diagnostics.validate_transition(sealed)
    missing = json.loads(json.dumps(row))
    missing["economicDiagnostics"]["domains"]["population"]["qol"] = None
    with pytest.raises(ValueError, match="unavailable reason"):
        diagnostics.validate_transition(missing)


def test_unbounded_finite_placeholder_is_not_summarized_as_an_observation() -> None:
    row = transition()
    cultivation = row["economicDiagnostics"]["domains"]["cultivation"]
    cultivation["plannerMarginalReturn"] = 0.0
    cultivation["plannerMarginalReturnBoundState"] = "positive-unbounded"
    dictionary = diagnostics.field_dictionary([row])
    field = next(
        item
        for item in dictionary["fields"]
        if item["field"]
        == "economicDiagnostics.domains.cultivation.plannerMarginalReturn"
    )
    assert field["unbounded"] == 1
    assert field["minimum"] is None and field["maximum"] is None


def test_incomplete_or_mutated_trace_is_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "trace"
    write_artifact(artifact, [transition()])
    with (artifact / "transitions.jsonl").open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="digest"):
        diagnostics.load_artifact(artifact)
