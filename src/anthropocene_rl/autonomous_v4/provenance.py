"""Reconstruct the typed provenance of immutable continuation-v4 checkpoints."""

import copy


def continuation_checkpoint_provenance(campaign, seed):
    """JSON object keys are strings; the frozen torch ledger used integer seed keys.

    Restore only this declared field. All other provenance remains subject to the
    trainer/inference loader's exact equality check; no checkpoint bytes are edited.
    """
    if campaign["config"]["version"] != "autonomous-collapse-continuation-v4":
        raise ValueError("Unsupported continuation provenance version")
    if seed not in campaign["config"]["seeds"]:
        raise ValueError("Undeclared training seed")
    provenance = copy.deepcopy(campaign["provenance"])
    ledger = provenance["cumulative_accounting"]["prior_by_seed"]
    expected = {"17": 32768, "29": 8192, "43": 8192}
    if {str(k): v for k, v in ledger.items()} != expected or len(ledger) != 3:
        raise ValueError("Invalid frozen prior-seed ledger")
    provenance["cumulative_accounting"]["prior_by_seed"] = {int(k): v for k, v in ledger.items()}
    return dict(
        provenance,
        training_seed=seed,
        parent_checkpoint_sha256=campaign["config"]["parent_checkpoint_sha256"][str(seed)],
    )
