# Read access for the autonomous Elder curriculum

The user authorizes attaching a GCP identity to this runtime to read the existing
GCS dataset. The runtime currently has no outbound identity bindings, ADC profile
or GCP configuration. Its exposed tool reports status but cannot attach an
account or edit the environment configuration. Complete the account attachment
through the runtime's secure environment configuration, without pasting keys
into chat. Prefer its supported short-lived/federated identity mechanism.

Grant the selected principal `roles/storage.objectViewer` on bucket
`anthropocene-rl-datasets-508912`. No project administrator or Storage write
permission is required. Allow HTTPS to `storage.googleapis.com` and
`oauth2.googleapis.com`; federated identity may also require
`sts.googleapis.com` and `iamcredentials.googleapis.com`.

After attachment, inspect current cloud environment status. Proceed only when
`observations_current` is true, the chosen identity is `ready`, and its required
network policy is `enforced`. Inspect the version1 non-secret manifest named by
`OIC_MANIFEST_PATH` and select the explicit GCP alias. Do not borrow Google Drive
credentials or silently select another default account.

The Storage SDK is prepared in the existing learner environment. To verify the
identity and historically pinned manifest, use:

```bash
.venv/bin/python scripts/check_gcs_runtime_identity.py \
  --identity-alias <attached-gcp-alias> \
  --project <gcp-project-id> \
  --output runs/gcs-runtime-identity-v1.json
```

This reads the exact historical manifest generation and checks generation,
size and content SHA256. It creates no credentials, copies no secret files and
writes no remote objects. The retained catalog has499 records:348 TRAIN,
75 DEVELOPMENT and76 HELDOUT. Materialize the pinned shards into a new bounded
cache, preserve original world/lineage/counterfactual partitions, and rebuild
observations/masks through native environment5. The old512-feature observations
and old actions are not new-model training data. HELDOUT stays sealed.

Account attachment is still pending until readiness and a real authenticated
read succeed. This document does not claim that creating a local SDK profile
creates a cloud principal or grants IAM permission.
