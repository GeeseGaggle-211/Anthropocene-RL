# Independent preflight review

Reviewer: `/root/independent_review`, independent code/evidence review.

Verdict: GO for the frozen v3 remedial native canary only. All 13 seed17 kappa=.05 tasks passed across the three raw artifacts, including enable gain .701093/critic MSE .345127 and all four recipes. Recorded policy/trainer source hashes match reviewed files. Entry probes, probability tests, native parallel/resume checks and input pinning are satisfactory.

Approved envelope: seeds29/43, 4,096 additional native training transitions each (8,192 total), at most four CPU workers, one Torch thread, 1,800 seconds; pinned parents/probe/config/bridge/bank; numerical, replay, recovery, optimizer-stall and resource pauses; economic outcomes diagnostic. Final13-head retained-Adam tests on both new checkpoints and separate final review are required before broader scale-out. Historical failed gates remain failed.

Reviewed config SHA256: `fb4c7bb725b046c3f01e3bc48fc661959d66b23a9119180f0cc47a516b7de18a`.
Reviewed runner SHA256: `d3e773f8ceb415e493abf72399138af70ce0a9052aadf916b3088a7753ca91fa`.
Learner prelaunch commit: `ec725030a1377c5cf45c586cd12b48f706104999`.

Independent tests:5 passed,2 native opt-in skipped. Primary native opt-in run:7 passed. Runner review caught and resolved self-hash provenance checks; frozen digests are now enforced before worker creation. Per-update durable traces preserve early failures.
