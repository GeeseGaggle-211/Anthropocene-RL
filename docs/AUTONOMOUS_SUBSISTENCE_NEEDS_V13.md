# Subsistence job needs profile V13

The requested experimental change provides an80% self-made clothing and housing baseline
for resident Gatherers, implicit and explicitly assigned. Dispatched Hunter
entities already fall outside the resident goods cohort and remain covered by
that separation. A single Gatherer does not exempt the whole settlement.

Authority is the accepted job ledger for the prior production cohort. For N
residents and G Gatherers, manufactured clothing/housing serves N-.80G person-equivalents. Non-Gatherers
receive manufactured coverage first, followed by upgrades to Gatherer gear.
Effective coverage is (.80G + physical covered equivalents)/N. Goods are
consumed only for N-.80G; no cloth, fabric, buildings, storage or trade stock is
created. Real shelter still determines social cohesion. Food/water shortages,
recipes, productivity, tools, chronology, growth, migration and fission rules
retain their existing mechanics. The changed need coverage naturally changes
their inputs. Current needs affect next turn's committed-QoL growth ceiling;
QoL mortality itself does not directly subtract recorded people.

Use explicit NeolithicConfig.SubsistenceNeedsCovered=true for this new profile.
It defaults false in the simulation API and stays false in historical saves
and baseline fixtures. The experimental branch's NewGame path enables it after
loading tuning, so newly started playable worlds use the rule; loading a save
does not enable it.
The flag and the separate physical clothing-coverage fraction are nonserialized
in the old metadata. Enabled worlds use ANTHROPOCENE-14: a flags byte records
existing history/growth extension presence; after the existing payload/extensions
a sorted camp table stores optional physical clothing coverage. Older envelopes
continue their exact bytes/fingerprints and reject no historical data. Save14
is rejected by old readers rather than silently executing different needs.

The self-made-gear component is capped at.80 even during the early expectation
ramp. Non-Gatherers retain that ramp; manufactured upgrades can reach full
coverage. With unchanged real-shelter cohesion, a well-fed24-person camp has
QoL.790536 and a positive growth ceiling. A100-person unsheltered camp remains
below the growth threshold; the rule guarantees no arbitrary population floor.

Pure demand/feedstock/needs helpers and migration/fission forecasts use the same
job exemption. Physical clothing coverage is kept separately, so automatically
met needs cannot be transferred as fabricated clothing when jobs or villages
change. Projected leavers use the existing stable proportional worker split;
arrivals become implicit Gatherers, as in current Immigration. Daughter forecasts
use the existing split that releases their Farmer assignments to Gatherers.

Cost is O(assigned roles) for needs/demand, with at most13 exposed roles; pure
proportional split estimates are O(roles log roles), bounded independently of dt,
map size or population. No new hunter/global ecological scan or sub-turn loop.

Native checks cover all/mixed/zero Gatherers, explicit assignments, non-exempt
needs and stock conservation, real social cohesion, forecasts/feedstock, actual
craft output, dispatch/reintegration, chronology, save/load/replay, and old-mode
fingerprint equality. Paired scripted food-rich/food-scarce rollouts test whether
the gear baseline allows small-camp recovery while shortages remain consequential. Model8
and action4/native6 wire widths remain unchanged; new physics/source and Save14
are pinned independently. Existing checkpoints are retained as behavioral probes,
not promoted calibrated critics for the changed profile. The unexecuted V12
old-physics collection plans are suspended, with their original pins preserved.
