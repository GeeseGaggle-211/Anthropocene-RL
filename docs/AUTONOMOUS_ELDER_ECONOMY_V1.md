# Autonomous Village Elder Economy — contract 1.0 (experimental)

Native base: `5559961164d6f9da8e6d20c1dc563ecec3eb04b7`.
Learner base: `fa71f370aa50982552584861e61682d7cd2897e0` (verified main).
Identifiers: action 2, observation 3, environment 3, replay 3. The existing
observation/environment 2 and five-head action 1 APIs remain unchanged.

## Authority and chronology

WorldState, population ledgers, inventories, material/ecology sources, fields,
herds and native TurnRecord remain authoritative. The new environment owns a
SaveCodec clone. All controlled living camps submit exactly one action on the
same committed WEGO boundary. Validation and compilation are pure; invalid
schema, dimensions, masked mass, nonfinite or out-of-range values fail before
resolution. Legal but economically poor actions receive no staffing rescue.
The native one-way prepare / escrow / resources / production / needs /
demographics / immigration / fission / knowledge sequence remains in force.
Newborns and fission daughters do not work at their creation boundary.

## Action 2

Workforce shares follow WorkforceRoles.All (13 entries), normalized on the
structurally legal simplex. Largest remainder assigns all current resident
workers; ties use registry order. Availability depends on capability/technology,
never local profitability, demand, stock shortages or tool supply.

Each role has conditional activity shares, in columns Food, Stone, Clay, Wood,
then the 11 CraftGood enum entries. Gatherers select food/material work;
craft roles select structurally available recipes. Other roles have an inert
row. Activity shares allocate worker-time within the apportioned profession;
unusable time is idle and is never reassigned to another activity. Native
recipe prerequisites, tools, source availability and storage cap realization.
FieldExpansionFraction selects the portion of Farmer time reserved for capital
expansion; remaining time operates fields. RetainedHerdFraction and Breeding
remain explicit bounded controls; unavailable controls are canonical zero.
Movement is stay-only for this vertical slice.

Production orders carry explicit activity budgets and authorize surplus beyond
local-use demand. Dependency order (tool inputs before equipment work, Animal
Fabric before tents, Housing Material before housing) is unchanged. Materials
can be extracted into inventory without a downstream local-use claim. Raw
materials retain the native unbounded-storage semantics. Food harvesting and
food reserve/consumption rules remain native in this first slice; surplus
claims here concern materials and manufactured goods.

At most one stock-only quote per Elder per boundary: known counterparty slot,
give good, want good, GiveQuantity and WantQuantity, each in [0.000001, 1,000,000].
Goods follow TradeGood (11 entries). No quote is explicit. Quantities specify
an exact offered ratio and a maximum lot. Reciprocal quotes match when goods
are reversed and both limits permit the lower-ID camp's offered ratio. Lots
scale down to both opening stocks and both quantity caps; no same-boundary
resale, made-to-order promise, utility veto or food-safety rescue. Native stock
escrow/settlement and receiver storage limits remain authoritative. Contacts
are owner-visible/remembered, bounded to eight and aligned exactly with the
context contact rows (out-of-range rows are masked), with no remote inventories or
other policies' quotes exposed. Native heuristic trade excludes controlled
camps; uncontrolled camps retain their native policy.

## Observation 3 and replay 3

Reuse the owner-filtered 512-slot observation-2 tensor as contextual features,
append explicit profession/activity masks, own tradeable stocks, active-goods
mask and eight counterparty-slot masks. Server-side slot IDs are metadata;
only bounded slots enter actions. All observations precede submission and are
rebuilt after the common step. Stale/unknown/moved/out-of-range contacts cannot
settle. Replay stores initial SaveCodec bytes, versions, build/config hashes,
controlled IDs, requested actions, background orders, staffing and endpoint
fingerprints; verification reruns the complete boundary and checks results.

## Learner and cost model

Use independent masked Dirichlet distributions for workforce and active
conditional activity simplexes. Zero/singleton simplexes are deterministic and
have zero log likelihood/entropy; absent entries are exactly zero and consume
no RNG. PPO scores the requested continuous shares, never rounded staffing or
realized output. Persist masks with rollouts. Explicit capital/livestock and
quote decisions have their own bounded/categorical likelihoods when learned;
scripted/disabled channels must not be represented as learned PPO samples.

Compiler cost is O(P log P + P A), P=13, A=15. Quote matching is O(C K + C log C)
with one quote and K<=8 contacts per controlled camp, using camp indexes.
Native physical cost continues to follow active tiles, camps and local sources,
not population or historical years. Observation construction reuses the existing owner-filtered builders, including
their per-owner world/history scans; those costs can grow faster than linearly
when all camps are controlled and are included in benchmarks. Learned quote
combinatorics are not introduced. Measure complete steps with
observation construction, background decisions, prepare and resolve; disclose
fixture scale and avoid extrapolating small benchmarks to mature empires.

## Acceptance gates and exclusions

Scripted allocation/recipe interventions must change actual native receipts.
Cover surplus, insufficient food/materials/tools/workers, storage, matched and
incompatible quotes, conservation, structural masks, simultaneous submission
order, knowledge isolation, save/replay, probability parity and a bounded CPU
PPO update. Run old-mode regression and matched complete-step benchmarks.
Independent final code review is required. No expensive PPO, GPU ecology,
merges, historical artifact overwrites or claims of trained policy quality.
Deferred surfaces must be called out in the delivery report. The initial learner learns workforce, activities and bounded capital/livestock
controls; quotes are native/scripted only. Its Gym space fixes quote=None, while
the multi-Elder transport accepts explicit quotes. The contextual transform is
stateless signed-log compression for mechanics checks, not a calibrated training
normalizer. The transient PolicyAuthority labor flag is cleared before Close and
is not saved; pending actions/production/escrow are never saved mid-turn.
