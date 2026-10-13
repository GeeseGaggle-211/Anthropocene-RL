# Current autonomous Elder direct-control scope

Model4/checkpoint4 retains native action3 and observation/environment/replay4.
The new critic and H100/H400 comparison do not silently change those action
semantics. The old five-head model remains separately versioned and intact.

| Economic choice | Current policy control | Remaining restriction |
| --- | --- | --- |
| Labor | Masked13-role shares, stable integer apportionment | Physical resident/tool/material constraints remain native. |
| Recipes and gathering | Conditional masked recipe shares, gatherer food/material split | Currently covered recipe faces do not include every advanced profession. |
| Cultivation | Farmer staffing and fraction devoted to field expansion | Native cultivation ranks/selects actual hexes; no policy hex-selection head. |
| Breeding | Masked coefficient0..1 directly sets managed breeding intensity | Husbandry is absent from the current cultivation-only training bank. |
| Retained herd | Fraction0..1 compiles to floor(opening domestic animals × fraction) | Cannot request a retained herd above opening size; native postbreeding slaughter removes excess. This restricts net herd-growth retention. |
| Trade | Quote enable, known counterparty, stocked offered good, distinct requested good, stock fraction and exchange ratio | One quote/eight contacts; reciprocal learned peer training remains incomplete. |
| Movement | Stay/known-site categorical choice | Knowledge and physical movement rules remain native. |
| Fission and migration | Consequences of physical demographics, resources, locations and relationships | Newly created daughters currently use separate native heuristic controllers, not automatic learned-Elder assignment. |

There is no learned old-style cultivation acreage goal or independent craft
production-target strategy in this contract. Staffing, expansion effort and
conditional recipes authorize actual production; legal poor choices are not
rescued. Retained herd control still compiles to the native slaughter target,
which should not be described as an unrestricted livestock growth decision.

Any mechanical pass of this gate is cultivation-centric and village-local. It
cannot certify explicit cultivation-hex authority, husbandry autonomy or all-domain
scale-out. The next action-contract change should expose knowledge-limited legal
cultivation hex choices and separate breeding from postgrowth slaughter/retention
so offspring can be retained. It needs new action/observation/replay versions,
conditional likelihood/mask/chronology/conservation tests, matching physical
observations and a separately bounded curriculum gate. Do not reinterpret existing
checkpoint heads or rewrite completed experimental evidence to supply that change.
