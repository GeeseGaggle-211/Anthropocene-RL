"""Read-only review: existing tensors/JSON/source bytes; no inference, samples or native steps."""
import gzip
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean

import torch

ROOT = Path('/workspace/Anthropocene-RL')
NATIVE = Path('/workspace/Anthropocene-elder')
OUT = Path('/tmp/autonomous-elder-training-assessment-astra-audit.json')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

probe_path = ROOT / 'runs/autonomous-canary-preflight-v1/probe-v2.pt'
probe = torch.load(probe_path, weights_only=False)
x = probe['x']
ema = x[:, 84:117].reshape(-1, 11, 3)
bank_path = ROOT / 'datasets/autonomous-ppo-gate-v1/manifest.json'
bank = json.loads(bank_path.read_text())['entries']
memory = []
for entry in bank:
    path = bank_path.parent / entry['snapshot']
    payload = gzip.open(path, 'rb').read()
    markers = re.findall(b'ElderMemory.{0,15}', payload)
    memory.append(dict(id=entry['id'], sha256=digest(path),
        camp_memory_markers=len(markers),
        null_memory_markers=sum(z.startswith(b'ElderMemory.\x03nil\x86\x01') for z in markers)))

campaign_path = ROOT / 'runs/autonomous-critic-horizon-gate-v5/summary.json'
campaign = json.loads(campaign_path.read_text())
panel_path = ROOT / 'runs/autonomous-horizon-welfare-v5-fixed/summary.json'
panel = json.loads(panel_path.read_text())
heur_path = ROOT / 'runs/autonomous-heuristic-welfare-v1/summary.json'
heur = json.loads(heur_path.read_text())
ids = set(panel['development_ids'])
native = next(g for g in heur['groups'] if g['policy'] == 'heuristic-native' and g['limit'] == 25)
native_episodes = [e for e in native['episodes'] if e['id'] in ids]
native_family_raw = {f: mean(e['raw_welfare'] for e in native_episodes if e['family'] == f)
                     for f in sorted({e['family'] for e in native_episodes})}
economics = []
for c in panel['cells']:
    long = c['groups'][-1]
    ep_path = panel_path.parent / (c['id'] + '-episodes.json')
    episodes = json.loads(ep_path.read_text())
    economics.append(dict(id=c['id'], aggregate=long['aggregate'],
        family_raw_welfare={f: v['raw_welfare'] for f,v in long['families'].items()},
        family_raw_welfare_relative_to_native={f: v['raw_welfare']/native_family_raw[f]-1 for f,v in long['families'].items()},
        focal_events=long['focal_events'],
        survival_final_populations=[e['final_population'] for e in episodes if e['family']=='survival'],
        field_loss_on_movement_intervals=sum(i['field_tiles_before'] > i['field_tiles_after'] and i['events']['movement'] > 0 for e in episodes for i in e['intervals']),
        food_exhausted_intervals=sum(i['food_stock_after'] <= 0 for e in episodes for i in e['intervals']),
        episode_sha256=digest(ep_path)))

training = []
for c in campaign['cells']:
    h = c['health'][-1]['health']
    fam = {f: [e for e in c['episodes'] if e['family']==f] for f in sorted({e['family'] for e in c['episodes']})}
    training.append(dict(id=c['id'],
        mean_preupdate_mse_over_variance=mean(u['critic_mse_over_return_variance'] for u in c['updates']),
        mean_preupdate_value_bias=mean(u['critic_bias'] for u in c['updates']),
        mean_actor_minibatches=mean(u['actor_updates'] for u in c['updates']),
        actor_tanh_saturation=[l['saturated_fraction'] for l in h['actor_layers']],
        active_simplex_clamp_fraction=h['active_clamp_fraction'],
        categorical=h['categorical'],
        family_episode_counts={f:len(es) for f,es in fam.items()},
        family_mean_undiscounted_raw_welfare={f:mean(e['welfare']*e['initial_population'] for e in es) for f,es in fam.items()},
        focal_trades=sum(e['events']['focal_trades'] for e in c['episodes'])))

source_paths = [
    ROOT/'integration/SnapshotScenarioFactory.cs',
    ROOT/'src/anthropocene_rl/autonomous_v4/contracts.py',
    ROOT/'src/anthropocene_rl/autonomous_v4/training.py',
    ROOT/'src/anthropocene_rl/policies/autonomous_ppo_v2.py',
    ROOT/'src/anthropocene_rl/training/autonomous_ppo_v3.py',
    NATIVE/'Assets/_Anthropocene/Simulation/AutonomousElderEnvironment.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/AutonomousElder.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/AutonomousElderTrade.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/TurnPlanning.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/TurnResolver.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/ElderObservationBuilder.cs',
    NATIVE/'Assets/_Anthropocene/Simulation/ElderInstitutionMemory.cs',
    ROOT/'runs/drive-architecture-gate-v5.txt',
]
result = dict(version='independent-astra-training-assessment-readonly-v1',
    native_steps=0, ppo_updates=0, sampled_actions=0,
    evidence_sha256={str(p):digest(p) for p in [probe_path,bank_path,campaign_path,panel_path,heur_path,*source_paths]},
    observations=dict(rows=len(x), features=x.shape[1],
        previous_decision_nonzero_values=int((x[:,40:56]!=0).sum()),
        previous_decision_present_rows=int((x[:,40]!=0).sum()),
        maturity_nonzero_values=int((x[:,117:120]!=0).sum()),
        ema_short_medium_max_abs_difference=float((ema[:,:,0]-ema[:,:,1]).abs().max()),
        ema_short_long_max_abs_difference=float((ema[:,:,0]-ema[:,:,2]).abs().max()),
        note='Probe verifies observed rows; source control-flow establishes why this is not persisted history. Snapshot nil scan is binary-format marker inspection, not a new native restore.'),
    bank_memory_markers=memory,
    bank_partitions={p:dict(snapshots=sum(e['partition']==p for e in bank),worlds=sorted({e['world_seed'] for e in bank if e['partition']==p})) for p in sorted({e['partition'] for e in bank})},
    native_matched_subset_raw_welfare_by_family=native_family_raw,
    training=training,economics=economics,
    horizon_math={str(h):dict(gamma=math.exp(-20/h), eight_decision_bootstrap_discount=math.exp(-160/h), gae_e_folding_years=1/(1/h-math.log(.95)/20)) for h in [100,400]},
    limits='Descriptive static audit only. No causal intervention. Worlds, samplers and model variants are not independent replications of each other.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(OUT)
print(json.dumps(result['observations'],indent=2))
print('Matched native raw welfare by family:',native_family_raw)
for e in economics:
    print(e['id'],{f:round(v*100,1) for f,v in e['family_raw_welfare_relative_to_native'].items()},e['survival_final_populations'])
