#!/bin/bash
# Unattended G0/G2-G4 pipeline for 25 Oct, run every 5 minutes by the g0watchdog systemd service.
# Idempotent: each stage skips when its output exists and relaunches from partials after a preemption.
# Stages: A critic3 (narrowed value support) | B G4 DAgger (pinned 55e2a15) | C G3 search teacher
#         D final actor refit on teacher + DAgger + search | E combine, evaluate, upload, stop the VM.
export PATH=/snap/bin:/usr/local/bin:/usr/bin:/bin:$PATH
H=/home/suyashbhatt11_gmail_com; R=$H/g0; PY=$H/m9-bench/venv/bin/python
F=$H/code/g0-final/rl; G4=$H/code/g4-dagger/rl; GCS=gs://anthropocene-rl-datasets-508912/datasets/labor-g0-20261024
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$H/code/g0-final/rl/integration/bin/model9/Anthropocene.Model9Bridge.dll
log() { echo "$(date -Is) $*" >> $R/pipeline.log; }
running() { pgrep -f "$1" > /dev/null; }
status() { gcloud storage cp $R/pipeline.log $GCS/status/pipeline.log > /dev/null 2>&1; gcloud storage cp $R/dagger/driver.log $GCS/status/dagger-driver.log > /dev/null 2>&1; }
[ -f $R/ALLDONE ] && { [ -f $H/r5/ACTIVE ] || sudo -n shutdown -h now; exit 0; }
[ -d $H/code/g0-final ] || ~/git/checkout.sh 8baf309 a645431 g0-final > /dev/null 2>&1 || { log "checkout g0-final failed"; exit 1; }
[ -d $H/code/g4-dagger ] || ~/git/checkout.sh 55e2a15 a645431 g4-dagger > /dev/null 2>&1

# A. Critic with narrowed support, continuing from the critic refit's latest weights.
if [ ! -f $R/eval-critic3.json ] && ! running 'g0stageA'; then
  (setsid nohup bash -c '
    exec -a g0stageA bash -s' > /dev/null 2>&1 <<EOF &
H=$H; R=$R; PY=$PY; F=$F
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$ANTHROPOCENE_MODEL9_BRIDGE
log() { echo "\$(date -Is) \$*" >> \$R/pipeline.log; }
pkill -f '^bash /home/[^ ]*/g0critic2[.]sh'; pkill -f 'g0launch[.]py pretrain-critic --checkpoint /home/[^ ]*/g0/critic(\.pt|2\.resume\.pt) '
cd \$F
if [ ! -f \$R/critic3-conv.pt ]; then
  src=\$R/critic2.pt; [ -f \$src ] || src=\$R/critic2.pt.partial
  log "A convert \$src to support -0.5,2.5,255"
  PYTHONPATH=\$F/src \$PY -m anthropocene_rl.autonomous_v7.labor_critic_support convert --checkpoint \$src --critic-value-support=-0.5,2.5,255 --out \$R/critic3-conv.pt >> \$R/critic3.log 2>&1 || { log "A convert failed"; exit 1; }
  PYTHONPATH=\$F/src \$PY -c "import torch; d=torch.load('\$R/critic3-conv.pt',weights_only=False); d.pop('g0_partial',None); torch.save(d,'\$R/critic3-conv.pt')"
fi
if [ ! -f \$R/critic3.pt ]; then
  ck=\$R/critic3-conv.pt; ep=4
  if [ -f \$R/critic3.pt.partial ]; then
    done=\$(PYTHONPATH=\$F/src \$PY -c "import torch;print(torch.load('\$R/critic3.pt.partial',weights_only=False)['g0_partial']['steps'])")
    cp \$R/critic3.pt.partial \$R/critic3.resume.pt; ck=\$R/critic3.resume.pt; ep=\$(( 4 - done / 375 )); [ \$ep -lt 1 ] && ep=1
  fi
  log "A fit critic3 from \$ck epochs=\$ep"
  PYTHONPATH=\$F/src OMP_NUM_THREADS=28 \$PY \$F/tools/g0launch.py pretrain-critic --checkpoint \$ck --dataset \$R/fit-data --epochs \$ep --batch-size 256 --lr 3e-4 --max-grad-norm 1.0 --allow-vm-scale --out \$R/critic3.pt >> \$R/critic3.log 2>&1 || { log "A fit failed"; exit 1; }
fi
PYTHONPATH=\$F/src OMP_NUM_THREADS=16 \$PY -m anthropocene_rl.autonomous_v7.labor_pretrain_eval critic-calibration --checkpoint \$R/critic3.pt --dataset \$R/cal-data --out \$R/eval-critic3.json > \$R/eval-critic3.log 2>&1 && log "A critic3 calibrated: \$(grep -E 'years +all|guard +all|feasibility +all' \$R/eval-critic3.log | tr -s ' ' | tr '\n' ';')"
EOF
  )
  log "A launched"
fi

# B. G4 learner-state recovery (keeps its pinned checkout; resumes from markers and partials).
if [ ! -f $R/dagger/round-5.done ] && ! running 'g4_dagger[.]sh'; then
  (cd $G4 && ANTHROPOCENE_MODEL9_BRIDGE=$G4/integration/bin/model9/Anthropocene.Model9Bridge.dll G4_WORLDS=90100-90129 G4_STARTS=6 G4_JOBS=20 setsid nohup bash tools/g4_dagger.sh >> $H/g4.out 2>&1 < /dev/null &)
  log "B G4 (re)launched"
fi

# C. G3 search teacher, after the critic stage frees its threads.
if [ -f $R/eval-critic3.json ] && [ ! -f $R/search-data/dataset.json ] && ! running 'labor_search'; then
  ACT=$R/pretrained.pt
  resume=""; [ -d $R/search-data ] && { mv $R/search-data $R/search-data.partial-$(date +%s); }
  (cd $F && PYTHONPATH=$F/src OMP_NUM_THREADS=1 setsid nohup $PY -m anthropocene_rl.autonomous_v7.labor_search \
     --normalizer-checkpoint $ACT --actor-checkpoint $ACT --worlds 90100-90159 --source h12 --turns 12 --roots-per-world 8 \
     --candidates 10 --actor-samples 4 --horizons 1,3 --keep all,3 --reeval-streams 2 --personality dirichlet:1 \
     --jobs 20 --out $R/search-data >> $R/search.log 2>&1 < /dev/null &)
  log "C search launched"
fi

# D. Final actor refit on teacher + all DAgger rounds + search labels (equal source mass).
if [ -f $R/dagger/round-5.done ] && [ -f $R/search-data/dataset.json ] && [ ! -f $R/final-actor.pt ] && ! running 'final-actor'; then
  ck=$R/dagger/round-5/actor.pt; ep=2
  [ -f $R/final-actor.pt.partial ] && { cp $R/final-actor.pt.partial $R/final-actor.resume.pt; ck=$R/final-actor.resume.pt; ep=1; }
  args="--dataset $R/fit-data"; for r in 1 2 3 4 5; do args="$args --dataset $R/dagger/round-$r/data"; done; args="$args --dataset $R/search-data"
  (cd $F && PYTHONPATH=$F/src OMP_NUM_THREADS=40 setsid nohup $PY $F/tools/g0launch.py pretrain-bc --checkpoint $ck $args \
     --epochs $ep --batch-size 256 --lr 5e-4 --max-grad-norm 1.0 --allow-vm-scale --out $R/final-actor.pt >> $R/final-actor.log 2>&1 < /dev/null &)
  log "D final refit launched from $ck"
fi

# E. Combine, evaluate, upload, stop.
if [ -f $R/final-actor.pt ] && [ -f $R/critic3.pt ] && [ ! -f $R/ALLDONE ] && ! running 'g0stageE'; then
  (setsid nohup bash -c 'exec -a g0stageE bash -s' > /dev/null 2>&1 <<EOF &
R=$R; PY=$PY; F=$F; GCS=$GCS
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$ANTHROPOCENE_MODEL9_BRIDGE
log() { echo "\$(date -Is) \$*" >> \$R/pipeline.log; }
cd \$F
PYTHONPATH=\$F/src \$PY - <<PYEOF >> \$R/pipeline.log 2>&1
import torch
a = torch.load("\$R/final-actor.pt", map_location="cpu", weights_only=False)
c = torch.load("\$R/critic3.pt", map_location="cpu", weights_only=False)
out = dict(a)
for k in ("task_critic", "cost_critics", "critic_optimizer", "heuristic_critic_pretraining", "critic_value_support_conversions"):
    if k in c: out[k] = c[k]
out["labor_model_contract"] = dict(a["labor_model_contract"], criticValueSupport=c["labor_model_contract"]["criticValueSupport"])
out["g0_combination"] = {"actor_from": "final-actor.pt (G4 round 5 + search refit)", "critics_from": "critic3.pt (narrowed support)"}
torch.save(out, "\$R/pretrained-final.pt"); print("combined pretrained-final.pt")
PYEOF
EV="\$PY -m anthropocene_rl.autonomous_v7.labor_pretrain_eval"; export PYTHONPATH=\$F/src
OMP_NUM_THREADS=8 \$EV actor-fidelity --checkpoint \$R/pretrained-final.pt --dataset \$R/cal-data --out \$R/eval-final-actor.json > \$R/eval-final-actor.log 2>&1 &
OMP_NUM_THREADS=12 \$EV critic-calibration --checkpoint \$R/pretrained-final.pt --dataset \$R/cal-data --out \$R/eval-final-critic.json > \$R/eval-final-critic.log 2>&1 &
OMP_NUM_THREADS=1 \$EV autonomous --checkpoint \$R/pretrained-final.pt --worlds 2000-2007 --turns 8 --modes h12,deterministic,stochastic --players 16 --width 128 --height 64 --separation 9 --jobs 24 --out \$R/eval-final-auto.json > \$R/eval-final-auto.log 2>&1 &
wait
log "E final eval: \$(grep -E 'FAIL|PASS' \$R/eval-final-auto.log | head -3 | tr -s ' ' | tr '\n' ';')"
for f in critic3.pt final-actor.pt pretrained-final.pt eval-critic3.json eval-final-actor.json eval-final-critic.json eval-final-auto.json pipeline.log; do gcloud storage cp \$R/\$f \$GCS/final/ > /dev/null 2>&1; done
gcloud storage cp -r \$R/search-data \$GCS/ > /dev/null 2>&1
for r in 1 2 3 4 5; do gcloud storage cp -r \$R/dagger/round-\$r \$GCS/dagger/ > /dev/null 2>&1; done
[ -f $H/r5/ACTIVE ] || { gcloud compute instances remove-resource-policies anthropocene-trainer-60 --zone=us-central1-c --resource-policies=g0-autostart4 > /dev/null 2>&1 && log "E removed autostart schedule"; }
log "E uploaded; ALLDONE; stopping VM"
touch \$R/ALLDONE
gcloud storage cp \$R/pipeline.log \$GCS/status/pipeline.log > /dev/null 2>&1
[ -f $H/r5/ACTIVE ] || sudo -n shutdown -h now
EOF
  )
  log "E launched"
fi
status
