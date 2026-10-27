#!/bin/bash
# r5e resume smoke: world 90100, 3 turns, short tail; kill mid-run, then --resume. Outside ~/r5.
H=/home/suyashbhatt11_gmail_com; C=$H/code/r5e/rl; S=$H/r5smoke; PY=$H/m9-bench/venv/bin/python
exec > $H/r5smoke.log 2>&1
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$C/integration/bin/model9/Anthropocene.Model9Bridge.dll PYTHONPATH=$C/src OMP_NUM_THREADS=1
rm -rf $S; mkdir -p $S; cd $C
ARGS="--normalizer-checkpoint $H/r5/actor-normalizer.pt --turns 3 --players 16 --width 128 --height 64 --separation 9 --personality dirichlet:1.0 --personality-key 4242 --return-tail randomized --tail-draws 2 --tail-cap 4 --jobs 1 --worlds 90100 --out $S/out"
date -Is; echo "run 1 (to be killed)"
setsid nice -n 10 $PY -m anthropocene_rl.autonomous_v7.labor_teacher_data $ARGS > $S/run1.log 2>&1 < /dev/null &
P=$!
for i in $(seq 1 120); do
  n=$(find $S/out/_work -mindepth 2 -maxdepth 3 -type d 2>/dev/null | wc -l)
  d=$(grep -c '"decisions"' $S/run1.log)
  [ $d -ge 2 ] && break; sleep 5
done
echo "before kill: progress lines $(grep -c '"decisions"' $S/run1.log); work entries:"; find $S/out -maxdepth 4 | sed "s#$S/##" | head -20
kill -9 -- -$P 2>/dev/null; sleep 3; pkill -9 -f "r5smoke/out" ; sleep 2
echo "killed; remaining: $(pgrep -fc 'r5smoke/out')"
tail -2 $S/run1.log | cut -c1-200
date -Is; echo "run 2 --resume"
nice -n 10 timeout 1800 $PY -m anthropocene_rl.autonomous_v7.labor_teacher_data $ARGS --resume > $S/run2.log 2>&1 < /dev/null; echo "resume rc=$?"
grep -iE 'resum|checkpoint' $S/run2.log | head -5 | cut -c1-300
tail -3 $S/run2.log | cut -c1-300
ls $S/out; python3 -c "import json;m=json.load(open('$S/out/dataset.json'));print('dataset.json shards', list(m['shards']))"
python3 -c "import json;m=json.load(open('$S/out/world-90100/meta.json'));print('rows',m['rows'],'decisions',m['decisions'],'tail',m['tail_decisions'])"
date -Is; echo SMOKE_END
