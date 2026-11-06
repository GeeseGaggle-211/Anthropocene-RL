H=/home/suyashbhatt11_gmail_com; C=$H/code/r5e/rl; S=$H/r5smoke; PY=$H/m9-bench/venv/bin/python
export ANTHROPOCENE_DOTNET=/usr/bin/dotnet ANTHROPOCENE_MODEL9_BRIDGE=$C/integration/bin/model9/Anthropocene.Model9Bridge.dll PYTHONPATH=$C/src OMP_NUM_THREADS=1
cd $C
ARGS="--normalizer-checkpoint $H/r5/actor-normalizer.pt --turns 3 --players 16 --width 128 --height 64 --separation 9 --personality dirichlet:1.0 --personality-key 4242 --return-tail randomized --tail-draws 2 --tail-cap 4 --jobs 1 --worlds 90100"
nice -n 10 timeout 1800 $PY -m anthropocene_rl.autonomous_v7.labor_teacher_data $ARGS --out $S/ref > $S/ref.log 2>&1 < /dev/null; echo "uninterrupted rc=$?"
python3 - <<PY
import json
a=json.load(open("$S/out/world-90100/meta.json")); b=json.load(open("$S/ref/world-90100/meta.json"))
print("data_digest resumed", a["data_digest"][:16], "uninterrupted", b["data_digest"][:16], "identical:", a["data_digest"]==b["data_digest"])
print("payload hashes identical:", a["payload"]==b["payload"], "| tail record identical:", a.get("return_tail_record",{}).get("cutoffs")==b.get("return_tail_record",{}).get("cutoffs"))
PY
rm -rf $S && echo "smoke dir deleted"
