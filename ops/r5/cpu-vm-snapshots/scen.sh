H=/home/suyashbhatt11_gmail_com; D=$H/r5; DS=${1:-$D/fit8-b01}
cd $H/code/r5d/rl
export PYTHONPATH=$H/code/r5d/rl/src OMP_NUM_THREADS=4
nice -n 10 timeout 3600 $H/m9-bench/venv/bin/python -m anthropocene_rl.autonomous_v7.labor_scenarios weights --dataset $DS --normalizer-checkpoint $D/actor-normalizer.pt < /dev/null > $D/scenarios-weights.out 2>&1; echo "weights rc=$?"
head -100 $D/scenarios-weights.out
nice -n 10 timeout 1800 $H/m9-bench/venv/bin/python $D/scen2.py $DS < /dev/null
grep -rn 'TradeFeatureNames' $H/code/r5d/sim/Assets/_Anthropocene/Simulation/*.cs $H/code/r5d/rl/integration/*.cs | head -2
grep -rn -A3 'TradeFeatureNames *=' $H/code/r5d/sim/Assets/_Anthropocene/Simulation/*.cs $H/code/r5d/rl/integration/*.cs | head -6
