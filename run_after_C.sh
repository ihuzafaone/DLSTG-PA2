#!/bin/bash
PY=$HOME/miniconda3/envs/PA1_DL/bin/python
cd "$(dirname "$0")"
$PY src/screen_d.py > logs/screen_d.log 2>&1
for m in D DB; do for s in 0 1; do $PY src/train.py --model $m --seed $s > logs/${m}_s${s}.log 2>&1; done; done
