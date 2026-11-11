#!/bin/sh
# usage: run_sets.sh <label-prefix>
cd /d/task-sebaie
B="C:/Users/youse/AppData/Local/Temp/claude/77938f618f9936c0/afcbcbce-8f83-49b3-a2f8-9d6a1782d4bb/scratchpad/bench"
for s in main issue5 issue7 issue8 issue32; do
  if [ "$s" = main ]; then c=evaluation/cases.json; else c=evaluation/cases_$s.json; fi
  echo "SET $s start $(date +%T)"
  python -m evaluation.run --label "$1-$s" --cases "$c" --out-dir "$B/results" --notes "benchmark $1, nogpu-startup image, GPU" 2>&1 | tail -4
  echo "SET $s done $(date +%T)"
done
echo "ALL DONE $1"
