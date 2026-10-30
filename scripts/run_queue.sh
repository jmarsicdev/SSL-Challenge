#!/usr/bin/env bash
# Run a queue of experiments sequentially, skipping any that already finished.
# Detached use (survives the Claude Code 2h background cap and the session):
#   nohup setsid scripts/run_queue.sh <queue-name> <queue-file> > /dev/null 2>&1 &
# Queue file lines:  <entry-script> <config-name> <seed>      e.g.  train_semisup.py semisup_r18sc_ts_k20_long 0
# Progress log: runs/queue_<queue-name>.log ; per-run logs stay in runs/<config>/s<seed>/train.log
set -u
cd "$(dirname "$0")/.."
name=${1:?queue name}; qfile=${2:?queue file}
log=runs/queue_${name}.log
PY=.venv/bin/python
echo "[$(date '+%F %T')] queue '$name' start, pid $$" >> "$log"
while read -r entry cfg seed; do
  [[ -z "${entry// }" || "$entry" == \#* ]] && continue
  if [[ -f "runs/$cfg/s$seed/metrics.json" ]]; then
    echo "[$(date '+%F %T')] skip $cfg s$seed (done)" >> "$log"; continue
  fi
  echo "[$(date '+%F %T')] start $cfg s$seed" >> "$log"
  t0=$(date +%s)
  if $PY "$entry" --config "configs/$cfg.yaml" --seed "$seed" > "runs/queue_${name}_current.out" 2>&1; then
    acc=$(grep -oE "BEST val acc [0-9.]+" "runs/queue_${name}_current.out" | tail -1)
    echo "[$(date '+%F %T')] done  $cfg s$seed  $acc  ($(( ($(date +%s)-t0)/60 )) min)" >> "$log"
  else
    echo "[$(date '+%F %T')] FAIL  $cfg s$seed  (see runs/$cfg/s$seed/train.log)" >> "$log"
    tail -5 "runs/queue_${name}_current.out" >> "$log"
  fi
done < "$qfile"
echo "[$(date '+%F %T')] queue '$name' finished" >> "$log"
