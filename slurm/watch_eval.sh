#!/bin/bash
J=1055429
MISSES=0
for i in $(seq 1 200); do
  ST=$(ssh -o ConnectTimeout=20 vista2 "squeue -j $J -h -o '%t' 2>/dev/null" 2>/dev/null)
  RC=$?
  if [ $RC -ne 0 ]; then
    MISSES=$((MISSES+1)); echo "[$(date +%H:%M:%S)] vista unreachable ($MISSES) — state UNKNOWN"
    [ $MISSES -ge 5 ] && { echo "giving up; reopen ssh and rerun"; exit 4; }
    sleep 120; continue
  fi
  MISSES=0
  ST=$(echo "$ST" | tr -d ' ')
  if [ -z "$ST" ]; then
    echo "=== eval job $J left the queue at $(date +%H:%M:%S) ==="
    ssh vista2 'cd $WORK/downscaling
      echo "--- errors ---"; grep -iE "error|traceback" logs/fueval_1055429.out logs/fueval_1055429.err 2>/dev/null | head -4
      echo; echo "=== SUMMARY ==="; sed -n "/=== PRODUCTS/,\$p" logs/fueval_1055429.out 2>/dev/null'
    exit 0
  fi
  [ "$ST" = "R" ] && [ $((i % 4)) -eq 0 ] && echo "[$(date +%H:%M:%S)] running"
  sleep 60
done
