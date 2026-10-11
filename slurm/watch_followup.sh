#!/bin/bash
# Watch the follow-up. Verifies RUNNING jobs are producing training steps —
# the earlier round had thirteen jobs sit for hours producing nothing because
# the watcher only checked queue membership.
STALL_MIN=12
LAST=""
for i in $(seq 1 400); do
  SNAP=$(ssh vista2 'cd $WORK/downscaling
    R=$(squeue -u $USER -h -o "%i %j %t %M" 2>/dev/null | grep -E "aofu")
    D=$(ls -d checkpoints/{arch-*,amse,ctx}_s*/last.pt 2>/dev/null | wc -l)
    echo "COUNTS $D"
    echo "$R"
    for j in $(echo "$R" | awk "\$3==\"R\"{print \$1}"); do
      for f in logs/aofu_$j.out logs/aofubld_$j.out; do
        [ -f "$f" ] && echo "STEPS $j $(grep -c "^\[train\] e" "$f" 2>/dev/null)"
      done
    done
    [ -d data_store_sc/aorc_ctx/dataset.zarr ] && echo "CTXSTORE yes" || echo "CTXSTORE no"' 2>/dev/null)

  # An unreachable host returns an empty snapshot. Treating that as "queue
  # empty" made the watcher announce FINISHED when the SSH control socket
  # dropped — it reported success for a run whose state it could not see.
  # A snapshot without a COUNTS line is a connection failure, not a result.
  if ! echo "$SNAP" | grep -q "^COUNTS"; then
    MISSES=$(( ${MISSES:-0} + 1 ))
    echo "[$(date +%H:%M:%S)] cannot reach Vista (attempt $MISSES) — job state UNKNOWN, not finished"
    if [ "$MISSES" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] giving up after $MISSES failures; reopen the ssh session and restart the watcher"
      exit 4
    fi
    sleep 120
    continue
  fi
  MISSES=0

  DONE=$(echo "$SNAP" | awk '/^COUNTS/{print $2}')
  CTX=$(echo "$SNAP" | awk '/^CTXSTORE/{print $2}')
  NRUN=$(echo "$SNAP" | awk '$3=="R"' | wc -l | tr -d ' ')
  NPD=$(echo "$SNAP"  | awk '$3=="PD"' | wc -l | tr -d ' ')

  STALL=""
  while read -r _ j steps; do
    [ -z "$j" ] && continue
    EL=$(echo "$SNAP" | awk -v j="$j" '$1==j && $3=="R"{print $4}')
    MIN=$(echo "$EL" | awk -F: '{if(NF==3) print $1*60+$2; else if(NF==2) print $1; else print 0}')
    # the build job legitimately logs no training steps
    case "$(echo "$SNAP" | awk -v j="$j" '$1==j{print $2}')" in aofubld) continue;; esac
    [ "${steps:-0}" -eq 0 ] && [ "${MIN:-0}" -ge "$STALL_MIN" ] && STALL="$STALL $j(${EL})"
  done < <(echo "$SNAP" | grep "^STEPS")

  if echo "$SNAP" | grep -q "DependencyNeverSatisfied"; then
    echo "[$(date +%H:%M:%S)] DEPENDENCY NEVER SATISFIED — the build job failed; shards are parked"
    exit 3
  fi

  if [ -n "$STALL" ]; then
    echo "[$(date +%H:%M:%S)] STALL:$STALL — running with zero training steps"
    exit 2
  fi

  KEY="$DONE/$NRUN/$NPD/$CTX"
  if [ "$KEY" != "$LAST" ]; then
    echo "[$(date +%H:%M:%S)] done $DONE/18 | running $NRUN | pending $NPD | ctx store $CTX"
    LAST="$KEY"
  fi

  if [ "$DONE" = "18" ] || { [ "$NRUN" = "0" ] && [ "$NPD" = "0" ]; }; then
    echo "[$(date +%H:%M:%S)] FINISHED: $DONE/18"
    ssh vista2 'cd $WORK/downscaling; ls -d checkpoints/{arch-*,amse,ctx}_s*/last.pt 2>/dev/null | sed "s|checkpoints/||;s|/last.pt||" | tr "\n" " "; echo'
    exit 0
  fi
  sleep 120
done
