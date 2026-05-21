#!/bin/bash
# Wait for SigLIP Phase 1 to complete (18/18 COMP DONE), then run Phase 2
LOG="./log"

echo "Waiting for SigLIP Phase 1 (18/18)..."
while true; do
    done=0
    for seed in 1 2 3; do
        for lmb in 30 50 100 300 1000 10000; do
            log="${LOG}/google--siglip-base-patch16-224/rnorm_ldlq64_seed${seed}/lmbda${lmb}.log"
            [ -f "$log" ] && grep -q "COMP DONE" "$log" 2>/dev/null && ((done++))
        done
    done
    echo "  SigLIP comp: $done/18 done"
    [ $done -ge 18 ] && break
    sleep 30
done

echo "SigLIP Phase 1 complete! Starting Phase 2..."
bash "$(dirname $0)/siglip_phase2.sh"
