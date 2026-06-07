#!/bin/bash
# Auto-resume LLM quarterly batch until complete
# Each run processes ~200 calls before timeout, this loops until done

CHECKPOINT="/home/z/my-project/delta_jfe/llm_quarterly_checkpoint.json"
SCRIPT="/home/z/my-project/delta_jfe/llm_quarterly_batch.py"

while true; do
    # Check if complete
    COMPLETED=$(python3 -c "import json; ckpt=json.load(open('$CHECKPOINT')); print(len(ckpt.get('completed',{})))")
    TOTAL=40020
    PCT=$(python3 -c "print(f'{$COMPLETED/$TOTAL*100:.0f}%')")
    echo "[$(date +%H:%M:%S)] Progress: $COMPLETED/$TOTAL ($PCT)"
    
    if [ "$COMPLETED" -ge "$TOTAL" ]; then
        echo "ALL DONE!"
        break
    fi
    
    # Run batch (timeout after 8 min to avoid tool timeout)
    timeout 480 python3 -u "$SCRIPT" 2>&1 | tail -3
    
    # Brief pause
    sleep 2
done
