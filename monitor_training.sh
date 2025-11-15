#!/bin/bash
# Monitor training progress in real-time

LOG_FILE="experiments/quantum_*/training.log"

echo "=================================="
echo "TRAINING MONITOR"
echo "=================================="
echo ""

# Find the log file
LOG=$(ls -t $LOG_FILE 2>/dev/null | head -1)

if [ -z "$LOG" ]; then
    echo "No training log found!"
    echo "Looking for: $LOG_FILE"
    exit 1
fi

echo "Monitoring: $LOG"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""
echo "=================================="
echo ""

# Show last 20 lines and follow
tail -n 20 -f "$LOG"
