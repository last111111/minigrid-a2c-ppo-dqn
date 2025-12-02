#!/bin/bash

# Test script for success rate tracking
# This will run a short PPO training to verify the success rate tracking works

echo "========================================="
echo "Testing Success Rate Tracking for PPO"
echo "========================================="
echo ""

# Run a short training session
python scripts/train.py \
    --algo ppo \
    --env MiniGrid-Empty-5x5-v0 \
    --model test_success_rate_tracking \
    --episodes 200 \
    --procs 4 \
    --frames-per-proc 128 \
    --epochs 4 \
    --batch-size 128 \
    --save-interval 10 \
    --log-interval 1

echo ""
echo "========================================="
echo "Test Complete!"
echo "========================================="
echo ""
echo "Check the following:"
echo "1. Training logs should show 'SR100' (Success Rate last 100 episodes)"
echo "2. Training logs should show 'SRb' (Success Rate current batch)"
echo "3. A file 'success_rate_plot.png' should be generated in:"
echo "   storage/test_success_rate_tracking/"
echo ""
echo "To view the plot:"
echo "  - Navigate to storage/test_success_rate_tracking/"
echo "  - Open success_rate_plot.png"
echo ""
