#!/bin/bash
# Training script for MiniGrid-MultiRoom-N2-S4-v0 environment
# This environment has 2 rooms with max size 4x4

for ((i=50000; i<=400000;i+=50000)); do
    python3 -m scripts.train --env MiniGrid-MultiRoom-N2-S4-v0 --algo ppo --model ppo-MultiRoom-N2-S4-$i --frames $i --procs 16 --save-interval 10
done
