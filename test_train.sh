#!/bin/bash
# Quick test with just 10000 frames to verify the success rate display works
python3 -m scripts.train --env MiniGrid-FourDoor-v0 --algo ppo --model ppo-test --frames 10000 --procs 16 --save-interval 10
