for ((i=50000; i<=400000;i+=50000)); do
python3 -m scripts.train --env MiniGrid-MultiRoom-N5-S25-v0 --algo ppo --model ppo-MultiRoom-N5-S25-$i --frames $i --procs 16 --save-interval 10
done
