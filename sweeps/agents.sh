#!/usr/bin/env bash

gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_agents=$1

echo "Got ${gpus} GPUs"
echo "Putting ${num_agents} per GPU"

for (( i=0; i<$gpus; i++ ))
do
    for (( j=0; j<$num_agents; j++ ))
    do
        CUDA_VISIBLE_DEVICES=$i wandb agent $2 &
        sleep 2 # add 2 second delay
    done
done

sleep 2d
