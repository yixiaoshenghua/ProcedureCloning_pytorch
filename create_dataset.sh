#!/bin/bash

for s in {0..5}; do
python dice_rl/scripts/create_dataset.py \
  --save_dir=maze_datasets \
  --env_name=maze:16-tunnel \
  --num_trajectory=4 \
  --max_trajectory_length=100 \
  --tabular_obs=0 \
  --alpha=1.0 \
  --seed=$s;
done
