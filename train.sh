#!/bin/zsh
conda activate epymarl
python src/main.py --config=qmix --env-config=gymma with env_args.time_limit=40 env_args.key="mpe:Coverage-v0"