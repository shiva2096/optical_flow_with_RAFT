#!/bin/bash
mkdir -p checkpoints
python3 -u train.py --name raft-sintel --stage sintel --validation sintel --gpus 0 --num_steps 25000 --batch_size 5 --lr 0.0001 --image_size 368 768 --wdecay 0.00001 --gamma=0.85
