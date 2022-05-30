#!/bin/bash
python main.py --sys_name beam --train True \
    --n_hidden 2 --dim_hidden 50 --passes 10 \
    --miles 600 1200 --gamma 0.1 --max_epoch 1800 \
    --noise_var 1e-5 --lr 1e-4 --lambda_d 50
