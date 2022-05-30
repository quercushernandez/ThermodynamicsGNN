#!/bin/bash
python main.py --sys_name cylinder --train True \
    --n_hidden 2 --dim_hidden 128 --passes 10 \
    --miles 500 1500 --gamma 0.1 --max_epoch 2000 \
    --noise_var 4e-4 --lr 1e-4 --lambda_d 10
