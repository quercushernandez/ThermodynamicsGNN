#!/bin/bash
python main.py --sys_name beam --train False \
    --n_hidden 2 --dim_hidden 50 --passes 10 \
    --gpu False
