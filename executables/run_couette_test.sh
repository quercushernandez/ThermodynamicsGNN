#!/bin/bash
python main.py --sys_name couette --train False \
    --n_hidden 2 --dim_hidden 10 --passes 10 \
    --gpu False
