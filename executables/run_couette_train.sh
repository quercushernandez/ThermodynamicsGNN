
python main.py --sys_name couette --train True \
    --n_hidden 2 --dim_hidden 10 --passes 10 \
    --miles 2000 4000 --gamma 0.1 --max_epoch 6000 \
    --noise_var 1e-2 --lr 1e-3 --lambda_d 100 \
    --gpu False
