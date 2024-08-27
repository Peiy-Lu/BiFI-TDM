#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 2 \
    --val_epoch 20 \
    --epoch_size 100 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 20 \
    --train_shot 5 \
    --train_query_shot 15 \
    --train_transform_type 0 \
    --test_way 5 \
    --test_shot 1 5 \
    --pre \
    --num_class 100 \
    --gamma_loss 0.25 \
    --noise \
    --noise_value 0.5 \
    --gpu 2 \