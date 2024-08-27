#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 3 \
    --val_epoch 20 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 5 \
    --train_shot 5 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 5 \
    --resnet \
    --num_class 51 \
    --gamma_loss 0.5 \
    --noise \
    --gpu 0
