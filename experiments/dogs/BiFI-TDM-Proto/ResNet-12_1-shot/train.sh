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
    --train_way 10 \
    --train_shot 1 \
    --train_transform_type 0 \
    --test_transform_type 0 \
    --test_shot 1 \
    --resnet \
    --num_class 60 \
    --gamma_loss 0.5 \
    --noise \
    --gpu 0
