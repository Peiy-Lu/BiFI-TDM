#!/bin/bash
python train.py \
    --opt sgd \
    --lr 1e-1 \
    --gamma 1e-1 \
    --epoch 400 \
    --stage 3 \
    --val_epoch 20 \
    --epoch_size 100 \
    --weight_decay 5e-4 \
    --nesterov \
    --train_way 10 \
    --train_shot 5 \
    --train_query_shot 15 \
    --train_transform_type 0 \
    --test_way 5 \
    --test_shot 1 5 \
    --pre \
    --resnet \
    --num_class 100 \
    --gamma_loss 0.25 \
    --noise \
    --cL_scale 0.1 \
    --gpu 3