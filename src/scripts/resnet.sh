#!/bin/bash

exp_name=FedAvg_CT
local_bs=128
local_ep=5
multiRun=no
model=resnet18
epochs=5000
exp_dir=../experiments/

cuda_visible_devices=(2 7)
norm_types=(bn ln)

mkdir -p $exp_dir$exp_name/

if [ $multiRun == 'no' ];
then
    echo $multiRun
    for ((i = 0; i < 1; i++))
    do
        CUDA_VISIBLE_DEVICES=${cuda_visible_devices[i]} python main.py --num_users 1 --local_ep 1 --dataset cifar10 --iid 0 --gpu 2 --lr 0.01 --local_bs 128 --exp_dir $exp_dir --exp_name $exp_name --norm ${norm_types[i]} --model $model --epochs $epochs --seed 7505 & sleep 60
    done
else
  for seed in 7505 3349 1055 9247 2716
  do
    echo $seed
    python main.py --dataset cifar --iid 0 --gpu 1 --lr 0.1 --local_bs 128 --exp_dir $exp_dir --exp_name $exp_name --norm $norm --model $model --epochs $epochs --seed $seed
  done
fi