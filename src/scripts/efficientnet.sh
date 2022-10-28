#!/bin/bash

exp_name=$1
norm=$2
local_bs=128
local_ep=5
multiRun=no
model=efficientnet_b0
epochs=5000
exp_dir=../experiments/

mkdir -p $exp_dir$exp_name/

if [ $multiRun == 'no' ];
then
  echo $multiRun
  python main.py --dataset cifar --iid 1 --gpu 1 --lr 0.01 --local_bs $local_bs --exp_dir $exp_dir --exp_name $exp_name --norm $norm --model $model --epochs $epochs --local_ep $local_ep
else
  for seed in 7505 3349 1055 9247 2716
  do
    echo $seed
    python main.py --dataset cifar --iid 0 --gpu 1 --lr 0.01 --local_bs 128 --exp_dir $exp_dir --exp_name $exp_name --norm $norm --model $model --epochs $epochs --seed $seed
  done
fi