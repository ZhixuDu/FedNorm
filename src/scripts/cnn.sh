#!/bin/bash

exp_name=FedAvg1
norm=$1
multiRun=yes
model=cnn
epochs=5000
exp_dir=../experiments/

mkdir -p $exp_dir$exp_name/

if [ $multiRun == 'no' ];
then
  echo $multiRun
  python main.py --dataset cifar --iid 0  --gpu 1 --lr 0.01 --local_bs 32 --exp_dir $exp_dir --exp_name $exp_name --norm $norm --model $model --epochs $epochs
else
  for seed in 7505 3349 1055 9247 2716
  do
    echo $seed
    python main.py --dataset cifar --iid 0 --gpu 1 --lr 0.01 --local_bs 32 --exp_dir $exp_dir --exp_name $exp_name --norm $norm --model $model --epochs $epochs --seed $seed
  done
fi
