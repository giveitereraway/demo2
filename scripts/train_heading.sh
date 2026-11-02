#!/bin/sh

env="SingleControl"
scenario="1/heading"
algo="ppo"
exp="heading_attention_h16_d128"
seed=1
export WANDB_MODE="offline"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"

CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --memory-encoder temporal_attention \
    --history-length 16 --attention-hidden-size 128 --attention-heads 4 \
    --act-hidden-size "128 128" \
    --user-name "sf" --use-wandb --wandb-name "aircraft"