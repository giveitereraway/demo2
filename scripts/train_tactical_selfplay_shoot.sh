#!/bin/sh

env="SingleCombat"
scenario="1v1/ShootMissile/Selfplay"
algo="ppo"
exp="1v1_shoot_selfplay_2e8"
seed=1
export WANDB_MODE="offline"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, seed is ${seed}"
CUDA_VISIBLE_DEVICES=1 python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 32 --cuda --log-interval 1 --save-interval 1 \
    --use-selfplay --selfplay-algorithm "fsp" --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 5 --eval-interval 10 --eval-episodes 5 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 2e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size "128 128" --act-hidden-size "128 128" --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --user-name "sf" --use-wandb --wandb-name "aircraft" \
    --use-prior
