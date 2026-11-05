#!/bin/sh
# PPO + GRU：	sh train_selfplay.sh ppo-gru
# PPO + 注意力：sh train_selfplay.sh ppo-attention
# SPO + GRU：sh train_selfplay.sh spo-gru
# SPO + 注意力：	sh train_selfplay.sh spo-attention

env="SingleCombat"
scenario="1v1/NoWeapon/Selfplay"
algo="ppo"
seed=1
export WANDB_MODE="offline"

case "${1:-ppo-attention}" in
    ppo-gru)
        policy_loss="clip"
        exp="1v1_ppo_gru_d128_2e8"
        set -- --memory-encoder gru \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length 10
        ;;
    ppo-attention)
        policy_loss="clip"
        exp="1v1_ppo_attention_h16_d128_2e8"
        set -- --memory-encoder temporal_attention \
            --history-length 16 --attention-hidden-size 128 --attention-heads 4
        ;;
    spo-gru)
        policy_loss="spo"
        exp="1v1_spo_gru_d128_2e8"
        set -- --memory-encoder gru \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length 10
        ;;
    spo-attention)
        policy_loss="spo"
        exp="1v1_spo_attention_h16_d128_2e8"
        set -- --memory-encoder temporal_attention \
            --history-length 16 --attention-hidden-size 128 --attention-heads 4
        ;;
    *)
        echo "用法：$0 {ppo-gru|ppo-attention|spo-gru|spo-attention}" >&2
        exit 2
        ;;
esac

echo "env=${env}, scenario=${scenario}, algo=${algo}, policy_loss=${policy_loss}, exp=${exp}, seed=${seed}, wandb=${WANDB_MODE}"

CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name "$env" --algorithm-name "$algo" --scenario-name "$scenario" --experiment-name "$exp" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1 \
    --use-selfplay --selfplay-algorithm fsp --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 10 --eval-episodes 10 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 2e8 \
    --policy-loss "$policy_loss" \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    "$@" \
    --act-hidden-size "128 128" \
    --user-name "sf" --use-wandb --wandb-name "aircraft"