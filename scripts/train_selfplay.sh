#!/bin/sh
# 纯 PPO（单帧 MLP）：sh train_selfplay.sh ppo-mlp
# PPO + GRU：	sh train_selfplay.sh ppo-gru
# PPO + IGRU：sh train_selfplay.sh ppo-igru
# PPO + 注意力：sh train_selfplay.sh ppo-attention
# SPO（单帧 MLP）：sh train_selfplay.sh spo-mlp
# SPO + GRU：sh train_selfplay.sh spo-gru
# SPO + 注意力：	sh train_selfplay.sh spo-attention

cd "$(dirname "$0")" || exit 1

env="SingleCombat"
scenario="1v1/NoWeapon/Selfplay"
algo="ppo"
variant="${1:-ppo-attention}"
observation="${2:-clean}"
seed="${SEED:-1}"
num_env_steps="${NUM_ENV_STEPS:-1e8}"

case "$variant" in
    ppo-mlp)
        policy_loss="clip"
        exp="1v1_ppo_mlp_d128_2e8"
        set -- --memory-encoder mlp --hidden-size "128 128"
        ;;
    ppo-gru|ppo-igru)
        policy_loss="clip"
        encoder="${variant#ppo-}"
        exp="1v1_ppo_${encoder}_d128_2e8"
        set -- --memory-encoder "$encoder" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length 10
        ;;
    ppo-attention)
        policy_loss="clip"
        exp="1v1_ppo_attention_h16_d128_2e8"
        set -- --memory-encoder temporal_attention \
            --history-length 16 --attention-hidden-size 128 --attention-heads 4
        ;;
    spo-mlp)
        policy_loss="spo"
        exp="1v1_spo_mlp_d128_2e8"
        set -- --memory-encoder mlp --hidden-size "128 128"
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
        echo "用法：$0 {ppo-mlp|ppo-gru|ppo-igru|ppo-attention|spo-mlp|spo-gru|spo-attention} [clean|mixed]" >&2
        exit 2
        ;;
esac

case "$observation" in
    clean)
        ;;
    mixed)
        exp="1v1_${variant}_ownship_mixed_${num_env_steps}"
        set -- "$@" --selfplay-ownship-degradation
        ;;
    *)
        echo "观测模式须为 clean 或 mixed" >&2
        exit 2
        ;;
esac

echo "env=${env}, scenario=${scenario}, algo=${algo}, policy_loss=${policy_loss}, exp=${exp}, seed=${seed}, obs=${observation}"

CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name "$env" --algorithm-name "$algo" --scenario-name "$scenario" --experiment-name "$exp" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 10 \
    --use-selfplay --selfplay-algorithm fsp --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 8 --eval-interval 10 --eval-episodes 32 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps "$num_env_steps" \
    --policy-loss "$policy_loss" \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    "$@" \
    --act-hidden-size "128 128" \
    --user-name "sf"
