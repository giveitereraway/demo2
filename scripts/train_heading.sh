#!/bin/sh
# 纯 PPO（单帧 MLP）：sh scripts/train_heading.sh clip mlp
# PPO + GRU：sh scripts/train_heading.sh clip gru
# PPO + 注意力：sh scripts/train_heading.sh clip temporal_attention
# SPO（单帧 MLP）：sh scripts/train_heading.sh spo mlp
# SPO + GRU：sh scripts/train_heading.sh spo gru
# SPO + 注意力：sh scripts/train_heading.sh spo temporal_attention
# 带噪训练：sh scripts/train_heading.sh clip gru noisy
# 带噪训练：sh scripts/train_heading.sh clip temporal_attention noisy

cd "$(dirname "$0")" || exit 1

loss="${1:-clip}"
encoder="${2:-temporal_attention}"
observation="${3:-clean}"
seed=1
exp="heading_${loss}_${encoder}"


case "$encoder" in
    mlp)
        set -- --hidden-size "128 128"
        ;;
    gru)
        set -- --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length 8
        ;;
    temporal_attention)
        set -- --history-length 16 --attention-hidden-size 128 --attention-heads 4
        ;;
    *)
        echo "编码器须为 mlp、gru 或 temporal_attention" >&2
        exit 1
        ;;
esac

case "$observation" in
    clean)
        ;;
    noisy)
        exp="${exp}_noisy"
        set -- "$@" --heading-measurement-noise
        ;;
    *)
        echo "观测模式须为 clean 或 noisy" >&2
        exit 1
        ;;
esac

CUDA_VISIBLE_DEVICES=0 python train/train_jsbsim.py \
    --env-name SingleControl --scenario-name 1/heading \
    --algorithm-name ppo --experiment-name "$exp" \
    --policy-loss "$loss" --memory-encoder "$encoder" "$@" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch 5 --buffer-size 3000 --num-env-steps 1e8 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --clip-param 0.2 \
    --max-grad-norm 2 --entropy-coef 1e-3 \
    --act-hidden-size "128 128" \
    --use-eval --heading-first-target \
    --eval-interval 10 --eval-episodes 32 --n-eval-rollout-threads 4 \
    --heading-error-limits 10 200 \
    --heading-target-pass-rate 0.75 --heading-target-eval-seed 2026 \
    --user-name sf
