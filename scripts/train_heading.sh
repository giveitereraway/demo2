#!/bin/sh
# 2e7 快速检测试验（PPO+GRU vs SPO+GRU-Attention）：
#   NUM_ENV_STEPS=2e7 sh scripts/train_heading.sh clip gru mixed
#   NUM_ENV_STEPS=2e7 sh scripts/train_heading.sh spo gru_attention mixed
# 1e8 正式多随机种子消融（SEED=1,2,3）：
#   SEED=1 sh scripts/train_heading.sh clip gru mixed
#   SEED=1 sh scripts/train_heading.sh clip gru_attention mixed
#   SEED=1 sh scripts/train_heading.sh spo gru mixed
#   SEED=1 sh scripts/train_heading.sh spo gru_attention mixed

cd "$(dirname "$0")" || exit 1

loss="${1:-clip}"
encoder="${2:-gru_attention}"
observation="${3:-mixed}"
seed="${SEED:-${4:-1}}"
num_env_steps="${NUM_ENV_STEPS:-1e8}"
ppo_epoch="${PPO_EPOCH:-8}"
num_mini_batch="${NUM_MINI_BATCH:-8}"
history_length="${HISTORY_LENGTH:-16}"
exp="${EXP_NAME:-heading_${loss}_${encoder}}"

case "$encoder" in
    mlp)
        set -- --hidden-size "128 128"
        ;;
    gru|igru)
        set -- --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length 10
        ;;
    gru_attention)
        set -- --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4 \
            --data-chunk-length 10
        ;;
    temporal_attention)
        set -- --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4
        ;;
    *)
        echo "编码器须为 mlp、gru、igru、gru_attention 或 temporal_attention" >&2
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
    mixed)
        exp="${exp}_mixed"
        set -- "$@" --heading-mixed-degradation
        ;;
    *)
        echo "观测模式须为 clean、noisy 或 mixed" >&2
        exit 1
        ;;
esac

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train/train_jsbsim.py \
    --env-name SingleControl --scenario-name 1/heading \
    --algorithm-name ppo --experiment-name "$exp" \
    --policy-loss "$loss" --memory-encoder "$encoder" "$@" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 1 \
    --num-mini-batch "$num_mini_batch" --buffer-size 3000 --num-env-steps "$num_env_steps" \
    --lr 3e-4 --gamma 0.99 --ppo-epoch "$ppo_epoch" --clip-param 0.2 \
    --max-grad-norm 2 --entropy-coef 1e-3 \
    --act-hidden-size "128 128" \
    --use-eval --heading-first-target \
    --eval-interval 10 --eval-episodes 32 --n-eval-rollout-threads 4 \
    --heading-error-limits 10 200 \
    --heading-target-pass-rate 0.75 --heading-target-eval-seed 2026 \
    --user-name sf
