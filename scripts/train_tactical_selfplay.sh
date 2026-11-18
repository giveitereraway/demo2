#!/bin/sh
# 第四章 1v1 战术分层 2x2 消融启动示例（SEED=1,2,3）：
#   SEED=1 sh scripts/train_tactical_selfplay.sh ppo-gru radar
#   SEED=1 sh scripts/train_tactical_selfplay.sh ppo-gru-attention radar
#   SEED=1 sh scripts/train_tactical_selfplay.sh spo-gru radar
#   SEED=1 sh scripts/train_tactical_selfplay.sh spo-gru-attention radar

cd "$(dirname "$0")" || exit 1

variant="${1:-spo-gru-attention}"
observation="${2:-radar}"
seed="${SEED:-${3:-1}}"
num_env_steps="${NUM_ENV_STEPS:-1e8}"
ppo_epoch="${PPO_EPOCH:-8}"
num_mini_batch="${NUM_MINI_BATCH:-8}"
selfplay_algo="${SELFPLAY_ALGO:-fsp}"
history_length="${HISTORY_LENGTH:-16}"

env="SingleCombat"
scenario="1v1/NoWeapon/TacticalHierarchySelfplay"
algo="ppo"

case "$variant" in
    ppo-gru)
        policy_loss="clip"
        encoder="gru"
        set -- --memory-encoder gru --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10
        ;;
    ppo-igru)
        policy_loss="clip"
        encoder="igru"
        set -- --memory-encoder igru --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10
        ;;
    ppo-gru-attention)
        policy_loss="clip"
        encoder="gru_attention"
        set -- --memory-encoder gru_attention --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4 \
            --data-chunk-length 10
        ;;
    spo-gru)
        policy_loss="spo"
        encoder="gru"
        set -- --memory-encoder gru --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10
        ;;
    spo-igru)
        policy_loss="spo"
        encoder="igru"
        set -- --memory-encoder igru --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 10
        ;;
    spo-gru-attention)
        policy_loss="spo"
        encoder="gru_attention"
        set -- --memory-encoder gru_attention --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4 \
            --data-chunk-length 10
        ;;
    *)
        echo "用法：$0 {ppo-gru|ppo-igru|ppo-gru-attention|spo-gru|spo-igru|spo-gru-attention} [radar|clean] [seed]" >&2
        exit 2
        ;;
esac

exp="${EXP_NAME:-1v1_tactical_${policy_loss}_${encoder}}"
case "$observation" in
    clean)
        ;;
    radar)
        exp="${exp}_radar"
        set -- "$@" --combat-radar-degradation
        ;;
    *)
        echo "观测模式须为 radar 或 clean" >&2
        exit 2
        ;;
esac

echo "env=${env}, scenario=${scenario}, algo=${algo}, policy_loss=${policy_loss}, encoder=${encoder}, obs=${observation}, exp=${exp}, seed=${seed}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train/train_jsbsim.py \
    --env-name "$env" --algorithm-name "$algo" --scenario-name "$scenario" --experiment-name "$exp" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 10 \
    --use-eval --n-eval-rollout-threads 4 --eval-interval 10 --eval-episodes 16 \
    --use-selfplay --selfplay-algorithm "$selfplay_algo" --n-choose-opponents 1 \
    --num-mini-batch "$num_mini_batch" --buffer-size 3000 --num-env-steps "$num_env_steps" \
    --policy-loss "$policy_loss" \
    --lr 3e-4 --gamma 0.99 --ppo-epoch "$ppo_epoch" --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    "$@" \
    --act-hidden-size "128 128" \
    --user-name "sf"