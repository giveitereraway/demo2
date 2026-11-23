#!/bin/sh
# 约 2e7 环境步快速检测试验：
#   NUM_ENV_STEPS=2e7 sh scripts/train_heading.sh clip gru mixed
#   NUM_ENV_STEPS=2e7 sh scripts/train_heading.sh spo gru_attention mixed
# 正式多随机种子实验（默认 99,936,000 环境步，SEED=1,2,3）：
#   SEED=1 sh scripts/train_heading.sh clip mlp mixed
#   SEED=1 sh scripts/train_heading.sh clip gru mixed
#   SEED=1 sh scripts/train_heading.sh clip lstm mixed
#   SEED=1 sh scripts/train_heading.sh clip igru mixed
#   SEED=1 sh scripts/train_heading.sh spo mlp mixed
#   SEED=1 sh scripts/train_heading.sh sac mlp mixed
#   SEED=1 sh scripts/train_heading.sh bdq mlp mixed
#   SEED=1 sh scripts/train_heading.sh clip gru_attention mixed
#   SEED=1 sh scripts/train_heading.sh spo gru mixed
#   SEED=1 sh scripts/train_heading.sh spo igru mixed
#   SEED=1 sh scripts/train_heading.sh spo gru_attention mixed

cd "$(dirname "$0")" || exit 1

method="${1:-clip}"
observation="${3:-mixed}"
seed="${SEED:-${4:-1}}"
num_env_steps="${NUM_ENV_STEPS:-99936000}"

case "$method" in
    clip|spo)
        algorithm=ppo
        encoder="${2:-gru_attention}"
        set -- --policy-loss "$method" --ppo-epoch "${PPO_EPOCH:-8}" \
            --num-mini-batch "${NUM_MINI_BATCH:-8}" --clip-param 0.2 --entropy-coef 1e-3
        ;;
    sac|bdq)
        algorithm="$method"
        encoder="${2:-mlp}"
        if [ "$encoder" != mlp ]; then
            echo "$algorithm 的 Heading 基线使用 mlp 编码器" >&2
            exit 1
        fi
        set -- --offpolicy-replay-size "${OFFPOLICY_REPLAY_SIZE:-500000}" \
            --offpolicy-batch-size "${OFFPOLICY_BATCH_SIZE:-256}" \
            --offpolicy-warmup-steps "${OFFPOLICY_WARMUP_STEPS:-10000}" \
            --offpolicy-train-freq "${OFFPOLICY_TRAIN_FREQ:-32}"
        ;;
    *)
        echo "算法须为 clip、spo、sac 或 bdq" >&2
        exit 1
        ;;
esac
exp="${EXP_NAME:-heading_${method}_${encoder}}"

case "$encoder" in
    mlp)
        set -- "$@" --hidden-size "128 128"
        ;;
    gru|lstm|igru)
        set -- "$@" --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --data-chunk-length "${DATA_CHUNK_LENGTH:-10}"
        ;;
    gru_attention)
        set -- "$@" --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --history-length "${HISTORY_LENGTH:-16}" --attention-hidden-size 128 --attention-heads 4 \
            --data-chunk-length "${DATA_CHUNK_LENGTH:-10}"
        ;;
    temporal_attention)
        set -- "$@" --history-length "${HISTORY_LENGTH:-16}" --attention-hidden-size 128 --attention-heads 4
        ;;
    *)
        echo "编码器须为 mlp、gru、lstm、igru、gru_attention 或 temporal_attention" >&2
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

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON:-python}" train/train_jsbsim.py \
    --env-name SingleControl --scenario-name 1/heading \
    --algorithm-name "$algorithm" --experiment-name "$exp" --memory-encoder "$encoder" "$@" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads "${N_ROLLOUT_THREADS:-32}" --cuda \
    --log-interval 1 --save-interval 1 \
    --buffer-size "${BUFFER_SIZE:-3000}" --num-env-steps "$num_env_steps" \
    --lr "${LR:-3e-4}" --gamma 0.99 --max-grad-norm 2 \
    --act-hidden-size "128 128" \
    --use-eval --heading-first-target \
    --eval-interval "${EVAL_INTERVAL:-10}" --eval-episodes "${EVAL_EPISODES:-32}" \
    --n-eval-rollout-threads "${N_EVAL_ROLLOUT_THREADS:-4}" \
    --heading-error-limits 10 200 \
    --heading-target-pass-rate 0.75 --heading-target-eval-seed 2026 \
    --user-name sf
