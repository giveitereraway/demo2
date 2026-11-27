#!/bin/sh
# 1v1 战术分层自博弈训练脚本（核心训练与评估超参已与 train_selfplay.sh 对齐，便于 1:1 对比）
# 纯 PPO（单帧 MLP）：   sh scripts/train_tactical_selfplay.sh ppo-mlp [clean|mixed|radar|joint] [seed]
# PPO + GRU：            sh scripts/train_tactical_selfplay.sh ppo-gru [clean|mixed|radar|joint] [seed]
# PPO + IGRU：           sh scripts/train_tactical_selfplay.sh ppo-igru [clean|mixed|radar|joint] [seed]
# PPO + 注意力：         sh scripts/train_tactical_selfplay.sh ppo-attention [clean|mixed|radar|joint] [seed]
# PPO + GRU + 注意力：   sh scripts/train_tactical_selfplay.sh ppo-gru-attention [clean|mixed|radar|joint] [seed]
# SPO（单帧 MLP）：      sh scripts/train_tactical_selfplay.sh spo-mlp [clean|mixed|radar|joint] [seed]
# SPO + GRU：            sh scripts/train_tactical_selfplay.sh spo-gru [clean|mixed|radar|joint] [seed]
# SPO + IGRU：           sh scripts/train_tactical_selfplay.sh spo-igru [clean|mixed|radar|joint] [seed]
# SPO + 注意力：         sh scripts/train_tactical_selfplay.sh spo-attention [clean|mixed|radar|joint] [seed]
# SPO + GRU + 注意力：   sh scripts/train_tactical_selfplay.sh spo-gru-attention [clean|mixed|radar|joint] [seed]
#
# 说明：
# - clean 无退化；mixed 退化高层本机观测；radar 退化雷达航迹；joint 对两者使用同一退化档位。
# - 支持环境变量覆盖：SEED、NUM_ENV_STEPS、PPO_EPOCH、NUM_MINI_BATCH、SELFPLAY_ALGO、
#   N_CHOOSE_OPPONENTS、N_EVAL_ROLLOUT_THREADS、EVAL_EPISODES、HISTORY_LENGTH、EXP_NAME。

cd "$(dirname "$0")" || exit 1

variant="${1:-spo-igru}"
observation="${2:-clean}"
seed="${SEED:-${3:-1}}"
num_env_steps="${NUM_ENV_STEPS:-1e8}"
ppo_epoch="${PPO_EPOCH:-4}"
num_mini_batch="${NUM_MINI_BATCH:-5}"
selfplay_algo="${SELFPLAY_ALGO:-fsp}"
n_choose_opponents="${N_CHOOSE_OPPONENTS:-4}"
n_eval_rollout_threads="${N_EVAL_ROLLOUT_THREADS:-8}"
eval_episodes="${EVAL_EPISODES:-32}"
history_length="${HISTORY_LENGTH:-16}"

env="SingleCombat"
scenario="1v1/NoWeapon/TacticalHierarchySelfplay"
algo="ppo"

case "$variant" in
    ppo-mlp)
        policy_loss="clip"
        encoder="mlp"
        set -- --memory-encoder mlp --hidden-size "128 128"
        ;;
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
    ppo-attention)
        policy_loss="clip"
        encoder="temporal_attention"
        set -- --memory-encoder temporal_attention \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4
        ;;
    ppo-gru-attention)
        policy_loss="clip"
        encoder="gru_attention"
        set -- --memory-encoder gru_attention --hidden-size "128 128" \
            --recurrent-hidden-size 128 --recurrent-hidden-layers 1 \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4 \
            --data-chunk-length 10
        ;;
    spo-mlp)
        policy_loss="spo"
        encoder="mlp"
        set -- --memory-encoder mlp --hidden-size "128 128"
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
    spo-attention)
        policy_loss="spo"
        encoder="temporal_attention"
        set -- --memory-encoder temporal_attention \
            --history-length "$history_length" --attention-hidden-size 128 --attention-heads 4
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
        echo "用法：$0 {ppo-mlp|ppo-gru|ppo-igru|ppo-attention|ppo-gru-attention|spo-mlp|spo-gru|spo-igru|spo-attention|spo-gru-attention} [clean|mixed|radar|joint] [seed]" >&2
        exit 2
        ;;
esac

case "$observation" in
    clean)
        obs_tag="clean"
        ;;
    mixed)
        obs_tag="ownship_mixed"
        set -- "$@" --selfplay-ownship-degradation
        ;;
    radar)
        # 每局双方使用相同的退化档位，独立采样观测噪声和丢帧。
        obs_tag="radar"
        set -- "$@" --combat-radar-degradation
        ;;
    joint)
        obs_tag="joint"
        set -- "$@" --selfplay-joint-degradation
        ;;
    *)
        echo "观测模式须为 clean、mixed、radar 或 joint" >&2
        exit 2
        ;;
esac

exp="${EXP_NAME:-1v1_tactical_${variant}_${obs_tag}_${num_env_steps}}"

echo "env=${env}, scenario=${scenario}, algo=${algo}, policy_loss=${policy_loss}, encoder=${encoder}, obs=${observation}, exp=${exp}, seed=${seed}, opponents=${n_choose_opponents}, ppo_epoch=${ppo_epoch}, num_mini_batch=${num_mini_batch}, eval=${n_eval_rollout_threads}x${eval_episodes}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train/train_jsbsim.py \
    --env-name "$env" --algorithm-name "$algo" --scenario-name "$scenario" --experiment-name "$exp" \
    --seed "$seed" --n-training-threads 1 --n-rollout-threads 32 --cuda \
    --log-interval 1 --save-interval 10 \
    --use-selfplay --selfplay-algorithm "$selfplay_algo" --n-choose-opponents "$n_choose_opponents" \
    --use-eval --n-eval-rollout-threads "$n_eval_rollout_threads" --eval-interval 10 --eval-episodes "$eval_episodes" \
    --num-mini-batch "$num_mini_batch" --buffer-size 3000 --num-env-steps "$num_env_steps" \
    --policy-loss "$policy_loss" \
    --lr 3e-4 --gamma 0.99 --ppo-epoch "$ppo_epoch" --clip-param 0.2 --max-grad-norm 2 --entropy-coef 1e-3 \
    "$@" \
    --act-hidden-size "128 128" \
    --user-name "sf"
