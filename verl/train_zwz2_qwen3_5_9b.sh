#!/usr/bin/env bash
# GRPO training launchers for ZwZ-v2: single-stage GRPO RL on synthesized
# multi-region MCQ perception data (see data_synthesis/). All paths are
# supplied via environment variables; MODEL_PATH / TRAIN_FILE / TEST_FILE are
# required. Run from this directory after `pip install -e .`.

set -xeuo pipefail

########################### user-adjustable ###########################
DEVICE=${DEVICE:-$(python3 -c 'import torch_npu' 2>/dev/null && echo npu || echo gpu)}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-ZWZ2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-zwz2_qwen3_5_9b}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-16}
NNODES=${NNODES:-4}

GEN_TP=${GEN_TP:-8}
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-16}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.4}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH to the base VLM checkpoint directory (e.g. Qwen3.5-9B)}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:-"./outputs/${EXPERIMENT_NAME}"}
RUNS_ROOT=${RUNS_ROOT:-"${EXPERIMENT_ROOT}"}
CKPTS_DIR=${CKPTS_DIR:-"${EXPERIMENT_ROOT}/checkpoints"}
LOGS_DIR=${LOGS_DIR:-"${RUNS_ROOT}/logs"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"${RUNS_ROOT}/rollouts"}
VAL_ROLLOUT_DIR=${VAL_ROLLOUT_DIR:-"${RUNS_ROOT}/rollouts_val"}
TRAIN_FILE=${TRAIN_FILE:?Set TRAIN_FILE to the training parquet exported by data_synthesis (05_export_dataset.py)}
TEST_FILE=${TEST_FILE:?Set TEST_FILE to the validation parquet}
REWARD_FUNCTION=${REWARD_FUNCTION:-"./examples/reward_function/perception_rule.py:compute_score"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-8}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1900}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-10000}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-10000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
SAVE_LIMIT=${SAVE_LIMIT:-100}
########################### end user-adjustable ###########################

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

n_devices_per_node=${NDEVICES_PER_NODE}
fsdp_size=${FSDP_SIZE}

case "${DEVICE}" in
    gpu)
        ;;
    npu)
        export HCCL_CONNECT_TIMEOUT=1500
        export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
        export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
        export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
        n_devices_per_node=16
        ;;
    *)
        echo "Unsupported DEVICE=${DEVICE}. Expected 'gpu' or 'npu'." >&2
        exit 1
        ;;
esac

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p \
    "${LOGS_DIR}" \
    "${CKPTS_DIR}" \
    "${ROLLOUT_DIR}" \
    "${VAL_ROLLOUT_DIR}"

reward_path="${REWARD_FUNCTION%%:*}"
reward_name="${REWARD_FUNCTION##*:}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${reward_path}" in
    /*) ;;
    *)  reward_path="${SCRIPT_DIR}/${reward_path#./}" ;;
esac

CUSTOM_CHAT_TEMPLATE_FILE=${CUSTOM_CHAT_TEMPLATE_FILE:-"${SCRIPT_DIR}/chat_templates/perception.jinja"}
export VERL_CUSTOM_CHAT_TEMPLATE="$(cat "${CUSTOM_CHAT_TEMPLATE_FILE}")"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.gen_batch_size=${GEN_BATCH_SIZE}
    data.val_batch_size=${VAL_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=False
    data.truncation=error
    data.image_key=images
    data.shuffle=True
    +data.mm_processor_kwargs.max_pixels=4194304
    +data.mm_processor_kwargs.min_pixels=65536
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config._attn_implementation=flash_attention_3
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_3
    'actor_rollout_ref.model.custom_chat_template=${oc.env:VERL_CUSTOM_CHAT_TEMPLATE}'
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp
    actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=False
    actor_rollout_ref.actor.fsdp_config.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.clip_ratio_low=0.2
    actor_rollout_ref.actor.clip_ratio_high=0.3
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=False
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.strategy=fsdp
    actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.entropy_checkpointing=False
    actor_rollout_ref.ref.fsdp_config.entropy_from_logits_with_chunking=False
    actor_rollout_ref.ref.fsdp_config.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=512
)

REWARD=(
    reward.custom_reward_function.path="${reward_path}"
    reward.custom_reward_function.name="${reward_name}"
)

TRAINER=(
    trainer.use_v1=False
    transfer_queue.enable=False
    trainer.critic_warmup=0
    trainer.logger=['console']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.resume_from_path=null
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.max_actor_ckpt_to_keep=${SAVE_LIMIT}
    trainer.max_critic_ckpt_to_keep=${SAVE_LIMIT}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=null
    trainer.rollout_data_dir=null
    trainer.validation_data_dir="${VAL_ROLLOUT_DIR}"
)

########################### launch ###########################
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Checkpoint dir: ${CKPTS_DIR}"
echo "Resume mode: auto"
echo "Nodes: ${NNODES}, devices per node: ${n_devices_per_node}"

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOGS_DIR}/${EXPERIMENT_NAME}-${start_time}.log"
