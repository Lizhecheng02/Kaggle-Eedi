#!/bin/bash


# model_name_or_path=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-dpsr/model_path/Mistral-7B-v0.3
model_name_or_path=/home/xuanming/pre-trained-models/LLM-Research/Meta-Llama-3___1-8B-Instruct
DATA_NAME="recall_top_100_for_rank"
DATA_DIR=/home/xuanming/kaggle/Eedi/rerank/NV-EMB-V2/
MODEL_USE='v0_round0_qlora_recall_top_100_for_rank_model'
OUTPUT=/home/xuanming/kaggle/Eedi/rerank/NV-EMB-V2/model_save/${MODEL_USE}

# export CUDA_VISIBLE_DEVICES=5,6,7
torchrun --nproc_per_node 1 \
-m run \
--output_dir ${OUTPUT} \
--model_name_or_path ${model_name_or_path} \
--train_data ${DATA_DIR}${DATA_NAME}.jsonl \
--learning_rate 2e-4 \
--num_train_epochs 3 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 8 \
--dataloader_drop_last True \
--query_max_len 512 \
--passage_max_len 512 \
--train_group_size 16 \
--logging_steps 1 \
--save_strategy epoch \
--save_steps 1 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage2.json \
--report_to "none" \
--warmup_ratio 0.05 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj lm_head