#! /usr/bin/env bash


### llama3-8b 增加额外数据，不进行增强  last

set -ex

NUM_GPUS=1


# deepspeed --include localhost:0 finetune.py\
CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS train_llama_instruction_stage2.py \
    --config ./config/train_llama_instruction_stage2.yaml \
