#!/bin/bash
#!/bin/bash



PATH_PRE="./"

TRAIN_DATA=/home/chenning/code/Eedi/recall/datas/emb_v1/finetune_train_cot.jsonl
MODEL_USE="nv_round1"
ZERO_STAGE=1
OUTPUT=${PATH_PRE}../recall_model_save/emb_v1/${MODEL_USE}_qlora_rerun_v1


#模型地址
MODEL_PATH=/home/chenning/opensource_models/nvidia/NV-Embed-v2
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:2,3,4,5 simcse_deepspeed_nv_qlora.py \
       --project_name ${name}_${MODEL_USE} \
       --train_data ${TRAIN_DATA} \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 4 \
       --train_group_size 4 \
       --gradient_accumulation_steps 8 \
       --query_max_len 1000 \
       --passage_max_len 500 \
       --earystop 0 \
       --save_batch_steps 100000000000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 20  \
       --learning_rate 1e-4 \
       --num_warmup_steps 100 \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing