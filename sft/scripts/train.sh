#!/bin/bash
dataset="merge"
export CUDA_VISIBLE_DEVICES=2,3
annotation="...json"
version="Astra"
visual_encoder_name="merlin"
savepath="./save/$dataset/$version"

if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
train.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --vicuna_model ${vicuna_model} \
    --batch_size 4 \
    --val_batch_size 4 \
    --freeze_vm False \
    --savedmodel_path ${savepath} \
    --max_length 1000 \
    --min_new_tokens 1 \
    --max_new_tokens 1000 \
    --repetition_penalty 2.0 \
    --length_penalty 2.0 \
    --num_workers 8 \
    --devices 2 \
    --max_epochs 10 \
    --limit_val_batches 0.1 \
    --val_check_interval 1 \
    --num_sanity_val_steps 2 \
    --llm_use_lora False \
    --llm_r 32 \
    --llm_alpha 64 \
    --lora_dropout 0.1 \
    --accumulate_grad_batches 2 \
    --learning_rate 3e-5 \
    --test_batch_size 8 \
    --visual_encoder_name ${visual_encoder_name} \
    --perceiver_whether_inital True \
    --whether_perceiver False \
    --perceiver_heads 8 \
    --perceiver_dim_head 256 \
    --vision_token_number 32 \
    --vision_dim 2048 \
    # --delta_file ${delta_file} \
    2>&1 | tee -a ${savepath}/log.txt
