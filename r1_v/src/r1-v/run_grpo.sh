cd /astra_open/r1_v

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0,1


torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    src/open_r1/grpo.py \
    --output_dir ../temp \
    --model_name_or_path ../hf/qwen25_vl \
    --dataset_name ../clevr_cogen_a_train \
    --deepspeed ../r1-v/local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 2 \
    --beta 0.0004 \
    --forte_path ../FORTE-main/data/FORTE_abdomen.json \
    --f1_cal micro \
    --whether_npz True \
    --prompt_style 'lung' \
    --reward_function "base" \
    --annotation ...json \