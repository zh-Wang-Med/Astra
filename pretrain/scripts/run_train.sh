accelerate launch --use_fsdp ../scripts/run_train.py \
    --data_train ../Astra/save/.json \
    --data_valid ../Astra/save/.json \
    --labels ../Astra/save/merge/.csv \
    --results_folder ../my_clip_result/repo