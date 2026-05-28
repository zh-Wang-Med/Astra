export CUDA_VISIBLE_DEVICES=0

savepath="../save/ctrg_related/temp"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
python ../train.py \
    --savedmodel_path $savepath \
    --trainroad "...csv" \
    --valroad "...csv" \
    --testroad "...csv" \
    --predictroad "...csv" \
    --learning_rate 2e-5 \
    --devices 1 \
    --num_workers 4 \
    --strategy "auto" \
    --batch_size 8 \
    --accumulate_grad_batches 2 \
    --max_epochs 10 \
    --text_help False \
    --linear_number 1 \
    --vision_encoder "ct_clip" \
    --visual_frozen False \
    --dataset "" \
    --train_num 0 \
    --ct_clip_pre "v3" \
    --load_pretrain True \
    --pretrain_path "../my_clip_result/myctrate_dataset_nlst123456/CTClip.80000.pt" \
    2>&1 | tee -a "../save/ctrg_related/temp/log.txt"