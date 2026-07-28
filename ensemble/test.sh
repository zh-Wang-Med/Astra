export CUDA_VISIBLE_DEVICES=1

savepath="../save/pretrain_related/merlin/v0_nlst_1234_full_2e6/ee"
if [ ! -d "$savepath" ]; then
  mkdir -p "$savepath"
  echo "Folder '$savepath' created."
else
  echo "Folder '$savepath' already exists."
fi
python ../train.py \
    --test \
    --savedmodel_path $savepath \
    --trainroad '...csv' \
    --valroad '...csv' \
    --testroad '...csv' \
    --predictroad '...csv' \
    --learning_rate 2e-5 \
    --devices 1 \
    --strategy "auto" \
    --batch_size 16 \
    --num_workers 8 \
    --accumulate_grad_batches 1 \
    --text_help False \
    --linear_number 1 \
    --vision_encoder "ct_clip" \
    --visual_frozen True \
    --dataset "merlin" \
    --train_num 0 \
    --ct_clip_pre "v3" \
    --load_pretrain True \
    --pretrain_path "." \
    --delta_file "." \
    2>&1 | tee -a "../save/pretrain_related/merlin/v0_nlst_1234_full_2e6/ee/log.txt"