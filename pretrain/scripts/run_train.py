
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import argparse

parser = argparse.ArgumentParser(description='CT-CLIP Training')

# 数据路径
parser.add_argument('--data_train', default='../CT-RATE/data/train')
parser.add_argument('--data_valid', default='../CT-RATE/data/val')
parser.add_argument('--nlst_path', default='/Astra/save/nlst.json')
parser.add_argument('--labels', default='...csv')
parser.add_argument('--reports_train', default='...csv')
parser.add_argument('--reports_valid', default='...csv')
parser.add_argument('--meta_train', default='../CT-RATE/ct_rate_meta/train_metadata.csv')
parser.add_argument('--meta_valid', default='../CT-RATE/ct_rate_meta/validation_metadata.csv')

# 训练参数
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--nlst_num', type=int, default=100)
parser.add_argument('--num_train_steps', type=int, default=100001)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--results_folder', default='../my_clip_result/temp')
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_path', default='../hf/BiomedVLP_cxr_bert')

args = parser.parse_args()

# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from CTCLIPTrainer import CTClipTrainer

print(f"🚀 Training with batch_size={args.batch_size}, steps={args.num_train_steps}, gpu={args.gpu}")

tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=True)
text_encoder = BertModel.from_pretrained(args.bert_path)

image_encoder = CTViT(
    dim=512, codebook_size=8192, image_size=480, patch_size=20,
    temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
    dim_head=32, heads=8
)

clip = CTCLIP(
    image_encoder=image_encoder, text_encoder=text_encoder,
    dim_text=768, dim_image=294912, dim_latent=512,
    extra_latent_projection=False, use_mlm=False,
    downsample_image_embeds=False, use_all_token_embeds=False
)

trainer = CTClipTrainer(
    clip,
    reports_file_train=args.reports_train,
    reports_file_valid=args.reports_valid,
    nlst_path=args.nlst_path,
    data_train=args.data_train,
    data_valid=args.data_valid,
    train_meta_file=args.meta_train,
    valid_meta_file=args.meta_valid,
    labels=args.labels,
    batch_size=args.batch_size,
    results_folder=args.results_folder,
    num_train_steps=args.num_train_steps,
    num_workers=args.num_workers,
    nlst_num=args.nlst_num,
)

trainer.train()
