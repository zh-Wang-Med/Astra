"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from copy import deepcopy
import re
import torch
import torch.distributed as dist

from lavis.common.registry import registry
from lavis.common.dist_utils import is_dist_avail_and_initialized
from lavis.models.blip_models import tie_encoder_decoder_weights
from lavis.models.blip_models.blip import BlipBase
from lavis.models.blip_models.blip_outputs import (
    BlipOutput,
)
from lavis.models.base_model import (
    MomentumDistilationMixin,
    SharedQueueMixin,
    all_gather_with_grad,
    concat_all_gather
)
from lavis.models.med import XBertEncoder
from torch import nn
import random
import os
import json
import numpy as np
import torch.nn.functional as F

@torch.no_grad()
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]
    data_list = [None] * world_size
    dist.all_gather_object(data_list, data)
    return data_list

@registry.register_model("blip_pretrain_vit")
class BlipPretrain(BlipBase, SharedQueueMixin, MomentumDistilationMixin):
    """
    BLIP pretrain model.

    Supported model types:
        - base: BLIP base model before pretraining.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/blip_pretrain_base.yaml",
        "fvlp": "configs/models/blip_pretrain_ct.yaml",
    }

    def __init__(
        self,
        image_encoder,
        text_encoder,
        text_decoder,
        alpha=0.4,
        embed_dim=256,
        tie_enc_dec_weights=True,
        max_txt_len=175
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        text_encoder.resize_token_embeddings(len(self.tokenizer))

        if tie_enc_dec_weights:
            tie_encoder_decoder_weights(
                encoder=text_encoder,
                decoder=text_decoder.bert,
                base_model_prefix="",
                skip_key="/attention",
            )

        self.visual_encoder = image_encoder

        self.text_encoder = text_encoder

        # creating projection layers for ITC
        text_width = text_encoder.config.hidden_size
        vision_width = 768

        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.alpha = alpha
        self.max_txt_len = max_txt_len

        self.organs = [
            'lung', 'heart', 'esophagus', 'aorta'
        ]
        
        self.attention = nn.MultiheadAttention(
            embed_dim=vision_width,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.vision_projs = nn.ModuleList([nn.Linear(vision_width, embed_dim) for _ in range(len(self.organs))])
        self.query_tokens = nn.Parameter(torch.zeros(len(self.organs), vision_width))

    def _rampup_factor(self, epoch, iters, num_iters_per_epoch):
        return min(1, (epoch * num_iters_per_epoch + iters) / (2 * num_iters_per_epoch))

    def forward(self, samples):
        image = samples["image"]
        seg = samples["seg"]

        with torch.no_grad():
            organ_mask_flags = torch.zeros(len(seg), len(self.organs), dtype=bool, device=seg.device)
            for i, pul_seg in enumerate(seg):
                boundaries = [
                    pul_seg[0], pul_seg[-1],
                    pul_seg[:, 0], pul_seg[:, -1],
                    pul_seg[:, :, 0], pul_seg[:, :, -1]
                ]
                
                non_zero_boundaries = [b[b != 0].flatten() for b in boundaries]
                boundary_values = torch.cat(non_zero_boundaries)
                boundary_organs = torch.unique(boundary_values)

                organ_ids, organ_counts = torch.unique(pul_seg, return_counts=True)
                organ_ids = organ_ids[organ_ids > 0]
                
                # remove incomplete organs caused by random crop.
                intact_organ_ids = [organ_id for organ_id in organ_ids if organ_id not in boundary_organs]
                intact_organ_ids = torch.tensor(intact_organ_ids).long()
                    
                organ_mask_flags[i][intact_organ_ids - 1] = True

        organ_captions = samples["text_input"]
        organ_abnormal_flags = samples["organ_abnormal_flags"]
        
        # image embeddings and features
        image_embeds, hidden_image_embeds = self.visual_encoder(image)

        B, L, C = image_embeds.size()
        
        with torch.no_grad():
            organ_token_flags = torch.zeros(B, len(self.organs), L, dtype=bool).to(image.device)
            for i in range(B):
                inds = torch.where(organ_mask_flags[i])[0]
                if not len(inds):
                    continue
                
                masks = torch.stack(
                    [torch.eq(seg[i], organ_id + 1) for organ_id in inds], dim=0).float()

                downsampled_masks = F.max_pool3d(
                    masks.unsqueeze(1),
                    kernel_size=(16, 16, 32),
                    stride=(16, 16, 32)
                )
                
                organ_token_flags[i][inds] = downsampled_masks.flatten(1) > 0

                assert all((downsampled_masks.flatten(1) > 0).sum(1) > 0)
        
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # criteria to calculate loss
        with torch.no_grad():
            organ_status_world = (organ_abnormal_flags & organ_mask_flags).sum(0)
            if is_dist_avail_and_initialized():
                dist.all_reduce(organ_status_world, op=dist.ReduceOp.SUM)
        
        cl_organ_ids = torch.where(organ_status_world)[0]
        
        organ_wise_loss_itm = {}
        for cl_organ_id in cl_organ_ids:
            organ_name = self.organs[cl_organ_id]

            template = f'{organ_name} shows no significant abnormalities.'

            cl_patient_ids = torch.where(organ_mask_flags[:, cl_organ_id])[0]

            if not len(cl_patient_ids):
                image_feat = torch.empty(0, 256, dtype=torch.float).to(image.device)
                text_feat = torch.empty(0, 256, dtype=torch.float).to(image.device)
                cl_text_input = []

            else:
                # image_feat = self.get_roi_features(image_embeds, organ_token_flags, cl_patient_ids, cl_organ_id)
                image_feat = self.get_roi_features(hidden_image_embeds, organ_token_flags, cl_patient_ids, cl_organ_id)
                image_feat = self.vision_projs[cl_organ_id](image_feat)
                image_feat = F.normalize(image_feat, dim=-1)

                cl_text_input = [organ_captions[organ_name][cl_patient_id] for cl_patient_id in cl_patient_ids]
                cl_text_input1 = [
                    text.replace(template, '') if text.startswith(template) and text != template else text for text in cl_text_input
                ]

                text = self.tokenizer(
                    cl_text_input1,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_txt_len,
                    return_tensors="pt",
                ).to(image.device)

                text_output = self.text_encoder.forward_text(text)
                text_embeds = text_output.last_hidden_state
                text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

            # NOTE: gather image and text feats
            if is_dist_avail_and_initialized():
                image_feat_all = [feat.to(image_feat.device) for feat in all_gather(image_feat)]
                text_feat_all = [feat.to(text_feat.device) for feat in all_gather(text_feat)]

                image_feat_all[dist.get_rank()] = image_feat
                text_feat_all[dist.get_rank()] = text_feat

                image_feat_all = torch.cat(image_feat_all, dim=0)
                text_feat_all = torch.cat(text_feat_all, dim=0)

            else:
                image_feat_all = image_feat
                text_feat_all = text_feat

            cl_text_input = np.array(cl_text_input)
            if is_dist_avail_and_initialized():
                gathered_cl_text_input = all_gather(cl_text_input)
                cl_text_input_all = np.concatenate(gathered_cl_text_input)
            else:
                cl_text_input_all = cl_text_input

            sim_i2t = image_feat_all @ text_feat_all.t() / self.temp
            sim_t2i = text_feat_all @ image_feat_all.t() / self.temp
            
            with torch.no_grad():
                sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
                sim_targets.fill_diagonal_(1)
                
                normal_flag = [text_input.startswith(template) for text_input in cl_text_input_all]
                normal_flag = np.array(normal_flag)

                semantic_matrix_batch = normal_flag[:, None] * normal_flag[None, :]
                semantic_matrix_batch = semantic_matrix_batch.astype(float)
                
                semantic_matrix_batch1 = cl_text_input_all[:, None] == cl_text_input_all[None, :]
                semantic_matrix_batch1 = semantic_matrix_batch1.astype(float)

                semantic_matrix_batch = semantic_matrix_batch + semantic_matrix_batch1
                semantic_matrix_batch = semantic_matrix_batch.astype(bool)
                
                semantic_matrix_batch = torch.from_numpy(semantic_matrix_batch).to(image.device)
                semantic_matrix_batch.fill_diagonal_(0)

                sim_targets += semantic_matrix_batch
                sim_targets /= sim_targets.sum(1, keepdim=True)

            if len(torch.unique(sim_targets)) == 1 or not len(cl_text_input):
                continue
            
            sim_i2t_targets = sim_targets
            sim_t2i_targets = sim_targets

            loss_i2t = - torch.sum(
                F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1
            ).mean()
            
            loss_t2i = - torch.sum(
                F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1
            ).mean()
            
            loss_itc = (loss_i2t + loss_t2i) / 2

            organ_wise_loss_itm.update({f'{organ_name}_itc': loss_itc})

        loss_itc = sum(organ_wise_loss_itm.values())
 
        return BlipOutput(
            loss=loss_itc,
            organ_wise_loss_itm=organ_wise_loss_itm
        )
    
    # def get_roi_features(self, image_embeds, organ_token_flags, cl_patient_ids, cl_organ_id):
    #     query = self.query_tokens[cl_organ_id].unsqueeze(0).unsqueeze(0)

    #     roi_feats = []
    #     for image_embed, tokens in zip(image_embeds[cl_patient_ids], organ_token_flags[cl_patient_ids, cl_organ_id]):
    #         key = value = image_embed[tokens].unsqueeze(0)

    #         updated_query_token, _ = self.attention(query, key, value)
    #         roi_feats.append(updated_query_token.squeeze(0))

    #     roi_feats = torch.cat(roi_feats, dim=0)
    #     return roi_feats

    def get_roi_features(self, hidden_image_embeds, organ_token_flags, cl_patient_ids, cl_organ_id):
        query = self.query_tokens[cl_organ_id].unsqueeze(0).unsqueeze(0)

        roi_feats = []
        for patient_id, tokens in zip(cl_patient_ids, organ_token_flags[cl_patient_ids, cl_organ_id]):
            # key = value = image_embed[tokens].unsqueeze(0)
            
            organ_tokens = []
            for ms_image_embed in hidden_image_embeds:
                organ_tokens.append(ms_image_embed[patient_id, tokens])
            key = value = torch.cat(organ_tokens, dim=0).unsqueeze(0)

            # key = value = [image_embed[tokens].unsqueeze(0)]

            updated_query_token, _ = self.attention(query, key, value)
            roi_feats.append(updated_query_token.squeeze(0))

        roi_feats = torch.cat(roi_feats, dim=0)
        return roi_feats

    @classmethod
    def from_config(cls, cfg=None):
        # set from_pretrained=True to load weights for 'bert-base-chinese'
        # image_encoder = ResNetEncoder.from_config(cfg, from_pretrained=True)

        import torch
        from lavis.models.blip_models.vit import ViT

        model = ViT(
            in_channels=1,
            img_size=(112, 256, 352),
            patch_size=(16, 16, 32),
            num_classes=0,
            dropout_rate=0.1,
            qkv_bias=True
        )
        
        ckpt = torch.load(
            '/storage/guoruizhe/cache/hub/datasets--ibrahimhamamci--CT-RATE/code/mae_pretrain_vit_base.pth',
            map_location='cpu'
        )

        from collections import OrderedDict
        new_ckpt = OrderedDict()
        for key, value in ckpt['model'].items():
            if key.startswith("decoder") or key == 'mask_token' or key == "cls_token" or key.startswith("patch_embed"):
                continue
            
            if key.startswith("pos_embed"):
                value = value[0, 1:].reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
                value = F.interpolate(value, size=(16, 11), mode='bilinear', align_corners=False)
                value = value.unsqueeze(2).repeat(1, 1, 7, 1, 1).flatten(2).permute(0, 2, 1)
                new_ckpt['patch_embedding.position_embeddings'] = value
                continue

            new_ckpt[key.replace('fc', 'linear').replace('proj', 'out_proj')] = value
        model.load_state_dict(new_ckpt, strict=False)

        image_encoder = model

        text_encoder = XBertEncoder.from_config(cfg, from_pretrained=True)
        text_decoder = None

        alpha = cfg.get("alpha", 0.4)
        max_txt_len = cfg.get("max_txt_len", 250)
        embed_dim = 256

        model = cls(
            image_encoder=image_encoder,
            text_encoder=text_encoder,
            text_decoder=text_decoder,
            embed_dim=embed_dim,
            alpha=alpha,
            tie_enc_dec_weights=False,
            max_txt_len=max_txt_len
        )

        model.load_checkpoint_from_config(cfg)

        return model

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data = param_m.data * self.momentum + param.data * (
                    1.0 - self.momentum
                )

    @torch.inference_mode()
    def forward_test_win(
        self, 
        images, 
        masks, 
        organ_logits,
        test_organs,
        text_feat_dict,
        organ_feat_dict,
        whole_organ_sizes,
        skip_organ=None
    ):
        # image_embeds  = self.visual_encoder(images)
        image_embeds, hidden_image_embeds = self.visual_encoder(images)

        B, L, C = image_embeds.size()

        margin = 2
        
        # remove channel dimension
        masks = masks.squeeze(1)
            
        # for i, (embed, mask) in enumerate(zip(image_embeds, masks)):
        for i, mask in enumerate(masks):
            boundaries = []
            for d in range(mask.dim()):
                start_slice = [slice(None)] * mask.dim()
                end_slice = [slice(None)] * mask.dim()
                
                start_slice[d] = slice(None, margin)
                end_slice[d] = slice(-margin, None)
                
                boundaries.append(mask[tuple(start_slice)][mask[tuple(start_slice)] > 0])
                boundaries.append(mask[tuple(end_slice)][mask[tuple(end_slice)] > 0])
            boundaries = torch.cat(boundaries)
            
            boundary_values = boundaries[boundaries > 0].flatten()
            boundary_organs = torch.unique(boundary_values)

            if skip_organ is not None:
                boundary_organs = boundary_organs[boundary_organs != skip_organ + 1]
            
            organ_ids, organ_counts = torch.unique(mask, return_counts=True)
            organ_ids = organ_ids.long()
            organ_counts = organ_counts[organ_ids != 0]
            organ_ids = organ_ids[organ_ids != 0]

            # organs not touch boundary
            intact_organ_ids = [organ_id for organ_id, organ_count in zip(organ_ids, organ_counts) if organ_id not in boundary_organs]
            intact_organ_ids = torch.tensor(intact_organ_ids, device=masks.device).long()
            intact_organ_ids = intact_organ_ids - 1
            
            if not len(intact_organ_ids):
                continue

            organ_sizes = dict(zip([self.organs[organ_id] for organ_id in intact_organ_ids], [organ_counts[organ_ids == organ_id + 1].item() for organ_id in intact_organ_ids]))

            for k, v in organ_sizes.items():
                if k in whole_organ_sizes and v / whole_organ_sizes[k] != 1:
                    print(f'Mask id: {i}', f'Rank: {dist.get_rank() if dist.is_initialized() else 0}', 'Incomplete', k, v / whole_organ_sizes[k])
                    if v / whole_organ_sizes[k] < 0.9 and self.organs.index(k) != skip_organ:
                        intact_organ_ids = intact_organ_ids[intact_organ_ids != self.organs.index(k)]
            
            for organ_id in intact_organ_ids:
                organ_name = self.organs[organ_id.item()]
                if organ_name not in test_organs:
                    continue
                
                organ_mask = torch.eq(mask, organ_id + 1).float()
                downsampled_masks = F.max_pool3d(
                    organ_mask.unsqueeze(0),
                    kernel_size=(16, 16, 32),
                    stride=(16, 16, 32)
                )
                
                tokens = downsampled_masks[0].flatten() > 0

                query = self.query_tokens[organ_id].unsqueeze(0).unsqueeze(0)
                # key = value = embed[tokens].unsqueeze(0)
                key = value = torch.cat([level_embeds[i][tokens] for level_embeds in hidden_image_embeds]).unsqueeze(0)

                updated_query_token, _ = self.attention(query, key, value)
                updated_query_token = updated_query_token.squeeze(0)

                image_feat = F.normalize(self.vision_projs[organ_id](updated_query_token), dim=-1)
                
                organ_feat_dict[organ_name] = image_feat.cpu().tolist()

                for item in organ_logits.keys():
                    if item[0] != organ_name:
                        continue

                    text_feat = text_feat_dict[item]

                    logits = image_feat @ text_feat.t() / self.temp
                    probs = logits.softmax(-1)
                    organ_logits[item].append(probs.cpu().tolist())
    
        return organ_logits

    def prepare_text_feat(self, test_items, length=None):
        if length is None:
            length = self.max_txt_len

        device = self.text_encoder.device
        text_feat_dict = {}
        for prompt, item in zip(*self._get_prompt(test_items)):
            text = self.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=length,
                return_tensors="pt",
            ).to(device)

            text_output = self.text_encoder.forward_text(text)
            text_embeds = text_output.last_hidden_state
            text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            text_feat_dict[tuple(item)] = text_feat

        return text_feat_dict
    
    @staticmethod
    def _get_prompt(
        test_items,
        organ_name: str = None
    ) -> str:
        if organ_name is not None:
            test_items = [item for item in test_items if item[0] == organ_name]

        negative_prompts = [item[2] for item in test_items]
        positive_prompts = [item[3] for item in test_items]

        prompts = list(zip(negative_prompts, positive_prompts))

        return prompts, test_items
