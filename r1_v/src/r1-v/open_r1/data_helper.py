import gc
import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
from transformers import BertTokenizer, AutoImageProcessor
import pandas as pd
import random
# from config.config import parser
import math
import torch
from tqdm import tqdm
import random
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
    CenterSpatialCropd,
    CropForegroundd,
    Resized,
)
import ipdb
import torch.nn.functional as F

def resize_chwd(x: torch.Tensor, target_hwd, mode: str):
    # x: [C, H, W, D]
    # F.interpolate 要求 3D 输入为 [N, C, D, H, W]
    ncdhw = x.unsqueeze(0).permute(0, 1, 4, 2, 3) # [1, C, D, H, W]
    size_3d = (target_hwd[2], target_hwd[0], target_hwd[1]) # (D, H, W)
    if mode == "nearest":
        out_ncdhw = F.interpolate(ncdhw, size=size_3d, mode=mode)
    else:
        out_ncdhw = F.interpolate(ncdhw, size=size_3d, mode=mode, align_corners=False)
    out_chwd = out_ncdhw.squeeze(0)
    return out_chwd

def threshold(x):
    # threshold at 1
    return x > -1000

ImageTransforms_merlin = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=200, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)

ImageTransforms_radfm = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(256,256,64)),
        # Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=200, b_min=-1.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)

ImageTransforms_merlin_both = Compose([
    LoadImaged(keys=["image", "mask"]),
    EnsureChannelFirstd(keys=["image", "mask"]),
    CropForegroundd(keys=["image", "mask"], source_key="image", select_fn=threshold),
    Resized(keys=["image"], spatial_size=(224, 224, 160), mode="trilinear"),
    Resized(keys=["mask"], spatial_size=(224, 224, 160), mode="nearest"),
    ScaleIntensityRanged(
    keys=["image"], a_min=-1000, a_max=200, b_min=0.0, b_max=1.0, clip=True
    ),
    ToTensord(keys=["image", "mask"]),
])

ImageTransforms_merlin_abd = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        SpatialPadd(keys=["image"], spatial_size=[224, 224, 160]),
        CenterSpatialCropd(
            roi_size=[224, 224, 160],
            keys=["image"],
        ),
        ToTensord(keys=["image"]),
    ]
)
ImageTransforms_merlin_abd_nospace = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)
ImageTransforms_merlin_chest = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=200, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)
ImageTransforms_merlin_chest_abd = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)

def pad_cam_list(cam_list, K):
    if len(cam_list) == 0:
        cam_list = [[5,3,3]]
    cam_list = cam_list[:K]
    pad_len = K - len(cam_list)
    if pad_len > 0:
        cam_list = cam_list + [[0, 0, 0] for _ in range(pad_len)]
    mask = [1] * (K - pad_len) + [0] * pad_len
    cam_tensor = torch.tensor(cam_list, dtype=torch.float32)
    # print(cam_tensor.shape)
    # cam_tensor = cam_tensor.reshape(K, 3)
    # print(cam_tensor.shape)
    return cam_tensor, mask

class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset

    def parse(self, features):
        to_return = {'id': str(features['id'])}

        original_text = features['whole_report'].lower().replace('normal..','normal.')

        to_return['input_text'] = original_text
        datalist = [
            {
                "image": features['image_path'],  # function returns local path to nifti file
            },
        ]
        if self.args.visual_encoder_name == "merlin":
            if self.args.whether_npz:
                mini_dataset = features['dataset']
                to_return['dataset'] = features['dataset']
                # image_tensor = self._load_npz(features['preprocess_path'])
                npz_path = features['preprocess_path']
                npz_data = np.load(npz_path)
                image = npz_data['image']  # (1, 224, 224, 160)
                
                # 转换为tensor
                image_tensor = torch.from_numpy(image).float()
                to_return['image'] = image_tensor
            else:
                mini_dataset = features['dataset']
                to_return['dataset'] = features['dataset']
                if mini_dataset in ['ct_rate','inspect','bimcv','guizhou','rsna_pe']:
                    image_tensor = torch.Tensor(ImageTransforms_merlin_chest(datalist)[0]['image'])
                if mini_dataset in ['merlin','atlas','amosmm','guizhou_abdomen','inhouse_duilie1']:
                    try:
                        image_tensor = torch.Tensor(ImageTransforms_merlin_abd(datalist)[0]['image'])
                    except:
                        image_tensor = torch.zeros(1, 224, 224, 160)
                if mini_dataset == 'ctrg':
                    image_tensor = torch.Tensor(ImageTransforms_merlin_ctrg(datalist)[0]['image'])
                    image_tensor = image_tensor.permute(0,2,3,1) 
                    image_tensor = torch.rot90(image_tensor, k=-1, dims=[1, 2]) 
                if mini_dataset == 'radchest':
                    image_tensor = torch.Tensor(ImageTransforms_merlin_radchest(datalist)[0]['image'])
                    image_tensor = image_tensor.permute(0,2,3,1) 
                    image_tensor = torch.rot90(image_tensor, k=-1, dims=[1, 2]) 
                if mini_dataset == 'nlst':
                    image_tensor = torch.Tensor(ImageTransforms_merlin_chest(datalist)[0]['image'])
                if image_tensor.shape != (1, 224, 224, 160):
                    print(datalist)
                to_return['image'] = image_tensor
        if self.args.visual_encoder_name == "radfm":
            mini_dataset = features['dataset']
            to_return['dataset'] = features['dataset']
            if mini_dataset in ['ct_rate','inspect','bimcv','guizhou','rsna_pe']:
                image_tensor = torch.Tensor(ImageTransforms_radfm(datalist)[0]['image'])
            if mini_dataset in ['merlin','atlas','amosmm','guizhou_abdomen']:
                # image_tensor = torch.Tensor(ImageTransforms_merlin_abd(datalist)[0]['image'])
                try:
                    image_tensor = torch.Tensor(ImageTransforms_radfm_abd(datalist)[0]['image'])
                except:
                    image_tensor = torch.zeros(1, 224, 224, 160)
            if mini_dataset == 'ctrg':
                image_tensor = torch.Tensor(ImageTransforms_radfm(datalist)[0]['image'])
            if mini_dataset == 'radchest':
                image_tensor = torch.Tensor(ImageTransforms_radfm(datalist)[0]['image'])
            if mini_dataset == 'nlst':
                image_tensor = torch.Tensor(ImageTransforms_radfm(datalist)[0]['image'])
            image_tensor = image_tensor.repeat(3, 1, 1, 1)

            to_return['image'] = image_tensor # !
        if self.args.visual_encoder_name == "ctclip":
            mini_dataset = features['dataset']
            to_return['dataset'] = features['dataset']
            if mini_dataset == "ct_rate":
                image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3(datalist)[0]['image'])
            if mini_dataset in ['inspect','bimcv','guizhou','rsna_pe']:
                image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_other_chest(datalist)[0]['image'])
            if mini_dataset in ['merlin','atlas','amosmm','guizhou_abdomen']:
                try:
                    image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_merlin(datalist)[0]['image'])
                except:
                    image_tensor = torch.zeros(1, 480, 480, 240)
            if mini_dataset == 'ctrg':
                image_tensor = torch.Tensor(ImageTransforms_radfm(datalist)[0]['image'])
            image_tensor = image_tensor.permute(0,3,1,2)
            to_return['image'] = image_tensor # !


        return to_return

    def transform_with_parse(self, inputs):
        return self.parse(inputs)



class ParseDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.meta = json.load(open(args.annotation, 'r'))
        self.meta = self.meta[split]
        self.parser = FieldParser(args)
        self.dataset = args.dataset

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.meta[index])

def create_datasets(args):
    train_dataset = ParseDataset(args, 'train')
    dev_dataset = ParseDataset(args, 'val')
    return train_dataset,dev_dataset



