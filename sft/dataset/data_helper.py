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
from config.config import parser
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
    Flipd,
)
import ipdb
import torch.nn.functional as F
from functools import lru_cache
from monai.transforms import MapTransform




class LoadNPZd(MapTransform):
    """从 NPZ 加载 CT 数据"""
    def __init__(self, keys, npz_key='ct'):
        super().__init__(keys)
        self.npz_key = npz_key
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                with np.load(d[key]) as npz_data:
                    arr = npz_data[self.npz_key]
            else:
                arr = d[key]
            
            arr = arr.astype(np.float32)
            
            # 添加通道维度：(D, H, W) -> (1, D, H, W)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            
            d[key] = arr
        
        return d



def threshold(x):
    # threshold at 1
    return x > -1000
def threshold_ctrg(x):
    # threshold at 1
    return x > 42
class ApplyCircularMaskd(MapTransform):
    """
    对CT图像应用圆形mask，圆外区域设为背景值
    """
    def __init__(self, keys, radius=None, center=None, fill_value=0, allow_missing_keys=False):
        """
        Args:
            keys: 要处理的字典键
            radius: 圆的半径（像素），如果为None则使用图像较小边的一半
            center: 圆心位置 (h, w)，如果为None则使用图像中心
            fill_value: 圆外区域填充的值，默认为0
            allow_missing_keys: 是否允许缺失键
        """
        super().__init__(keys, allow_missing_keys)
        self.radius = radius
        self.center = center
        self.fill_value = fill_value
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            img = img.permute(0,2,3,1)
            
            # 检查是torch tensor还是numpy array
            is_torch = isinstance(img, torch.Tensor)
            
            # 统一转为numpy处理
            if is_torch:
                img_np = img.detach().cpu().numpy() if img.is_cuda else img.numpy()
            else:
                img_np = np.array(img)
            
            # 处理不同维度的图像
            if img_np.ndim == 4:  # (C, H, W, D)
                C, H, W, D = img_np.shape
                
                # 确定圆心
                if self.center is None:
                    center_h = H // 2
                    center_w = W // 2
                else:
                    center_h, center_w = self.center
                
                # 确定半径
                if self.radius is None:
                    radius = min(H, W) // 2
                else:
                    radius = self.radius
                
                # 创建网格坐标
                y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                
                # 计算距离
                distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
                
                # 创建圆形mask
                circular_mask = distances <= radius
                
                # 对每个切片应用mask
                for z_idx in range(D):
                    for c_idx in range(C):
                        img_np[c_idx, :, :, z_idx] = np.where(
                            circular_mask,
                            img_np[c_idx, :, :, z_idx],
                            self.fill_value
                        )
            
            elif img_np.ndim == 3:  # (C, H, W)
                C, H, W = img_np.shape
                
                if self.center is None:
                    center_h = H // 2
                    center_w = W // 2
                else:
                    center_h, center_w = self.center
                
                if self.radius is None:
                    radius = min(H, W) // 2
                else:
                    radius = self.radius
                
                y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
                distances = np.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
                circular_mask = distances <= radius
                
                for c_idx in range(C):
                    img_np[c_idx, :, :] = np.where(
                        circular_mask,
                        img_np[c_idx, :, :],
                        self.fill_value
                    )
            
            # img_np = img_np.permute(0,3,1,2)
            # 转回原来的类型
            if is_torch:
                img = torch.from_numpy(img_np).to(img.dtype).to(img.device)
                d[key] = img.permute(0,3,1,2)
            else:
                d[key] = img_np
        
        return d


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

ImageTransforms_radfm_abd = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(256,256,64)),
        # Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)

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

ImageTransforms_merlin_radchest = Compose([
    LoadNPZd(keys=["image"], npz_key='ct'),
    # EnsureChannelFirstd(keys=["image"]),
    Flipd(keys=["image"], spatial_axis=0),
    CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
    Resized(keys=['image'], spatial_size=(160, 224, 224)),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, 
        a_max=200, 
        b_min=0.0, 
        b_max=1.0, 
        clip=True
    ),
    ToTensord(keys=["image"]),
])


ImageTransforms_merlin_ctrg = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ApplyCircularMaskd(keys=["image"], radius=220, fill_value=0),  # 添加圆形mask
        Flipd(keys=["image"], spatial_axis=0),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold_ctrg),
        Resized(keys=['image'], spatial_size=(160,224,224)),
        # Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=42, a_max=246, b_min=0, b_max=1, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)




# 如果需要自定义中心裁剪+padding的逻辑
class CenterCropPadd(MapTransform):
    """
    中心裁剪或 padding 到目标尺寸
    """
    def __init__(self, keys, spatial_size, mode="constant", value=-1.0):
        super().__init__(keys)
        self.spatial_size = spatial_size  # (H, W, D)
        self.mode = mode
        self.value = value
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]  # shape: (C, H, W, D) 或 (C, X, Y, Z)
            
            # 获取当前尺寸（假设最后3个维度是空间维度）
            current_shape = img.shape[-3:]  # (H, W, D)
            target_shape = self.spatial_size
            
            # 计算裁剪/padding
            crop_slices = []
            pad_widths = []
            
            for i in range(3):
                curr_size = current_shape[i]
                target_size = target_shape[i]
                
                if curr_size >= target_size:
                    # 需要裁剪
                    start = (curr_size - target_size) // 2
                    end = start + target_size
                    crop_slices.append(slice(start, end))
                    pad_widths.append((0, 0))
                else:
                    # 需要 padding
                    crop_slices.append(slice(None))
                    pad_before = (target_size - curr_size) // 2
                    pad_after = target_size - curr_size - pad_before
                    pad_widths.append((pad_before, pad_after))
            
            # 裁剪
            if img.ndim == 4:  # (C, H, W, D)
                img = img[:, crop_slices[0], crop_slices[1], crop_slices[2]]
            else:
                img = img[crop_slices[0], crop_slices[1], crop_slices[2]]
            
            # Padding
            if any(p[0] + p[1] > 0 for p in pad_widths):
                if img.ndim == 4:
                    pad_widths = [(0, 0)] + pad_widths  # 添加 channel 维度
                img = np.pad(img, pad_widths, mode=self.mode, constant_values=self.value)
            
            d[key] = img
        
        return d

ImageTransforms_ct_clip_v3 = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(0.75, 0.75, 0.5), mode=("bilinear")),
    ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        ),
    CenterCropPadd(
        keys=["image"],
        spatial_size=(480, 480, 240),  # (D, H, W)
        mode="constant",
        value=-1.0  # 对应归一化后的最小值
    ),
    ToTensord(keys=["image"]),
])

ImageTransforms_ct_clip_v3_other_chest = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(0.75, 0.75, 1.5), mode=("bilinear")),
    ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        ),
    CenterCropPadd(
        keys=["image"],
        spatial_size=(480, 480, 240),  # (D, H, W)
        mode="constant",
        value=-1.0  # 对应归一化后的最小值
    ),
    ToTensord(keys=["image"]),
])
ImageTransforms_ct_clip_v3_merlin = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(0.75, 0.75, 1.5), mode=("bilinear")),
    ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
        ),
    CenterCropPadd(
        keys=["image"],
        spatial_size=(480, 480, 240),  # (D, H, W)
        mode="constant",
        value=-1.0  # 对应归一化后的最小值
    ),
    ToTensord(keys=["image"]),
])


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
    test_dataset = ParseDataset(args, 'test')
    return train_dataset,dev_dataset,test_dataset



