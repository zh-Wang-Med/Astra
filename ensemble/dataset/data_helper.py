
import re
import torch.utils.data as data
import pandas as pd
import torch
from tqdm import tqdm
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
    Transposed,
    Lambda,
    Flipd,
    MapTransform,
)
import numpy as np
import os
from scipy.ndimage import zoom
def threshold(x):
    # threshold at 1
    return x > -1000

class LoadNPZd_with_spacing(MapTransform):
    """从 NPZ 加载 CT 数据并进行重采样"""
    def __init__(self, keys, npz_key='ct', src_spacing=(0.8, 0.8, 0.8), tgt_spacing=(1.5, 0.75, 0.75)):
        super().__init__(keys)
        self.npz_key = npz_key
        self.src_spacing = np.array(src_spacing)
        self.tgt_spacing = np.array(tgt_spacing)
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                with np.load(d[key]) as npz_data:
                    arr = npz_data[self.npz_key]
            else:
                arr = d[key]
            
            arr = arr.astype(np.float32)
            
            # 1. 确保是 (D, H, W)
            if arr.ndim == 4 and arr.shape[0] == 1:
                arr = arr[0] # 先去掉通道维方便缩放

            # 2. 计算缩放因子 (zoom factor)
            # factor = 原始间距 / 目标间距
            zoom_factors = self.src_spacing / self.tgt_spacing
            
            # 3. 执行重采样
            # order=3 是双三次插值（适合图像），order=1 是线性插值（速度快）
            # 对于 CT 图像，建议使用 order=3 或 order=1
            arr = zoom(arr, zoom_factors, order=1, mode='constant', cval=arr.min())
            
            # 4. 重新添加通道维度：(D, H, W) -> (1, D, H, W)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            
            d[key] = arr
        
        return d
    

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


# 完整的 Transform Pipeline
ImageTransforms_ct_clip_v2 = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
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




ImageTransforms_ct_clip_v3_rsna_pe = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
    Orientationd(keys=["image"], axcodes="RAS"),
    Spacingd(keys=["image"], pixdim=(0.75, 0.75, 1.5), mode=("bilinear")),
    ScaleIntensityRanged(
            keys=["image"], a_min=-600, a_max=800, b_min=-1.0, b_max=1.0, clip=True
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

ImageTransforms_ct_clip_my = Compose([
    LoadImaged(keys=["image"],),
    EnsureChannelFirstd(keys=["image"]),
    CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
    Resized(keys=['image'], spatial_size=(480, 480, 240)),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, 
        a_max=1000, 
        b_min=-1.0, 
        b_max=1.0, 
        clip=True
    ),
    ToTensord(keys=["image"]),
])

ImageTransforms_merlin = Compose(
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

ImageTransforms_merlin_rsna_pe = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 3), mode=("bilinear")),
        CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
        Resized(keys=['image'], spatial_size=(224,224,160)),
        ScaleIntensityRanged(
            keys=["image"], a_min=-600, a_max=800, b_min=0.0, b_max=1.0, clip=True
        ),
        ToTensord(keys=["image"]),
    ]
)



ImageTransforms_fvlm = Compose(
    [
        LoadImaged(keys=["image"], image_only=True, ensure_channel_first=True),
        Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        Resized(keys=['image'], spatial_size=(112, 256, 352)),
        ScaleIntensityRanged(
                keys=["image"], a_min=-1150, a_max=350,
                b_min=0.0, b_max=1.0, clip=True
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
# ImageTransforms_ct_clip_v3_radchest = Compose([
#     LoadNPZd(keys=["image"], npz_key='ct'),
#     Flipd(keys=["image"], spatial_axis=0),
#     CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
#     Resized(keys=['image'], spatial_size=(240, 480, 480)),
#     ScaleIntensityRanged(
#             keys=["image"], a_min=-1000, a_max=1000, b_min=-1.0, b_max=1.0, clip=True
#         ),
#     ToTensord(keys=["image"]),
# ])

ImageTransforms_ct_clip_v3_radchest = Compose([
    LoadNPZd_with_spacing(keys=["image"], npz_key='ct'),
    # EnsureChannelFirstd(keys=["image"]),
    Flipd(keys=["image"], spatial_axis=0),
    # CropForegroundd(keys=['image'], source_key='image', select_fn=threshold),
    # Resized(keys=['image'], spatial_size=(160, 224, 224)),
    CenterCropPadd(
        keys=["image"],
        spatial_size=(240, 480, 480),  # (D, H, W)
        mode="constant",
        value=-1.0  # 对应归一化后的最小值
    ),
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-1000, 
        a_max=1000, 
        b_min=-1.0, 
        b_max=1.0, 
        clip=True
    ),
    ToTensord(keys=["image"]),
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

# 自定义加载器：专门读取 .npy 文件并处理维度
class LoadNPYd(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            # 1. 加载数据并转为 float32
            arr = np.load(d[key]).astype(np.float32)
            
            # 2. 检查并添加通道维度 (D, H, W) -> (1, D, H, W)
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            
            d[key] = arr
        return d

# 定义 Transform 流水线
ImageTransforms_CCII = Compose([
    # 1. 从路径加载 npy 数组
    LoadNPYd(keys=["image"]),
    
    # 2. 强度缩放：因为原始是 uint8 (0-255)，映射到 (-1.0, 1.0)
    # 这对应了你参考代码中 ScaleIntensityRanged 的逻辑
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=0, 
        a_max=255, 
        b_min=-1.0, 
        b_max=1.0, 
        clip=True
    ),
    
    # 3. Resize：将原始 (16, 512, 512) 调整为 (240, 480, 480)
    # mode="trilinear" 用于 3D 图像平滑缩放
    Resized(
        keys=["image"], 
        spatial_size=(20, 480, 480),  # 顺序为 (D, H, W)
        mode="trilinear"
    ),
    
    # 4. 转为 Tensor
    ToTensord(keys=["image"]),
])

ImageTransforms_CCII_high = Compose([
    # 1. 从路径加载 npy 数组
    LoadNPYd(keys=["image"]),
    
    # 2. 强度缩放：因为原始是 uint8 (0-255)，映射到 (-1.0, 1.0)
    # 这对应了你参考代码中 ScaleIntensityRanged 的逻辑
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=0, 
        a_max=255, 
        b_min=-1.0, 
        b_max=1.0, 
        clip=True
    ),
    
    # 3. Resize：将原始 (16, 512, 512) 调整为 (240, 480, 480)
    # mode="trilinear" 用于 3D 图像平滑缩放
    Resized(
        keys=["image"], 
        spatial_size=(240, 480, 480),  # 顺序为 (D, H, W)
        mode="trilinear"
    ),
    
    # 4. 转为 Tensor
    ToTensord(keys=["image"]),
])


class ParseDataset(data.Dataset):
    def __init__(self, road, args, split='train'):
        self.road = road
        self.args = args
        df = pd.read_csv(self.road)
        if self.args.dataset == 'ctrate':
            selected_data = df.iloc[:, 0:22]
            disease_num = 18
        if self.args.dataset == 'radchest':
            selected_data = df.iloc[:, 0:20]
            disease_num = 16
        if self.args.dataset == 'merlin':
            selected_data = df.iloc[:,0:34]
            disease_num = 30
        if self.args.dataset == 'rsna_pe':
            selected_data = df.iloc[:, 0:8]
            disease_num = 4
        self.split = split

        # ========== 新增：控制训练集样本数量 ==========
        if split == 'train' and hasattr(args, 'train_num') and args.train_num > 0:
            # 如果是训练集且指定了 train_num (大于0)，则只使用前 train_num 个样本
            total_samples = len(selected_data)
            train_num = min(args.train_num, total_samples)  # 防止超出范围
            selected_data = selected_data.iloc[:train_num]
            print(f"📊 训练集: 使用 {train_num}/{total_samples} 个样本")
        elif split == 'train':
            print(f"📊 训练集: 使用全部 {len(selected_data)} 个样本")

        self.study_id_list = selected_data.iloc[:, 0].tolist()
        self.imagepath_list = selected_data.iloc[:, 1].tolist()
        if self.args.text_help:
            if self.args.dataset == 'ctrate':
                self.generated_reports_list = selected_data.iloc[:, 21].tolist()
            if self.args.dataset == 'radchest':
                self.generated_reports_list = selected_data.iloc[:, 19].tolist()
            if self.args.dataset == 'merlin':
                self.generated_reports_list = selected_data.iloc[:, 33].tolist()
            if self.args.dataset == 'rsna_pe':
                self.generated_reports_list = selected_data.iloc[:, 7].tolist()

        # 创建18个列表来存储从第4列到第21列的数据
        self.label_lists = [[] for _ in range(disease_num)]

        # 使用循环将数据分配给对应的列表
        for i in range(disease_num):
            # print(i+3)
            self.label_lists[i] = selected_data.iloc[:, i+3].tolist()
        
    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        dict = {}
        dict['id'] = self.study_id_list[index]


        # npz
        if self.args.vision_encoder == "merlin":
            if self.args.dataset == "ctrate":
                npz_path = "../npz_data/ct_rate/"+dict['id'].replace('nii.gz','npz')
                if os.path.exists(npz_path):
                    npz_data = np.load(npz_path)
                    image = npz_data['image']  # (1, 224, 224, 160)
                    image_tensor = torch.from_numpy(image).float()
                    dict['image'] = image_tensor
                else:
                    datalist = [
                        {
                            "image": self.imagepath_list[index],  # function returns local path to nifti file
                        },
                    ]
                    image_tensor = []
                    image_tensor = torch.Tensor(ImageTransforms_merlin(datalist)[0]['image'])
                    # image_tensor = image_tensor.permute(0,3,1,2).unsqueeze(0) 
                    dict['image'] = image_tensor # !
            if self.args.dataset in ["rsna_pe"]:
                datalist = [
                    {
                        "image": self.imagepath_list[index],  # function returns local path to nifti file
                    },
                ]
                image_tensor = torch.Tensor(ImageTransforms_merlin_rsna_pe(datalist)[0]['image'])
                dict['image'] = image_tensor # !
                # np.savez_compressed(npz_path, image=image_tensor.numpy())
            if self.args.dataset == "radchest":
                datalist = [
                        {
                            "image": self.imagepath_list[index],  # function returns local path to nifti file
                        },
                    ]
                image_tensor = torch.Tensor(ImageTransforms_merlin_radchest(datalist)[0]['image'])
                image_tensor = image_tensor.permute(0,2,3,1) 
                image_tensor = torch.rot90(image_tensor, k=-1, dims=[1, 2])
                dict['image'] = image_tensor # !
            if self.args.dataset == "merlin":
                datalist = [
                        {
                            "image": self.imagepath_list[index],  # function returns local path to nifti file
                        },
                    ]
                image_tensor = torch.Tensor(ImageTransforms_merlin_abd(datalist)[0]['image'])
                dict['image'] = image_tensor # !
        if self.args.vision_encoder == "ct_clip":
            if self.args.dataset == "ctrate":
                datalist = [
                    {
                        "image": self.imagepath_list[index],  # function returns local path to nifti file
                    },
                ]
                image_tensor = []
                if self.args.ct_clip_pre == 'ori':
                    npz_path = '../ctrate_classification/'+dict['id'].replace('nii.gz','npz')
                    if os.path.exists(npz_path):
                        npz_data = np.load(npz_path)
                        image = npz_data['video_tensor']  # (1, 224, 224, 160)
                        image_tensor = torch.from_numpy(image).float()
                        image_tensor = image_tensor.permute(0,2,3,1)
                    else:
                        print('npz not find')
                        image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3(datalist)[0]['image'])
                    # dict['image'] = image_tensor
                if self.args.ct_clip_pre == 'v3':
                    npz_path = '../ctrate_mypretrain/'+dict['id'].replace('nii.gz','npz')
                    if os.path.exists(npz_path):
                        npz_data = np.load(npz_path)
                        image = npz_data['video_tensor']  # (1, 224, 224, 160)
                        image_tensor = torch.from_numpy(image).float()
                        image_tensor = image_tensor.permute(0,2,3,1)
                    else:
                        print('npz not find')
                        image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3(datalist)[0]['image'])
                    # image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3(datalist)[0]['image'])
                if self.args.ct_clip_pre == 'my':
                    image_tensor = torch.Tensor(ImageTransforms_ct_clip_my(datalist)[0]['image'])
                image_tensor = image_tensor.permute(0,3,1,2)
                dict['image'] = image_tensor # !
            if self.args.dataset in ["rsna_pe"]:
                datalist = [
                    {
                        "image": self.imagepath_list[index],  # function returns local path to nifti file
                    },
                ]
                npz_path = '../rsna_ped/ctclip_npz/'+dict['id']+".npz"
                if os.path.exists(npz_path):
                    try:
                        npz_data = np.load(npz_path)
                        image = npz_data['video_tensor']  # (1, 224, 224, 160)
                        image_tensor = torch.from_numpy(image).float()
                        # image_tensor = image_tensor.permute(0,2,3,1)
                    except:
                        image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_rsna_pe(datalist)[0]['image'])
                        np.savez_compressed(npz_path, video_tensor=image_tensor.numpy())
                else:
                    print('npz not find')
                    image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_rsna_pe(datalist)[0]['image'])
                    np.savez_compressed(npz_path, video_tensor=image_tensor.numpy())
                image_tensor = image_tensor.permute(0,3,1,2)
                dict['image'] = image_tensor # !
            if self.args.dataset == "radchest":
                datalist = [
                    {
                        "image": self.imagepath_list[index],  # function returns local path to nifti file
                    },
                ]
                image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_radchest(datalist)[0]['image'])
                image_tensor = image_tensor.permute(0, 2, 3 ,1)
                image_tensor = torch.rot90(image_tensor, k=-1, dims=[1, 2]) 
                image_tensor = image_tensor.permute(0,3,1,2)
                dict['image'] = image_tensor # !
            if self.args.dataset == "merlin":
                datalist = [
                        {
                            "image": self.imagepath_list[index],  # function returns local path to nifti file
                        },
                    ]
                npz_path = "../merlin/classification/ctclip_class/npz/"+dict['id']+".npz"
                if os.path.exists(npz_path):
                    npz_data = np.load(npz_path)
                    image = npz_data['video_tensor']  # (1, 224, 224, 160)
                    image_tensor = torch.from_numpy(image).float()
                    # image_tensor = image_tensor.permute(0,2,3,1)
                else:
                    print('npz not find')
                    image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_merlin(datalist)[0]['image'])
                    np.savez_compressed(npz_path, video_tensor=image_tensor.numpy())
                image_tensor = image_tensor.permute(0,3,1,2)
                dict['image'] = image_tensor # !
                # image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_merlin(datalist)[0]['image'])
                # image_tensor = image_tensor.permute(0,3,1,2)
                # dict['image'] = image_tensor # !

        if self.args.dataset == "ctrate":
            label_names = [
                'Medical_material', 'Arterial_wall_calcification', 'Cardiomegaly', 'Pericardial_effusion',
                'Coronary_artery_wall_calcification', 'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Mosaic_attenuation_pattern', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening'
            ]
        if self.args.dataset == "radchest":
            label_names = [
                'Medical_material', 'Cardiomegaly', 'Pericardial_effusion',
                'Hiatal_hernia', 'Lymphadenopathy', 'Emphysema',
                'Atelectasis', 'Lung_nodule', 'Lung_opacity', 'Pulmonary_fibrotic_sequela',
                'Pleural_effusion', 'Peribronchial_thickening',
                'Consolidation', 'Bronchiectasis', 'Interlobular_septal_thickening', 'calcification'
            ]
        if self.args.dataset == "merlin":
            label_names = [
                'submucosal_edema', 'renal_hypodensities', 'aortic_valve_calcification',
                'coronary_calcification', 'thrombosis', 'metastatic_disease',
                'pancreatic_atrophy', 'renal_cyst', 'osteopenia',
                'surgically_absent_gallbladder', 'atelectasis', 'abdominal_aortic_aneurysm',
                'anasarca', 'hiatal_hernia', 'lymphadenopathy',
                'prostatomegaly', 'biliary_ductal_dilation', 'cardiomegaly',
                'splenomegaly', 'hepatomegaly', 'atherosclerosis',
                'ascites', 'pleural_effusion', 'hepatic_steatosis',
                'appendicitis', 'gallstones', 'hydronephrosis',
                'bowel_obstruction', 'free_air', 'fracture'
            ]
        if self.args.dataset == "rsna_pe":
            label_names = ['leftsided_pe','rightsided_pe','central_pe','chronic_pe']



        for i, name in enumerate(label_names):
            label_value = torch.tensor(float(self.label_lists[i][index]))
            dict[name] = label_value
        if self.args.text_help:
            # dict['generated_report'] = self.generated_reports_list[index]
            if self.args.dataset == "ctrate":
                embedding_npz_path = "../radgenome_ct/qwen3_embeddings/r1_v1_v2_8000step/"+dict['id'].replace('nii.gz','npz')
            if self.args.dataset == "radchest":
                embedding_npz_path = "../rad_chest/qwen3_embeddings/r1_v1_v2_8000step/" + str(dict['id']) + ".npz"
            if self.args.dataset == "merlin":
                embedding_npz_path = "../merlin/qwen3_embeddings/r1_v1_v2_8000step/"+dict['id']+ ".npz"
            if self.args.dataset in ["rsna_pe"]:
                embedding_npz_path = "../rsna_ped/rsna_ped/qwen3_embeddings/r1_v1_v2_8000step/v1/"+dict['id']+ ".npz"
            npz_data = np.load(embedding_npz_path)
            text_embedding = npz_data['embedding']  # (1, 224, 224, 160)
            text_embedding_tensor = torch.from_numpy(text_embedding).float()
            dict['text_embedding'] = text_embedding_tensor
        return dict



def create_datasets(args):
    predict_dataset = ParseDataset(args.predictroad, args, 'val')
    train_dataset = ParseDataset(args.trainroad, args, 'train')
    val_dataset = ParseDataset(args.valroad, args, 'val')
    test_dataset = ParseDataset(args.testroad, args, 'test')
    return train_dataset,val_dataset,test_dataset,predict_dataset

