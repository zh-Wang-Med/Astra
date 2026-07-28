import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm
import ipdb
def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        percent = 100
        num_files = int((len(self.samples) * percent) / 100)
        #num_files = 2286
        self.samples = self.samples[:num_files]
        print(len(self.samples))
        self.count = 0

        df = pd.read_csv(meta_file) #select the metadata
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    def prepare_samples(self):
        samples = []
        # ipdb.set_trace()
        # for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
        #     for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

        #         for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
        #             accession_number = nii_file.split("/")[-1]
        #             #accession_number = accession_number.replace(".npz", ".nii.gz")
        #             if accession_number not in self.accession_to_text:
        #                 continue

        #             impression_text = self.accession_to_text[accession_number]

        #             if impression_text == "Not given.":
        #                 impression_text=""

        #             input_text_concat = ""
        #             for text in impression_text:
        #                 input_text_concat = input_text_concat + str(text)
        #             input_text_concat = impression_text[0]
        #             input_text = f'{impression_text}'
        #             samples.append((nii_file, input_text_concat))
        #             self.paths.append(nii_file)

        for nii_file in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*.nii.gz'))):
            accession_number = nii_file.split("/")[-1]
            #accession_number = accession_number.replace(".npz", ".nii.gz")
            if accession_number not in self.accession_to_text:
                ipdb.set_trace()
                continue

            impression_text = self.accession_to_text[accession_number]
            # ipdb.set_trace()

            if impression_text == "Not given.":
                impression_text=""

            input_text_concat = ""
            for text in impression_text:
                input_text_concat = input_text_concat + str(text)
            input_text_concat = impression_text[0]
            input_text = f'{impression_text}'
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)



    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")
        npz_base_path = "../dataset/ctrate/ctrate_ori_pretrain"
        npz_file = os.path.join(npz_base_path, f"{name_acc}.npz")
        if os.path.exists(npz_file):
            try:
                data = np.load(npz_file)
                video_tensor = torch.from_numpy(data['video_tensor'])
            except Exception as e:
                video_tensor = self.nii_to_tensor(nii_file)
                os.makedirs(npz_base_path, exist_ok=True)
                np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())
        else:
            video_tensor = self.nii_to_tensor(nii_file)
            os.makedirs(npz_base_path, exist_ok=True)
            np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())
        # video_tensor = self.nii_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text





# my 
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
    MapTransform
)
class LoadNPY(MapTransform):
    """从 NPZ 或 NPY 加载 CT 数据"""
    def __init__(self, keys, npz_key='ct', offset=1024):
        super().__init__(keys)
        self.npz_key = npz_key
        self.offset = offset  # NLST数据需要减去1024转换为HU值
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if isinstance(d[key], str):
                file_path = d[key]
                
                if file_path.endswith('.npz'):
                    with np.load(file_path) as npz_data:
                        arr = npz_data[self.npz_key]
                
                elif file_path.endswith('.npy'):
                    arr = np.load(file_path)
                    
                    if arr.dtype == np.uint16:
                        arr = arr.astype(np.float32) - self.offset
                
                else:
                    raise ValueError(f"不支持的文件格式: {file_path}")
            else:
                arr = d[key]
            
            arr = arr.astype(np.float32)
            
            if arr.ndim == 3:
                arr = arr[np.newaxis, ...]
            
            d[key] = arr
        
        return d
def threshold(x):
    # threshold at 1
    return x > -1000
ImageTransforms_merlin_nlst = Compose([
    LoadNPY(keys=["image"], npz_key='ct', offset=1000),  # 支持NPZ和NPY
    # Flipd(keys=["image"], spatial_axis=0),
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
class my_CTReportDataset(Dataset):
    def __init__(self, data_folder, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.reports =[]
        self.image_paths = []
        self.count = 0
        self.load_info(self.data_folder)

    def __len__(self):
        return len(self.reports)

    def load_info(self, data_folder):
        import json
        import os
        
        self.reports = []
        self.image_paths = []
        
        # 读取 JSON
        with open(data_folder, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        base_path = "../datasets/Med_datasets/NLST/jhcnas3/CTs/data/NLST_convert_v1/"
        
        for case_id, report_list in data_dict.items():
            image_path = os.path.join(base_path, f"{case_id}.npy")
            
            if not os.path.exists(image_path):
                continue
            
            report = report_list[0] if isinstance(report_list, list) and len(report_list) > 0 else ""
            
            self.reports.append(report)
            self.image_paths.append(image_path)
        
        print(f"加载了 {len(self.reports)} 个样本")
        
        if len(self.reports) == 0:
            raise ValueError("没有找到任何有效样本！")
        
    def nii_img_to_tensor(self, path):
        datalist = [
                    {
                        "image": path,  # function returns local path to nifti file
                    },
                ]
        image_tensor = torch.Tensor(ImageTransforms_merlin_nlst(datalist)[0]['image'])
        image_tensor = image_tensor.permute(0,3,1,2) 
        return image_tensor

    def __getitem__(self, index):
        # nii_file, input_text = self.samples[index]
        nii_file = self.image_paths[index]
        input_text = self.reports[index]
        video_tensor = self.nii_img_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text




# my ctrate
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
            
            current_shape = img.shape[-3:]  # (H, W, D)
            target_shape = self.spatial_size
            
            crop_slices = []
            pad_widths = []
            
            for i in range(3):
                curr_size = current_shape[i]
                target_size = target_shape[i]
                
                if curr_size >= target_size:
                    start = (curr_size - target_size) // 2
                    end = start + target_size
                    crop_slices.append(slice(start, end))
                    pad_widths.append((0, 0))
                else:
                    crop_slices.append(slice(None))
                    pad_before = (target_size - curr_size) // 2
                    pad_after = target_size - curr_size - pad_before
                    pad_widths.append((pad_before, pad_after))
            if img.ndim == 4:  # (C, H, W, D)
                img = img[:, crop_slices[0], crop_slices[1], crop_slices[2]]
            else:
                img = img[crop_slices[0], crop_slices[1], crop_slices[2]]
            
            if any(p[0] + p[1] > 0 for p in pad_widths):
                if img.ndim == 4:
                    pad_widths = [(0, 0)] + pad_widths  
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
ImageTransforms_ct_clip_v3_nlst = Compose([
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
class my_ctrate_CTReportDataset(Dataset):
    def __init__(self, data_folder, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.reports =[]
        self.image_paths = []
        self.count = 0
        self.load_info(self.data_folder)

    def __len__(self):
        return len(self.reports)

    def load_info(self, data_folder):
        """从 JSON 加载数据"""
        import json
        import os
        
        self.reports = []
        self.image_paths = []
        
        with open(data_folder, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        base_radgenemo_path = "../dataset/radgenome_chestct/dataset/train_preprocessed/"
        base_nlst_path = "../NLST_convert_v1/"
        
        split = ''
        for case_id, report_list in data_dict.items():
            if case_id.startswith('train') or case_id.startswith('val'):
                split = 'radgenome'
            else:
                split = 'nlst'

            if split == 'radgenome':
                case_id_split = case_id.split('_')
                image_path = base_radgenemo_path + 'train_' + case_id_split[1] + '/train_' + case_id_split[1] + case_id_split[2] + '/' + case_id
            if split == 'nlst':
                image_path = os.path.join(base_nlst_path, f"{case_id}.nii.gz")
            
            if not os.path.exists(image_path):
                continue
            
            report = report_list[0] if isinstance(report_list, list) and len(report_list) > 0 else ""
            
            self.reports.append(report)
            self.image_paths.append(image_path)
        
        print(f"加载了 {len(self.reports)} 个样本")
        
        if len(self.reports) == 0:
            raise ValueError("没有找到任何有效样本！")
        
    def nii_img_to_tensor(self, path,split):
        datalist = [
                    {
                        "image": path,  # function returns local path to nifti file
                    },
                ]
        if split == 'radgenome':
            image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3(datalist)[0]['image'])
        if split == 'nlst':
            image_tensor = torch.Tensor(ImageTransforms_ct_clip_v3_nlst(datalist)[0]['image'])
        image_tensor = image_tensor.permute(0,3,1,2) 
        return image_tensor

    def __getitem__(self, index):
        # nii_file, input_text = self.samples[index]
        nii_file = self.image_paths[index]
        input_text = self.reports[index]
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")
        split = ''
        if name_acc.startswith('train') or name_acc.startswith('val'):
            split = 'radgenome'
        else:
            split = 'nlst'

        npz_base_path = "../dataset/ctrate/ctrate_mypretrain"
        npz_file = os.path.join(npz_base_path, f"{name_acc}.npz")
        if os.path.exists(npz_file):
            try:
                data = np.load(npz_file)
                video_tensor = torch.from_numpy(data['video_tensor'])
            except Exception as e:
                video_tensor = self.nii_img_to_tensor(nii_file,split)
                os.makedirs(npz_base_path, exist_ok=True)
                np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())
        else:
            video_tensor = self.nii_img_to_tensor(nii_file,split)
            os.makedirs(npz_base_path, exist_ok=True)
            np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())

        # video_tensor = self.nii_img_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text




# nlst+oridata and process
class CTReportDataset_nlst_orianything(Dataset):
    def __init__(self, data_folder, reports_file, meta_file,nlst_path,nlst_num=100, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.nlst_path = nlst_path
        self.nlst_num = nlst_num
        self.min_slices = min_slices
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.nlst_samples = self.load_info(nlst_path)
        percent = 100
        num_files = int((len(self.nlst_samples) * percent) / 100)
        #num_files = 2286
        self.nlst_samples = self.nlst_samples[:num_files]
        self.samples = self.samples + self.nlst_samples
        print(len(self.samples))
        self.count = 0

        df = pd.read_csv(meta_file) #select the metadata
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']

        return accession_to_text


    def load_info(self, nlst_path):
        """从 JSON 加载数据"""
        import json
        import os
        

        samples = []
        self.reports = []
        self.image_paths = []
        
        # 读取 JSON
        with open(nlst_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        base_radgenemo_path = "../dataset/radgenome_chestct/dataset/train_preprocessed/"
        base_nlst_path = "../NLST_convert_v1/"
        
        # 处理每个条目
        split = ''
        for case_id, report_list in data_dict.items():
            if case_id.startswith('train') or case_id.startswith('val'):
                split = 'radgenome'
            else:
                split = 'nlst'

            if split == 'radgenome':
                case_id_split = case_id.split('_')
                image_path = base_radgenemo_path + 'train_' + case_id_split[1] + '/train_' + case_id_split[1] + case_id_split[2] + '/' + case_id
            if split == 'nlst':
                image_path = os.path.join(base_nlst_path, f"{case_id}.nii.gz")
            
            # 检查文件存在
            if not os.path.exists(image_path):
                continue
            
            # 提取报告（list 的第一个元素）
            report = report_list[0] if isinstance(report_list, list) and len(report_list) > 0 else ""
            samples.append((image_path, report))
            self.reports.append(report)
            self.image_paths.append(image_path)
        
        print(f"加载了 {len(self.reports)} 个样本")
        
        if len(self.reports) == 0:
            raise ValueError("没有找到任何有效样本！")

        return samples
        
    def prepare_samples(self):
        samples = []
        for nii_file in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*.nii.gz'))):
            accession_number = nii_file.split("/")[-1]
            #accession_number = accession_number.replace(".npz", ".nii.gz")
            if accession_number not in self.accession_to_text:
                ipdb.set_trace()
                continue

            impression_text = self.accession_to_text[accession_number]
            # ipdb.set_trace()

            if impression_text == "Not given.":
                impression_text=""

            input_text_concat = ""
            for text in impression_text:
                input_text_concat = input_text_concat + str(text)
            input_text_concat = impression_text[0]
            input_text = f'{impression_text}'
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
        return samples

    def __len__(self):
        return len(self.samples)



    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        img_data = slope * img_data + intercept

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data ) / 1000)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        return tensor

    def nii_img_to_tensor_nlst(self,path):
        """直接从 NIfTI 文件获取所有需要的信息"""
        
        # 加载 NIfTI 文件
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        header = nii_img.header
        
        # 从 header 获取 spacing
        pixdim = header['pixdim'][1:4]  # [x_spacing, y_spacing, z_spacing]
        x_spacing = float(pixdim[0])
        y_spacing  = float(pixdim[1])
        z_spacing = float(pixdim[2])
        
        # 从 header 获取 slope 和 intercept
        # slope = float(header.get('scl_slope', 1.0))
        # intercept = float(header.get('scl_inter', 0.0))
        
        # 如果 slope 为 0，设置为 1（避免除零）
        # if slope == 0:
        #     slope = 1.0
        
        # 目标 spacing
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        
        current = (z_spacing, x_spacing, y_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        # ipdb.set_trace()
        # # 应用 slope 和 intercept
        # img_data = slope * img_data + intercept
        # ipdb.set_trace()
        
        # 转置
        img_data = img_data.transpose(2, 0, 1)
        
        # 转为 tensor 并 resize
        tensor = torch.tensor(img_data, dtype=torch.float32)
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))
        
        # HU 窗口
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        
        # 归一化
        img_data = (img_data / 1000).astype(np.float32)
        
        # 转为 tensor
        tensor = torch.tensor(img_data)
        
        # Crop/Pad 到目标大小
        target_shape = (480, 480, 240)
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        
        # Crop
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]
        
        # Pad
        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before
        
        tensor = torch.nn.functional.pad(
            tensor,
            (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
            value=-1
        )
        
        # Permute
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)
        
        return tensor


    def __getitem__(self, index):
        nii_file, input_text = self.samples[index]
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")

        split = ''
        if name_acc.startswith('train') or name_acc.startswith('val'):
            split = 'ctrate'
        else:
            split = 'nlst'

        npz_base_path = "../dataset/ctrate/ctrate_ori_pretrain"
        npz_file = os.path.join(npz_base_path, f"{name_acc}.npz")
        if os.path.exists(npz_file):
            try:
                data = np.load(npz_file)
                video_tensor = torch.from_numpy(data['video_tensor'])
            except Exception as e:
                if split == 'ctrate':
                    video_tensor = self.nii_to_tensor(nii_file)
                if split == 'nlst':
                    video_tensor = self.nii_img_to_tensor_nlst(nii_file)
                os.makedirs(npz_base_path, exist_ok=True)
                np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())
        else:
            if split == 'ctrate':
                video_tensor = self.nii_to_tensor(nii_file)
            if split == 'nlst':
                video_tensor = self.nii_img_to_tensor_nlst(nii_file)
            os.makedirs(npz_base_path, exist_ok=True)
            np.savez_compressed(npz_file, video_tensor=video_tensor.numpy())
        # video_tensor = self.nii_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text
