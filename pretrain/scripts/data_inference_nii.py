import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import tqdm
import nibabel as nib
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



class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, min_slices=20, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        df = pd.read_csv(meta_file) #select the metadata
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)
        # ipdb.set_trace()

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)
        # ipdb.set_trace()

        # for patient_folder in tqdm.tqdm(patient_folders):
        #     accession_folders = glob.glob(os.path.join(patient_folder, '*'))
        #     ipdb.set_trace()

        #     for accession_folder in accession_folders:
        #         nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))
        #         ipdb.set_trace()

        #         for nii_file in nii_files:
        #             accession_number = nii_file.split("/")[-1]

        #             if accession_number not in self.accession_to_text:
        #                 continue

        #             impression_text = self.accession_to_text[accession_number]
        #             text_final = ""
        #             for text in list(impression_text):
        #                 text = str(text)
        #                 if text == "Not given.":
        #                     text = ""

        #                 text_final = text_final + text

        #             onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
        #             if len(onehotlabels) > 0:
        #                 samples.append((nii_file, text_final, onehotlabels[0]))
        #                 self.paths.append(nii_file)

        for patient_folder in tqdm.tqdm(patient_folders):
            nii_file = patient_folder
            accession_number = nii_file.split("/")[-1]

            rere_id = accession_number.split("_")[-1].split(".")[0]
            # ipdb.set_trace()

            if rere_id != '1':
                continue

            if accession_number not in self.accession_to_text:
                continue

            impression_text = self.accession_to_text[accession_number]
            text_final = ""
            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""

                text_final = text_final + text

            onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
            if len(onehotlabels) > 0:
                samples.append((nii_file, text_final, onehotlabels[0]))
                self.paths.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        row = df[df['VolumeName'] == file_name]
        # ipdb.set_trace()
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
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

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
        nii_file, input_text, onehotlabels = self.samples[index]
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")
        npz_base_path = "../dataset/ctrate/ctrate_classification"
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
        
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        return video_tensor, input_text, onehotlabels, name_acc





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
                
                # 判断文件类型
                if file_path.endswith('.npz'):
                    # 加载 NPZ 文件（RAD-ChestCT）
                    with np.load(file_path) as npz_data:
                        arr = npz_data[self.npz_key]
                
                elif file_path.endswith('.npy'):
                    # 加载 NPY 文件（NLST）
                    arr = np.load(file_path)
                    
                    # 如果是 uint16，转换为 HU 值
                    if arr.dtype == np.uint16:
                        arr = arr.astype(np.float32) - self.offset
                        # print(f"检测到 uint16 数据，已转换为 HU 值 (减去 {self.offset})")
                
                else:
                    raise ValueError(f"不支持的文件格式: {file_path}")
            else:
                arr = d[key]
            
            # 确保是 float32
            arr = arr.astype(np.float32)
            
            # 添加通道维度：(D, H, W) -> (1, D, H, W)
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
class my_CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, min_slices=20, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.reports =[]
        self.image_paths = []
        self.label_list = []
        self.count = 0
        self.load_info(self.data_folder,labels)

    def __len__(self):
        return len(self.reports)

    def load_info(self, data_folder, labels):
        """从 JSON 加载数据，从 CSV 加载标签"""
        import json
        import os
        import pandas as pd
        import numpy as np
        
        self.reports = []
        self.image_paths = []
        self.label_list = []
        
        print("=" * 80)
        print("加载数据和标签")
        print("=" * 80)
        
        # 读取 JSON
        with open(data_folder, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # 读取 CSV
        labels_df = pd.read_csv(labels)
        
        # ========== 关键修改：统一 ID 格式 ==========
        # 去除前导零，统一为整数字符串格式
        def normalize_id(id_val):
            """统一 ID 格式：去除前导零"""
            try:
                return str(int(id_val))
            except:
                return str(id_val)
        
        # 标准化 CSV 中的 AccNum
        first_col = labels_df.columns[0]  # 假设是 'AccNum'
        labels_df[first_col] = labels_df[first_col].apply(normalize_id)
        
        # Pred 列名
        pred_cols = [
            'Pred_Medical material',
            'Pred_Arterial wall calcification',
            'Pred_Cardiomegaly',
            'Pred_Pericardial effusion',
            'Pred_Coronary artery wall calcification',
            'Pred_Hiatal hernia',
            'Pred_Lymphadenopathy',
            'Pred_Emphysema',
            'Pred_Atelectasis',
            'Pred_Lung nodule',
            'Pred_Lung opacity',
            'Pred_Pulmonary fibrotic sequela',
            'Pred_Pleural effusion',
            'Pred_Mosaic attenuation pattern',
            'Pred_Peribronchial thickening',
            'Pred_Consolidation',
            'Pred_Bronchiectasis',
            'Pred_Interlobular septal thickening'
        ]
        
        # 检查列是否存在
        missing_cols = [col for col in pred_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"CSV 中缺失以下列: {missing_cols}")
        
        # 创建标签映射（使用标准化的 ID）
        accnum_to_labels = {}
        for _, row in labels_df.iterrows():
            acc_num = row[first_col]  # 已经标准化过
            label_values = row[pred_cols].values.astype(np.float32)
            accnum_to_labels[acc_num] = label_values
        
        print(f"  CSV 标签映射: {len(accnum_to_labels)} 个")
        
        # 基础路径
        base_path = "../mnt_det_bak/datasets/Med_datasets/NLST/jhcnas3/CTs/data/NLST_convert_v1/"
        
        # 统计
        matched_count = 0
        missing_file_count = 0
        missing_label_count = 0
        
        # 处理每个条目
        for case_id, report_list in data_dict.items():
            # ========== 关键：标准化 JSON 中的 case_id ==========
            case_id_normalized = normalize_id(case_id)
            
            # 构建图像路径（使用原始 case_id，保留前导零）
            image_path = os.path.join(base_path, f"{case_id}.npy")
            
            # 检查文件
            if not os.path.exists(image_path):
                missing_file_count += 1
                if missing_file_count <= 3:
                    print(f"  ⚠ 图像不存在: {case_id}")
                continue
            
            # 检查标签（使用标准化的 ID）
            if case_id_normalized not in accnum_to_labels:
                missing_label_count += 1
                if missing_label_count <= 3:
                    print(f"  ⚠ 标签缺失: {case_id} (标准化: {case_id_normalized})")
                continue
            
            # 提取报告
            report = report_list[0] if isinstance(report_list, list) and len(report_list) > 0 else ""
            
            # 获取标签
            label = accnum_to_labels[case_id_normalized]
            
            # 添加到列表
            self.reports.append(report)
            self.image_paths.append(image_path)
            self.label_list.append(label)
            matched_count += 1
        
        # 统计信息
        print("\n" + "=" * 80)
        print("加载统计:")
        print(f"  JSON 条目: {len(data_dict)}")
        print(f"  CSV 标签: {len(labels_df)}")
        print(f"  成功匹配: {matched_count}")
        print(f"  缺失图像: {missing_file_count}")
        print(f"  缺失标签: {missing_label_count}")
        print(f"  最终样本: {len(self.reports)}")
        print("=" * 80)
        
        # 示例
        if len(self.reports) > 0:
            print("\n样本示例:")
            sample_id = list(data_dict.keys())[0]
            print(f"  原始 ID: {sample_id}")
            print(f"  标准化 ID: {normalize_id(sample_id)}")
            print(f"  图像路径: {self.image_paths[0]}")
            print(f"  标签: {self.label_list[0]}")
            print(f"  报告: {self.reports[0][:100]}...")
        
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
        onehotlabels = self.label_list[index]
        video_tensor = self.nii_img_to_tensor(nii_file)
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')
        name_acc = nii_file.split("/")[-1].replace(".npy", "")
        return video_tensor, input_text, onehotlabels, name_acc














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
class my_ctrate_CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, min_slices=20, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.reports =[]
        self.image_paths = []
        self.label_list = []
        self.count = 0
        self.load_info(self.data_folder,labels)

    def __len__(self):
        return len(self.reports)

    def load_info(self, data_folder, labels):
        """从 JSON 加载数据，从 CSV 加载标签"""
        import json
        import os
        import pandas as pd
        import numpy as np
        
        self.reports = []
        self.image_paths = []
        self.label_list = []
        
        print("=" * 80)
        print("加载数据和标签")
        print("=" * 80)
        
        # 读取 JSON
        with open(data_folder, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        
        # 读取 CSV
        labels_df = pd.read_csv(labels)
        
        # ========== 关键修改：统一 ID 格式 ==========
        # 去除前导零，统一为整数字符串格式
        def normalize_id(id_val):
            """统一 ID 格式：去除前导零"""
            try:
                return str(int(id_val))
            except:
                return str(id_val)
        
        # 标准化 CSV 中的 AccNum
        first_col = labels_df.columns[0]  # 假设是 'AccNum'
        labels_df[first_col] = labels_df[first_col].apply(normalize_id)
        
        # Pred 列名
        pred_cols = [
            'Pred_Medical material',
            'Pred_Arterial wall calcification',
            'Pred_Cardiomegaly',
            'Pred_Pericardial effusion',
            'Pred_Coronary artery wall calcification',
            'Pred_Hiatal hernia',
            'Pred_Lymphadenopathy',
            'Pred_Emphysema',
            'Pred_Atelectasis',
            'Pred_Lung nodule',
            'Pred_Lung opacity',
            'Pred_Pulmonary fibrotic sequela',
            'Pred_Pleural effusion',
            'Pred_Mosaic attenuation pattern',
            'Pred_Peribronchial thickening',
            'Pred_Consolidation',
            'Pred_Bronchiectasis',
            'Pred_Interlobular septal thickening'
        ]
        
        # 检查列是否存在
        missing_cols = [col for col in pred_cols if col not in labels_df.columns]
        if missing_cols:
            raise ValueError(f"CSV 中缺失以下列: {missing_cols}")
        
        # 创建标签映射（使用标准化的 ID）
        accnum_to_labels = {}
        for _, row in labels_df.iterrows():
            acc_num = row[first_col]  # 已经标准化过
            label_values = row[pred_cols].values.astype(np.float32)
            accnum_to_labels[acc_num] = label_values
        
        print(f"  CSV 标签映射: {len(accnum_to_labels)} 个")
        
        # 基础路径
        base_radgenemo_path = "../dataset/radgenome_chestct/dataset/valid_preprocessed/"
        base_nlst_path = "../NLST_convert_v1/"
        
        # 统计
        matched_count = 0
        missing_file_count = 0
        missing_label_count = 0
        
        # 处理每个条目
        split = ''
        for case_id, report_list in data_dict.items():
            # ========== 关键：标准化 JSON 中的 case_id ==========
            case_id_normalized = normalize_id(case_id)
            
            # 构建图像路径（使用原始 case_id，保留前导零）
            if case_id.startswith('train') or case_id.startswith('val'):
                split = 'radgenome'
            else:
                split = 'nlst'

            if split == 'radgenome':
                case_id_split = case_id.split('_')
                image_path = base_radgenemo_path + 'valid_' + case_id_split[1] + '/valid_' + case_id_split[1] + case_id_split[2] + '/' + case_id
            if split == 'nlst':
                image_path = os.path.join(base_nlst_path, f"{case_id}.nii.gz")

            # case_id_split = case_id.split('_')
            # image_path = base_path + 'valid_' + case_id_split[1] + '/valid_' + case_id_split[1] + case_id_split[2] + '/' + case_id
            # ipdb.set_trace()
            # image_path = os.path.join(base_path, f"{case_id}.npy")
            
            # 检查文件
            if not os.path.exists(image_path):
                missing_file_count += 1
                if missing_file_count <= 3:
                    print(f"  ⚠ 图像不存在: {case_id}")
                continue
            
            # 检查标签（使用标准化的 ID）
            if case_id_normalized not in accnum_to_labels:
                missing_label_count += 1
                if missing_label_count <= 3:
                    print(f"  ⚠ 标签缺失: {case_id} (标准化: {case_id_normalized})")
                continue
            
            # 提取报告
            report = report_list[0] if isinstance(report_list, list) and len(report_list) > 0 else ""
            
            # 获取标签
            label = accnum_to_labels[case_id_normalized]
            
            # 添加到列表
            self.reports.append(report)
            self.image_paths.append(image_path)
            self.label_list.append(label)
            matched_count += 1
        
        # 统计信息
        print("\n" + "=" * 80)
        print("加载统计:")
        print(f"  JSON 条目: {len(data_dict)}")
        print(f"  CSV 标签: {len(labels_df)}")
        print(f"  成功匹配: {matched_count}")
        print(f"  缺失图像: {missing_file_count}")
        print(f"  缺失标签: {missing_label_count}")
        print(f"  最终样本: {len(self.reports)}")
        print("=" * 80)
        
        # 示例
        if len(self.reports) > 0:
            print("\n样本示例:")
            sample_id = list(data_dict.keys())[0]
            print(f"  原始 ID: {sample_id}")
            print(f"  标准化 ID: {normalize_id(sample_id)}")
            print(f"  图像路径: {self.image_paths[0]}")
            print(f"  标签: {self.label_list[0]}")
            print(f"  报告: {self.reports[0][:100]}...")
        
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
        onehotlabels = self.label_list[index]
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
        # name_acc = nii_file.split("/")[-1].replace(".npy", "")
        return video_tensor, input_text, onehotlabels, name_acc