import os
import glob
import numpy as np
import torch
import nibabel as nib
import pandas as pd
from functools import partial
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 假设你有 resize_array 函数
def resize_array(tensor, current_spacing, target_spacing):
    """你的 resize_array 实现"""
    # 这里需要你的实际实现
    import torch.nn.functional as F
    
    # 计算缩放因子
    scale_factors = [c / t for c, t in zip(current_spacing, target_spacing)]
    
    # 使用三线性插值
    tensor = F.interpolate(
        tensor,
        scale_factor=scale_factors,
        mode='trilinear',
        align_corners=False
    )
    
    return tensor


def process_single_nii(args):
    """
    处理单个 NIfTI 文件
    
    Args:
        args: (nii_file, output_dir, meta_df)
    
    Returns:
        (success, nii_file, message)
    """
    nii_file, output_dir, meta_df = args
    
    try:
        # 获取文件名
        file_name = os.path.basename(nii_file)
        name_acc = file_name.replace(".nii.gz", "")
        npz_file = os.path.join(output_dir, f"{name_acc}.npz")
        
        # 如果已存在，跳过
        if os.path.exists(npz_file):
            return (True, nii_file, "Already exists")
        
        # 读取 NIfTI
        nii_img = nib.load(nii_file)
        img_data = nii_img.get_fdata()
        
        # 获取 metadata
        row = meta_df[meta_df['VolumeName'] == file_name]
        if len(row) == 0:
            return (False, nii_file, "Not found in metadata")
        
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
        z_spacing = float(row["ZSpacing"].iloc[0])
        
        # 目标 spacing
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5
        
        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)
        
        # 应用 slope 和 intercept
        img_data = slope * img_data + intercept
        
        # 转置
        img_data = img_data.transpose(2, 0, 1)
        
        # 转为 tensor 并 resize
        tensor = torch.tensor(img_data).unsqueeze(0).unsqueeze(0)
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0].numpy()
        
        # 再次转置
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
        
        # 计算 crop
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)
        
        # Crop
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
        
        # 保存为 NPZ
        np.savez_compressed(npz_file, video_tensor=tensor.numpy())
        
        return (True, nii_file, "Success")
        
    except Exception as e:
        return (False, nii_file, f"Error: {str(e)}")


def batch_process_nii_to_npz(
    data_folder,
    meta_file,
    output_dir,
    num_workers=16
):
    """
    并行处理所有 NIfTI 文件
    
    Args:
        data_folder: NIfTI 文件所在目录
        meta_file: metadata CSV 文件
        output_dir: 输出 NPZ 文件目录
        num_workers: 并行进程数
    """
    
    print("=" * 80)
    print("批量处理 NIfTI 文件")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取 metadata
    print(f"读取 metadata: {meta_file}")
    meta_df = pd.read_csv(meta_file)
    print(f"✓ Metadata 行数: {len(meta_df)}")
    
    # 查找所有 NIfTI 文件
    print(f"\n扫描目录: {data_folder}")
    nii_files = glob.glob(os.path.join(data_folder, "*.nii.gz"))
    print(f"✓ 找到 {len(nii_files)} 个 NIfTI 文件")
    
    # 检查已存在的文件
    existing_files = set(
        f.replace('.npz', '.nii.gz')
        for f in glob.glob(os.path.join(output_dir, "*.npz"))
    )
    
    # 过滤掉已处理的文件
    nii_files_to_process = [
        f for f in nii_files
        if os.path.basename(f) not in existing_files
    ]
    
    print(f"✓ 已存在: {len(existing_files)} 个")
    print(f"✓ 待处理: {len(nii_files_to_process)} 个")
    
    if len(nii_files_to_process) == 0:
        print("\n所有文件已处理完成！")
        return
    
    # 准备任务参数
    tasks = [(nii_file, output_dir, meta_df) for nii_file in nii_files_to_process]
    
    # 并行处理
    print(f"\n开始处理（{num_workers} 个进程）...")
    
    success_count = 0
    failed_count = 0
    failed_files = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_nii, task): task for task in tasks}
        
        # 使用 tqdm 显示进度
        with tqdm(total=len(futures), desc="处理进度") as pbar:
            for future in as_completed(futures):
                success, nii_file, message = future.result()
                
                if success:
                    success_count += 1
                else:
                    failed_count += 1
                    failed_files.append((nii_file, message))
                
                pbar.update(1)
                pbar.set_postfix({
                    '成功': success_count,
                    '失败': failed_count
                })
    
    # 打印结果
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"总文件数: {len(nii_files_to_process)}")
    print(f"成功:     {success_count} ({success_count/len(nii_files_to_process)*100:.1f}%)")
    print(f"失败:     {failed_count} ({failed_count/len(nii_files_to_process)*100:.1f}%)")
    
    # 保存失败列表
    if failed_files:
        failed_log = os.path.join(output_dir, "failed_files.txt")
        with open(failed_log, 'w') as f:
            for nii_file, message in failed_files:
                f.write(f"{nii_file}\t{message}\n")
        
        print(f"\n失败文件列表保存到: {failed_log}")
        print("\n失败的文件（前 10 个）:")
        for nii_file, message in failed_files[:10]:
            print(f"  {os.path.basename(nii_file)}: {message}")
    
    print("=" * 80)


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    # 配置参数
    data_folder = "../data/CT-RATE/data/train"
    meta_file = "../data/CT-RATE/ct_rate_meta/train_metadata.csv"  # 修改为你的 metadata 路径
    output_dir = "../dataset/ctrate/ctrate_ori_pretrain"
    num_workers = 8  # 根据你的 CPU 核心数调整
    
    # 运行批处理
    batch_process_nii_to_npz(
        data_folder=data_folder,
        meta_file=meta_file,
        output_dir=output_dir,
        num_workers=num_workers
    )
