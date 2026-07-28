import torch
from collections import OrderedDict
import os
from safetensors import safe_open
import ipdb
def convert_safetensors_keys(input_path, output_path):
    """
    Loads a .safetensors file, modifies its state_dict keys based on specific rules,
    and saves the result as a .pth file.

    Args:
        input_path (str): Path to the input .safetensors file.
        output_path (str): Path where the new .pth file will be saved.
    """
    print(f"正在读取 safetensors 文件: {input_path}")

    new_state_dict = OrderedDict()

    try:
        with safe_open(input_path, framework="pt", device="cpu") as f:
            
            print("文件打开成功。开始转换键名...")
            
            for key in f.keys():
                new_key = key
                
                if key.startswith("language_model.perceiver.llama_proj."):
                    new_key = key.replace("language_model.perceiver.llama_proj.", "llama_proj.", 1)
                    print(f"  应用规则 (c): '{key}' -> '{new_key}'")
                
                elif key.startswith("language_model.perceiver.layer_norm."):
                    new_key = key.replace("language_model.perceiver.layer_norm.", "layer_norm.", 1)
                    print(f"  应用规则 (d): '{key}' -> '{new_key}'")

                elif key.startswith("language_model.visual_encoder."):
                    new_key = key.replace("language_model.visual_encoder.", "visual_encoder.", 1)
                    print(f"  应用规则 (a): '{key}' -> '{new_key}'")
                
                elif key.startswith("language_model.perceiver."):
                    new_key = key.replace("language_model.perceiver.", "perceiver.", 1)
                    print(f"  应用规则 (b): '{key}' -> '{new_key}'")
                
                else:
                    # print(f"  保持不变: '{key}'")
                    pass
                
                new_state_dict[new_key] = f.get_tensor(key)

        print("键名转换完成。")

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        print(f"正在将转换后的 state_dict 保存到: {output_path}")
        torch.save(new_state_dict, output_path)

        print("转换成功完成！")

    except FileNotFoundError:
        print(f"错误: 文件未找到 at '{input_path}'")
    except Exception as e:
        print(f"发生未知错误: {e}")


if __name__ == '__main__':

    input_safetensors_file = "../model.safetensors"
    output_pth_file = "../converted_weights.pth" # <--- 你可以修改为你想要的输出文件名和路径
    convert_safetensors_keys(input_safetensors_file, output_pth_file)


