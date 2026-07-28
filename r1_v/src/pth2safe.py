import torch
from collections import OrderedDict
import os

def convert_checkpoint_keys(input_path, output_dir):
    """
    Loads a checkpoint, modifies its state_dict keys, and saves it in a
    transformers-compatible format.

    Args:
        input_path (str): Path to the input .pth checkpoint file.
        output_dir (str): Directory where the new 'pytorch_model.bin' will be saved.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
        print(f"Loading checkpoint from '{input_path}' to device '{device}'...")
        
        original_state_dict = torch.load(input_path, map_location=torch.device(device))['model']
        
        print("Checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
        return
    except KeyError:
        print("Error: The loaded checkpoint does not contain a 'model' key.")
        return

    new_state_dict = OrderedDict()

    print("Starting key conversion...")
    
    for key, value in original_state_dict.items():
        new_key = key
        
        if key.startswith("language_model."):
            new_key = key[15:]
            print(f"  '{key}' -> '{new_key}'")

        elif key.startswith("llama_proj.") or key.startswith("layer_norm."):
            new_key = f"perceiver.{key}"
            print(f"  '{key}' -> '{new_key}'")
            
        else:
            pass

        new_state_dict[new_key] = value

    print("Key conversion finished.")

    output_path = os.path.join(output_dir, "pytorch_model.bin")
    print(f"Saving converted state_dict to '{output_path}'...")
    torch.save(new_state_dict, output_path)

    print("Conversion complete!")
    print(f"The converted model weights are saved at: {output_path}")
    print(f"You can now try to load this model using `YourModelClass.from_pretrained('{output_dir}')`.")
    print("Remember to also copy the necessary config.json and tokenizer files to the output directory.")


if __name__ == '__main__':

    input_checkpoint_path = "../merged_dict.pth"  # merge lora results
    output_model_dir = "../" #
    convert_checkpoint_keys(input_checkpoint_path, output_model_dir)

