import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import torch
import torch.nn as nn
# import torch.nn.functional as F
# import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer, Qwen2_5_VLConfig
from transformers.configuration_utils import PretrainedConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, PreTrainedModel
# from transformers import SwinModel
# from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import ipdb
from utils import PerceiverResampler
from transformers import get_cosine_schedule_with_warmup
from merlin import Merlin_modified
os.environ["TOKENIZERS_PARALLELISM"] = "true"

class swiglu(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x = swish * self.w3(x)
        x = self.w2(x)
        return x


class R2GenGPT(nn.Module):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.target_dtype = torch.bfloat16
        # self.visual_encoder = Merlin_modified(ImageEmbedding=True)

        print('Loading qwen2.5_vl')
        self.qwen_processor = AutoProcessor.from_pretrained("../hf/qwen25_vl")
        self.tokenizer = self.qwen_processor.tokenizer
        print("DEBUG: Attempting to load model with local_files_only=True")
        self.language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("../base_e10_trans", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        self.config = self.language_model.config
        print('Loading LLAMA Done, full tune')
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest CT image.'
        # self.prompt = "Generate a comprehensive and detailed diagnosis report for this abdomen CT image. Structure the report by describing the following regions in this exact order: lower thorax, liver and biliary tree, gallbladder, spleen, pancreas, adrenal glands, kidneys and ureters, gastrointestinal tract, peritoneum, pelvic, vasculature, lymph nodes, musculoskeletal. For any region without abnormalities, state 'normal.'."
        self.chest_prompt = "Generate a comprehensive and detailed diagnosis report for this chest CT image. Structure the report by describing the following regions in this exact order: abdomen, bone, breast, esophagus, heart, lung, mediastinum, pleura, thyroid, trachea and bronchie. For any region without abnormalities, state 'normal.'."
        self.merlin_prompt = "Generate a comprehensive and detailed diagnosis report for this abdomen CT image. Structure the report by describing the following regions in this exact order: lower thorax, liver and biliary tree, gallbladder, spleen, pancreas, adrenal glands, kidneys and ureters, gastrointestinal tract, peritoneum, pelvic, vasculature, lymph nodes, musculoskeletal. For any region without abnormalities, state 'normal.'."
        self.atlas_prompt = "Please analyze the liver and biliary tree, pancreas, and kidneys and ureters areas from this abdominal CT scan. For any region without abnormalities, state 'normal.'."
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0




    def encode_img(self, images):
        # x_processed = images.permute(0, 1, 4, 2, 3) # 变成 (B, 1, W, T, H)? -> 你的原始permute是(0,1,4,2,3)，请确认维度含义。假设 T,H,W -> T,W,H
                                                   # 常见的医学图像是 (B, C, D, H, W) 或 (B, D, H, W, C), 假设你的输入是 (B, C, H, W, D) with C=1
                                                   # 那么 permute(0, 1, 4, 2, 3) -> (B, C, D, H, W)
                                                   # 让我们假设输入是 (B, 1, H, W, D)，目标是 (B, 3, D, H, W)
        processed_image = images.permute(0, 1, 4, 2, 3)  # -> (B, 1, D, H, W)
        # processed_image = torch.cat((processed_image, processed_image, processed_image), dim=1)
        processed_image = processed_image.repeat(1, 3, 1, 1, 1) 


        image_embeds = self.language_model.visual_encoder(processed_image)
        b, c, x, y, z = image_embeds.shape
        image_embeds = image_embeds.permute(0, 2, 3, 4, 1)
        image_embeds = image_embeds.reshape(b, x*y*z, c)
        # if self.args.whether_perceiver is True:
        # image_embeds = self.perceiver(image_embeds.unsqueeze(1).unsqueeze(1)).squeeze(1)
        # inputs_llama = self.llama_proj(image_embeds)
        inputs_llama = self.language_model.perceiver(image_embeds.unsqueeze(1).unsqueeze(1))
        # inputs_llama = self.language_model.layer_norm(self.language_model.perceiver(image_embeds.unsqueeze(1).unsqueeze(1)))
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama, atts_llama
    
   
    def infer(self, samples):
        # ipdb.set_trace()
        image = samples["image"].unsqueeze(0).to(torch.bfloat16)
        img_embeds, atts_img = self.encode_img(image)
        # img_embeds = self.layer_norm(img_embeds)  # layernorm修改
        batch_size = 1
        prompt = []
        for b in range(batch_size):
            if self.args.prompt_style == "lung":
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.prompt}<|im_end|>\n<|im_start|>assistant\n')
            else:
                dataset = samples['dataset']
                if dataset == 'merlin':
                    prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.merlin_prompt}<|im_end|>\n<|im_start|>assistant\n')
                if dataset == 'atlas':
                    prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.atlas_prompt}<|im_end|>\n<|im_start|>assistant\n')
                if dataset in ['ct_rate','inspect','bimcv']:
                    prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.chest_prompt}<|im_end|>\n<|im_start|>assistant\n')

        pad_size = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        to_regress_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=False,
            # max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)
        to_regress_embeds = self.language_model.model.embed_tokens(to_regress_tokens.input_ids)
        to_regress_embeds_writable = to_regress_embeds.clone()
        self.tokenizer.padding_side = pad_size

        # 替换<|image_pad|>为img_embeds
        image_pad_id = 151655
        for b in range(batch_size):
            image_pad_positions = (to_regress_tokens.input_ids[b] == image_pad_id).nonzero().squeeze()
            if len(image_pad_positions) >= 32:
                start_pos = image_pad_positions[0]
                to_regress_embeds_writable[b, start_pos:start_pos+32] = img_embeds[b]
            else:
                print(f"Warning: Sample {b} does not have enough image_pad tokens.")

        wrapped_img_embeds = to_regress_embeds_writable
        wrapped_atts_img = to_regress_tokens.attention_mask
        return wrapped_img_embeds, wrapped_atts_img, to_regress_tokens.input_ids

    def prompt_forward(self, input_ids,samples,num_generations):
        # ipdb.set_trace()
        image = samples["image"].unsqueeze(0).to(torch.bfloat16).repeat_interleave(num_generations, dim=0)
        # ipdb.set_trace()
        img_embeds, atts_img = self.encode_img(image)
        # img_embeds = self.layer_norm(img_embeds)  # layernorm修改
        to_regress_embeds = self.language_model.model.embed_tokens(input_ids)
        to_regress_embeds_writable = to_regress_embeds.clone()

        # 替换<|image_pad|>为img_embeds
        image_pad_id = 151655
        for b in range(num_generations):
            image_pad_positions = (input_ids[b] == image_pad_id).nonzero().squeeze()
            if len(image_pad_positions) >= 32:
                start_pos = image_pad_positions[0]
                to_regress_embeds_writable[b, start_pos:start_pos+32] = img_embeds[b]
            else:
                print(f"Warning: Sample {b} does not have enough image_pad tokens.")

        wrapped_img_embeds = to_regress_embeds_writable
        return wrapped_img_embeds


    def forward(self, input_ids, attention_mask, samples, num_generations):
        inputs_embeds_final = self.prompt_forward(input_ids,samples, num_generations)
        outputs = self.language_model(
            inputs_embeds=inputs_embeds_final,
            attention_mask=attention_mask,
            use_cache = False,
            return_dict=True,
        )
        # ipdb.set_trace()
        return outputs
    
    
    def generate(self, samples, num_generations,max_completion_length):
        img_embeds, atts_img, input_ids = self.infer(samples)

        inputs_embeds = img_embeds
        attention_mask = atts_img

        # print(inputs_embeds.shape)
        # self.tokenizer.padding_side = "left"
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            do_sample=True,
            max_new_tokens=max_completion_length,
            temperature=1,
            num_return_sequences = num_generations,
        )
        # self.tokenizer.padding_side = "right"
        # ipdb.set_trace()
        # hypo = self.qwen_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return outputs,input_ids, attention_mask,inputs_embeds

    

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel,RobertaModel
from scipy.special import expit
class RadBertClassifier(nn.Module):
    def __init__(self,n_classes=10):
      super().__init__()
    
      self.config = AutoConfig.from_pretrained('../hf/reg2rg/RadBERT-RoBERTa-4m')
      self.model = RobertaModel(config=self.config)
    
      self.classifier=nn.Linear(self.model.config.hidden_size,n_classes) 
      self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def forward(self,input_ids, attn_mask):
      if len(input_ids.size()) == 1:
        input_ids = input_ids.unsqueeze(0)
        attn_mask = attn_mask.unsqueeze(0)
      output = self.model(input_ids=input_ids,attention_mask=attn_mask)
      output = self.classifier(output.pooler_output)
            
      return output

class RadBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RadBertClassifier(18)
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model_path = "../hf/reg2rg/RadBertClassifier.pth"
        pth = torch.load(model_path, map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.load_state_dict(pth,strict=False)
        self.model.eval()
        
    def forward(self,input_ids, attn_mask):
        out = self.model(input_ids,attn_mask).detach().cpu().numpy()
        pred_labels = expit(out)
        pred_labels[pred_labels>=0.5]=1
        pred_labels[pred_labels<0.5]=0
            
        return pred_labels

