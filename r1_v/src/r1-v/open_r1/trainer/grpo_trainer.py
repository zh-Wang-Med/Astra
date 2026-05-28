# forte reward
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoModel,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
import torch.nn as nn
from scipy.special import expit
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from sklearn.metrics import multilabel_confusion_matrix
from my_model import RadBertClassifier,RadBert,R2GenGPT
import ipdb
import numpy as np
import copy
import deepspeed
import torch.distributed as dist
from merlin import Merlin_modified
from utils import PerceiverResampler
from scipy.special import expit
import wandb
import json
import re
wandb.login(key="f3bdeb60504fe5dbcd50a1fa4b2ae69b48a5a65d")

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
# --- 静态定义 ---
KEYWORD_CATEGORIES_all = ['degree', 'landmark', 'feature', 'impression']
KEYWORD_CATEGORIES_degree = ['degree']
# ANATOMICAL_REGIONS = [
#     "lung", "trachea and bronchie", "mediastinum", "heart",
#     "esophagus", "pleura", "bone", "thyroid", "abdomen", "others"
# ]

ANATOMICAL_REGIONS_merlin = [
    "lower thorax", "liver and biliary tree", "gallbladder", "spleen", 
    "pancreas", "adrenal glands", "kidneys and ureters", "gastrointestinal tract", "peritoneum", "pelvic","vasculature","lymph nodes","musculoskeletal"
]
ANATOMICAL_REGIONS_atlas = [
    "liver and biliary tree", "pancreas", "kidneys and ureters"
]

ANATOMICAL_REGIONS_chest = [
        "abdomen",
        "bone",
        "breast",
        "esophagus",
        "heart",
        "lung",
        "mediastinum",
        "pleura",
        "thyroid",
        "trachea and bronchie" 
    ]

def find_repeated_strings(text, min_length=5, min_repetitions=2):
    """
    检测文本中连续重复出现的字符串
    
    Args:
        text: 报告文本
        min_length: 最小字符串长度（太短的不检测，避免误报）
        min_repetitions: 最小重复次数
    
    Returns:
        list: 重复的字符串及其出现次数，格式为 [{'string': ..., 'count': ..., 'position': ..., 'length': ...}, ...]
    """
    # 用空格分词
    words = text.split()
    
    if len(words) < min_length * min_repetitions:
        return []
    
    repeated_patterns = []
    i = 0
    
    while i < len(words):
        # 尝试不同长度的模式
        for pattern_len in range(min_length, len(words) - i + 1):
            # 提取当前模式
            pattern = words[i:i + pattern_len]
            pattern_str = ' '.join(pattern)
            
            # 检查从当前位置开始，这个模式连续重复了多少次
            repetition_count = 1
            j = i + pattern_len
            
            while j + pattern_len <= len(words):
                next_segment = words[j:j + pattern_len]
                if next_segment == pattern:
                    repetition_count += 1
                    j += pattern_len
                else:
                    break
            
            # 如果满足最小重复次数要求
            if repetition_count >= min_repetitions:
                repeated_patterns.append({
                    'pattern': pattern_str,
                    'count': repetition_count,
                    'start_position': i,
                    'word_length': pattern_len
                })
                
                # 跳过已检测的重复部分
                i = j
                break
        else:
            # 如果没有找到重复模式，移动到下一个词
            i += 1
    
    # 去重并格式化结果（保留最长的重复模式）
    final_results = []
    used_positions = set()
    
    # 按照起始位置和长度排序，优先处理更长的模式
    repeated_patterns.sort(key=lambda x: (x['start_position'], -x['word_length']))
    
    for item in repeated_patterns:
        start = item['start_position']
        end = start + item['word_length'] * item['count']
        
        # 检查这个范围是否已被使用
        if not any(pos in used_positions for pos in range(start, end)):
            final_results.append({
                'string': item['pattern'],
                'count': item['count'],
                'position': item['start_position'],
                'length': len(item['pattern'])  # 字符长度
            })
            # 标记这些位置已使用
            used_positions.update(range(start, end))
    
    return final_results

class ComplexRewardCalculator:
    def __init__(self, keyword_json_path):
        """
        初始化奖励计算器。
        :param keyword_json_path: 包含关键词和同义词的JSON文件路径。
        :param device: PyTorch设备 (e.g., 'cuda:0' or 'cpu')。
        """
        # with open(keyword_json_path, 'r') as f:
        #     self.json_data = json.load(f)

        with open("../FORTE-main/data/FORTE_chestCT_icd_1126.json", 'r') as f:
            self.json_data_ctrate = json.load(f)
        with open("../FORTE-main/data/FORTE_inspect_icd1221.json", 'r') as f:
            self.json_data_inspect = json.load(f)
        with open("../FORTE-main/data/FORTE_bimcv_icd1221.json", 'r') as f:
            self.json_data_bimcv = json.load(f)
        with open("../FORTE-main/data/FORTE_abdomen_icd1208.json", 'r') as f:
            self.json_data_abodmen = json.load(f)
        print("ComplexRewardCalculator initialized.")

    # --- 辅助函数 (严格按照参考逻辑) ---
    def _extract_keywords_from_text(self, text, category_data):
        # 确保文本前后有空格，以便进行全词匹配
        text = f" {text.lower()} "
        keywords = set()
        for key in category_data:
            if key in text and ('without'+key) not in text:
                keywords.add(key.strip()) # 添加不带空格的关键词
        return list(keywords)

    def _map_to_synonyms(self, key, json_data):
        for main_key, synonyms in json_data.items():
            if key in synonyms:
                return set(synonyms)
        return {key}

    def _map_to_representative_synonym(self, key, category_data):
        # # 传入的key已经是strip过的
        # key_with_space = ' ' + key # 恢复用于匹配JSON的格式
        for main_key, synonyms in category_data.items():
            if key in synonyms:
                return main_key.strip()
        return key

    # --- 预处理和解析函数 ---
    def _preprocess_text(self, text):
        if not isinstance(text, str):
            return ""
        text = text.replace('\n', '').replace('\r', '').replace('normal.','normal')
        text = text.lower()
        text = re.sub(r'\bformal\b', 'normal', text)
        return text

    def _parse_structured_report(self, text,ANATOMICAL_REGIONS):
        report_dict = {}
        # 对文本进行预处理
        processed_text = self._preprocess_text(text)
        
        found_regions = []
        for region in ANATOMICAL_REGIONS:
            # 使用\b来确保匹配到的是完整的单词
            pattern = r'\b' + re.escape(region) + r'\b\s*:'
            match = re.search(pattern, processed_text, re.IGNORECASE)
            if match:
                found_regions.append({
                    'name': region,
                    'start': match.start(),
                    'content_start': match.end()
                })
        
        # 按找到的顺序排序，以正确切分内容
        found_regions.sort(key=lambda x: x['start'])

        for i, region_info in enumerate(found_regions):
            current_region_name = region_info['name']
            content_start_index = region_info['content_start']
            content_end_index = len(processed_text)
            if i + 1 < len(found_regions):
                content_end_index = found_regions[i+1]['start']
            content = processed_text[content_start_index:content_end_index].strip()
            report_dict[current_region_name] = content

        # 为未在报告中明确提及的区域设置默认值
        for region in ANATOMICAL_REGIONS:
            if region not in report_dict:
                report_dict[region] = "normal"
                # report_dict[region] = ""
        return report_dict

    # --- TP/FP/FN 和 F1 计算函数 ---
    def _calculate_tp_fp_fn(self, gt_set, out_set, category_data):
        # gt_set 和 out_set 已经是剥离空格的关键词列表
        # if not gt_set and not out_set: return 0, 0, 0
        # if not gt_set: return 0, len(out_set), 0
        # if not out_set: return 0, 0, len(gt_set)

        if not gt_set and not out_set: return 0, 0, 0
        if not gt_set: 
            out_representative = {self._map_to_representative_synonym((' ' + keyword), category_data) for keyword in out_set}
            return 0, len(out_representative), 0
        if not out_set: 
            gt_representative = {self._map_to_representative_synonym((' ' + keyword), category_data) for keyword in gt_set}
            return 0, 0, len(gt_representative)

        # gt_representative = {self._map_to_representative_synonym(keyword, category_data) for keyword in gt_set}
        # out_representative = {self._map_to_representative_synonym(keyword, category_data) for keyword in out_set}

        gt_representative = {self._map_to_representative_synonym((' ' + keyword), category_data) for keyword in gt_set}
        out_representative = {self._map_to_representative_synonym((' ' + keyword), category_data) for keyword in out_set}
        
        tp = sum(1 for gt_key in gt_representative if any(gt_key in out_synonyms for out_synonyms in map(lambda x: self._map_to_synonyms(x, category_data), out_representative)))
        # tp = sum(1 for gt_key in gt_representative if any(gt_key in out_synonyms for out_synonyms in map(lambda x: map_to_synonyms((x), json_data), out_representative)))
        fp = len(out_representative) - tp
        fn = len(gt_representative) - tp
        # if tp < 1:
        #     ipdb.set_trace()
        return tp, fp, fn

    def _calculate_f1_from_counts(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1

    
    def calculate_rewards(self, reward_texts,mini_dataset):
        """
        为一批生成的文本计算奖励分数，采用新的统一逻辑。

        新逻辑：
        1. 如果GT和Pred的所有区域都为'normal'，则奖励为 1.0。
        2. 否则，奖励为所有非'normal'区域的关键词的Micro-F1分数。

        :param reward_texts: 一个列表，第一个元素是GT文本，其余是模型生成的文本。
        :return: 一个PyTorch张量，包含每个生成文本的奖励分数。
        """
        gt_full_text = reward_texts[0]
        completions = reward_texts[1:]
        
        if mini_dataset == 'merlin':
            ANATOMICAL_REGIONS = ANATOMICAL_REGIONS_merlin
        if mini_dataset == 'atlas':
            ANATOMICAL_REGIONS = ANATOMICAL_REGIONS_atlas
        if mini_dataset in ['ct_rate','inspect','bimcv']:
            ANATOMICAL_REGIONS = ANATOMICAL_REGIONS_chest

        if mini_dataset in ['ct_rate','inspect','merlin','bimcv']:
            KEYWORD_CATEGORIES = KEYWORD_CATEGORIES_all
        if mini_dataset =='atlas':
            KEYWORD_CATEGORIES = KEYWORD_CATEGORIES_all
        # 1. 解析GT报告，并检查GT是否所有区域都为normal
        gt_regions = self._parse_structured_report(gt_full_text,ANATOMICAL_REGIONS)
        is_gt_all_normal = all(
            (gt_regions.get(region, "normal").strip() in ('normal', ''))
            for region in ANATOMICAL_REGIONS
        )

        final_rewards = []
        for comp_text in completions:
            repeat = find_repeated_strings(comp_text)
            if repeat:
                final_rewards.append(0.0)
                continue
            # 2. 解析每个生成的文本
            pred_regions = self._parse_structured_report(comp_text,ANATOMICAL_REGIONS)

            has_empty_region = any(
                pred_regions.get(region, "").strip() == ""
                for region in ANATOMICAL_REGIONS
            )
            
            if has_empty_region:
                final_rewards.append(0.0)
                continue
            
            # 3. 检查生成的文本是否所有区域都为normal
            is_pred_all_normal = all(
                (pred_regions.get(region, "normal").strip() in ('normal', ''))
                for region in ANATOMICAL_REGIONS
            )
            
            # --- 条件一：如果GT和Pred所有区域都为normal，奖励为1.0 ---
            if is_gt_all_normal and is_pred_all_normal:
                final_rewards.append(1.0)
                continue

            # --- 条件二：否则，计算Micro-F1分数作为奖励 ---
            total_tp, total_fp, total_fn = 0, 0, 0
            
            for region in ANATOMICAL_REGIONS:
                gt_region_text = gt_regions.get(region, "normal").strip()
                pred_region_text = pred_regions.get(region, "normal").strip()

                is_gt_region_normal = (gt_region_text in ('normal', ''))
                is_pred_region_normal = (pred_region_text in ('normal', ''))
                
                # 如果这个区域在两份报告中都是normal，则跳过，不参与F1计算
                if is_gt_region_normal and is_pred_region_normal:
                    continue

                # 只要有一方不是normal，就遍历所有关键词类别进行计算
                for category in KEYWORD_CATEGORIES:
                    # category_data = self.json_data[category]
                    if mini_dataset == 'ct_rate':
                        category_data = self.json_data_ctrate[category]
                    if mini_dataset == 'inspect':
                        category_data = self.json_data_inspect[category]
                    if mini_dataset == 'bimcv':
                        category_data = self.json_data_bimcv[category]
                    if mini_dataset in ['merlin','atlas']:
                        category_data = self.json_data_abodmen[category]
                    
                    # 如果GT的该区域是normal，则其关键词集为空
                    gt_keywords = set() if is_gt_region_normal else set(self._extract_keywords_from_text(gt_region_text, category_data))
                    
                    # 如果Pred的该区域是normal，则其关键词集为空
                    out_keywords = set() if is_pred_region_normal else set(self._extract_keywords_from_text(pred_region_text, category_data))

                    # 计算并累加TP, FP, FN
                    tp, fp, fn = self._calculate_tp_fp_fn(gt_keywords, out_keywords, category_data)
                    # if tp > 0:
                    #     ipdb.set_trace()
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn


            reward = self._calculate_f1_from_counts(total_tp, total_fp, total_fn)
            final_rewards.append(reward)
                
        return torch.tensor(final_rewards, dtype=torch.float)

class Qwen2VLGRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        model_args: GRPOConfig = None ,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        # ipdb.set_trace()
        self.model_args = model_args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            model_name = "Qwen2.5-VL"
            args = GRPOConfig(f"{model_name}-GRPO")

        model = R2GenGPT(model_args)
        if is_deepspeed_zero3_enabled():
            self.ref_model = R2GenGPT(model_args)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None


        processing_class = AutoProcessor.from_pretrained("../hf/qwen25_vl")
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        processing_class.image_processor.max_pixels = max_pixels
        processing_class.image_processor.min_pixels = min_pixels


        # 假设您的关键词JSON文件路径
        keyword_json_path = model_args.forte_path
        print(keyword_json_path)
        # ipdb.set_trace()

        # 实例化奖励计算器
        # 您只需要在训练开始时执行一次
        self.reward_calculator = ComplexRewardCalculator(keyword_json_path)
        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        # model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)



    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, samples,num_generations):
        logits = model(input_ids=input_ids, attention_mask=attention_mask,samples=samples, num_generations=num_generations).logits  # (B, L, V)  # function 1
        
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # fix
            # inputs_embeds, input_mask, input_ids = model.prompt_with_text_prompt_infer(inputs[0])
            

            prompt_completion_ids,input_ids,input_mask,inputs_embeds = unwrapped_model.generate(inputs[0],self.num_generations,self.max_completion_length)  # function 2

            prompt_length = input_ids.size(1)
            prompt_ids = input_ids.repeat_interleave(self.num_generations, dim=0)
            # print(len(inputs))
            # print(prompt_ids.size())
            # print(prompt_completion_ids.size())
            # ipdb.set_trace()
            prompt_completion_ids = torch.cat([prompt_ids, prompt_completion_ids], dim=1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = input_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        # is_eos = completion_ids == self.processing_class.pad_token_id 
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # ipdb.set_trace()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        inputs_embeds = inputs_embeds.repeat_interleave(self.num_generations, dim=0)
        # pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1)
        # image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)
        # per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        # per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, completion_ids, attention_mask,inputs_embeds)
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, inputs[0], self.num_generations)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        # with torch.inference_mode():
        with torch.no_grad(): 
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, inputs[0], self.num_generations)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, completion_ids, attention_mask,inputs_embeds)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

        # Compute the KL divergence between the model and the reference model
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]


        reward_text = [inputs[0]['input_text']] + completions
        mini_dataset = inputs[0]['dataset']
        # reward_text_tokenized = self.reward_tokenizer(reward_text,return_tensors='pt',max_length=512,padding='max_length',truncation=True).to(self.reward_model.model.device)
        # out = self.reward_model(reward_text_tokenized['input_ids'],reward_text_tokenized['attention_mask']).detach().cpu().numpy()
        # pred_labels = expit(out)
        # pred_labels[pred_labels>=0.5]=1
        # pred_labels[pred_labels<0.5]=0
        # rewards = calculate_f1(pred_labels).to(self.reward_model.model.device).float()
        if self.model_args.f1_cal == 'micro':
            rewards = self.reward_calculator.calculate_rewards(reward_text,mini_dataset).to(inputs_embeds.device).float()
        # if self.model_args.f1_cal == 'macro':
        #     rewards = self.reward_calculator.calculate_rewards_macro_f1(reward_text,mini_dataset).to(inputs_embeds.device).float()
        # ipdb.set_trace()
        # print(self.model_args.f1_cal)
        print(mini_dataset)
        print(rewards)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        # per_token_loss = -(per_token_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
