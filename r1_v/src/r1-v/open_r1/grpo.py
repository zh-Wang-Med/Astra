# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration
from data_helper import create_datasets
# from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer, Qwen2VLGRPOVLLMTrainerModified,Qwen2VLGRPOTrainer_NLG
from open_r1.trainer.my_model import R2GenGPT
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import ipdb
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
import deepspeed
import torch
import logging

class SuppressPyRuSHFilter(logging.Filter):
    def filter(self, record):
        # 如果日志信息中包含这个特定的字符串，就返回 False 屏蔽它
        return "not a eligible syntax" not in record.getMessage()

# 将过滤器添加到根记录器 (root logger)
logging.getLogger().addFilter(SuppressPyRuSHFilter())

from loguru import logger
import sys

# 移除默认的控制台输出处理器
logger.remove()
# 重新添加处理器，但将级别设为 INFO 或更高（这样 DEBUG 就不显示了）
logger.add(sys.stderr, level="INFO")


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    
    # New arguments from data_helper
    dataset: str = field(default="merge", metadata={"help": "Dataset name"})
    annotation: str = field(default="../ct_rate_with_npz.json", metadata={"help": "Path to annotation file"})
    with_prompt: bool = field(default=True, metadata={"help": "Whether to include prompt"})
    with_vision_prompt: bool = field(default=False, metadata={"help": "Whether to include vision prompt"})
    vision_prompt_type: str = field(default="v2", metadata={"help": "Type of vision prompt"})
    visual_encoder_name: str = field(default="merlin", metadata={"help": "Name of visual encoder"})
    batch_image: bool = field(default=False, metadata={"help": "Whether to batch images"})
    cam_number: int = field(default=6, metadata={"help": "Number of CAMs"})
    region_mask: bool = field(default=False, metadata={"help": "Whether to use region mask"})
    anatomy_number: int = field(default=3, metadata={"help": "Number of anatomies"})
    model_v2_type: str = field(default="v1", metadata={"help": "Model v2 type"})
    forte_path: str = field(default="../FORTE-main/data/FORTE_abdomen_icd1208.json", metadata={"help": "Model v2 type"})
    f1_cal: str = field(default="micro", metadata={"help": "Model v2 type"})
    prompt_style: str = field(default="v2", metadata={"help": "Model v2 type"})
    reward_function: str = field(default="base", metadata={"help": "Model v2 type"})
    stage_class: int = field(default=1, metadata={"help": "Stage class"})
    stage1_vqa: bool = field(default=False, metadata={"help": "Whether to use stage1 VQA"})
    whether_npz: bool = field(default=True, metadata={"help": "Whether to use stage1 VQA"})
    whether_space: bool = field(default=True, metadata={"help": "Whether to use stage1 VQA"})
    
    

def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            answer = parse(content)
            if float(verify(answer, parse(sol))) > 0:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    # ipdb.set_trace()

    train_dataset, val_dataset = create_datasets(script_args)
    if script_args.reward_function == "base":
        trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainerModified
    if script_args.reward_function == "NLG":
        trainer_cls = Qwen2VLGRPOTrainer_NLG
    print("using: ", trainer_cls)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        model_args = script_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # ipdb.set_trace()
    print(script_args)
    print(training_args)
    print(model_args)
    training_args.torch_compile = False
    main(script_args, training_args, model_args)
