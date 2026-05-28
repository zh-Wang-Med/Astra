import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
# from transformers import SwinModel
# from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
import ipdb
from models.utils import PerceiverResampler
from transformers import get_cosine_schedule_with_warmup
from merlin import Merlin
from models.vit_3d import ViT
from collections import defaultdict  # <--- 添加这一行
import gc
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


class Astra(pl.LightningModule):
    """
    Astra model.
    """
    def __init__(self, args):
        super().__init__()
        torch.cuda.empty_cache()
        gc.collect()
        self.args = args
        self.save_hyperparameters(args)

        if args.visual_encoder_name == "merlin":
            self.visual_encoder = Merlin(ImageEmbedding=True)
        if args.visual_encoder_name == "radfm":
            self.visual_encoder = ViT(
                image_size=512,          # image size
                frames=512,               # max number of frames
                image_patch_size=32,     # image patch size
                frame_patch_size=4,      # frame patch size
                dim=768,
                depth=12,
                heads=8,
                mlp_dim=2048,
                dropout=0.1,
                emb_dropout=0.1
            )
            vit3d_ckpt = torch.load("../RadFM_vit3d.pth", map_location='cpu')
            self.visual_encoder.load_state_dict(vit3d_ckpt, strict=True)


        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False

        print('Loading qwen2.5_vl')
        self.qwen_processor = AutoProcessor.from_pretrained("../qwen25vl")
        self.tokenizer = self.qwen_processor.tokenizer
        if args.precision == "bf16-mixed":
            self.language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("../qwen25vl", torch_dtype=torch.bfloat16)
        else:
            self.language_model = Qwen2_5_VLForConditionalGeneration.from_pretrained("../qwen25vl", torch_dtype=torch.float32)
        del self.language_model.visual
         
        if args.llm_use_lora:
            self.embed_tokens = self.language_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.llm_r, lora_alpha=args.llm_alpha, lora_dropout=args.lora_dropout, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            )
            self.language_model = get_peft_model(self.language_model, peft_config)
            self.language_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.language_model.get_input_embeddings()
            for name, param in self.language_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done, frozee parameters')

        # self.perceiver = PerceiverResampler(dim=2048,dim_head=256,heads=8,num_latents=64)
        if args.whether_perceiver is True:
            # self.perceiver = PerceiverResampler(dim=768,num_latents=32)
            self.perceiver = PerceiverResampler(dim=args.vision_dim,dim_head=args.perceiver_dim_head,heads=args.perceiver_heads,num_latents=args.vision_token_number)
            if args.perceiver_whether_inital is True and args.perceiver_dim_head == 64 and args.perceiver_heads == 8 and args.vision_token_number == 32 and args.vision_dim == 768:
                state_dict = torch.load("../RadFM_perceiver_fc.pth", map_location='cpu')
                self.perceiver.load_state_dict(state_dict['perceiver'])
        
        self.llama_proj = nn.Linear(args.vision_dim,3584)
        self.layer_norm = nn.LayerNorm(self.language_model.config.hidden_size)
        # self.end_sym = args.end_sym

        self.chest_prompt = "Generate a comprehensive and detailed diagnosis report for this chest CT image. Structure the report by describing the following regions in this exact order: abdomen, bone, breast, esophagus, heart, lung, mediastinum, pleura, thyroid, trachea and bronchie. For any region without abnormalities, state 'normal.'."
        self.merlin_prompt = "Generate a comprehensive and detailed diagnosis report for this abdomen CT image. Structure the report by describing the following regions in this exact order: lower thorax, liver and biliary tree, gallbladder, spleen, pancreas, adrenal glands, kidneys and ureters, gastrointestinal tract, peritoneum, pelvic, vasculature, lymph nodes, musculoskeletal. For any region without abnormalities, state 'normal.'."
        self.atlas_prompt = "Please analyze the liver and biliary tree, pancreas, and kidneys and ureters areas from this abdominal CT scan. For any region without abnormalities, state 'normal.'."

        self.end_sym = "<|im_end|>"
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if self.args.stage1_frozen == False:
            if args.delta_file is not None:
                if self.args.llm_use_lora:
                    state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
                else:
                    state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))
                self.load_state_dict(state_dict=state_dict, strict=False)
                print(f'Load checkpoint from {args.delta_file}')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        # print(images.shape)
        print(f"Max: {images.max():.4f}, Min: {images.min():.4f}" + (" ⚠️ ATTENTION" if images.max() < 0.8 or images.min() > 0.2 else ""))

        if self.args.visual_encoder_name == "radfm":
            image_embeds, pos_embedding = self.visual_encoder(images)
        if self.args.visual_encoder_name == "merlin":
            image_embeds = self.visual_encoder(images)
            b, c, x, y, z = image_embeds.shape
            print(image_embeds.shape)
            image_embeds = image_embeds.permute(0, 2, 3, 4, 1)
            image_embeds = image_embeds.reshape(b, x*y*z, c)
        if self.args.whether_perceiver is True:
            image_embeds = self.perceiver(image_embeds.unsqueeze(1).unsqueeze(1)).squeeze(1)
        inputs_llama = self.llama_proj(image_embeds)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(images.device)
        return inputs_llama, atts_llama
    



    def prompt_dy_wrap_train(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        batch_size = img_embeds.shape[0]
        prompt = []
        for b in range(batch_size):
            dataset = samples['dataset'][b]
            gt = samples["input_text"][b]
            if dataset == 'merlin':
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.merlin_prompt}<|im_end|>\n<|im_start|>assistant\n{gt}{self.end_sym}')
            if dataset == 'atlas':
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.atlas_prompt}<|im_end|>\n<|im_start|>assistant\n{gt}{self.end_sym}')
            if dataset in ['ct_rate','inspect','bimcv']:
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.chest_prompt}<|im_end|>\n<|im_start|>assistant\n{gt}{self.end_sym}')

        to_regress_tokens = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(img_embeds.device)
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        # 替换<|image_pad|>为img_embeds
        image_pad_id = 151655
        for b in range(batch_size):
            image_pad_positions = (to_regress_tokens.input_ids[b] == image_pad_id).nonzero().squeeze()
            if len(image_pad_positions) >= 32:
                start_pos = image_pad_positions[0]
                to_regress_embeds[b, start_pos:start_pos+32] = img_embeds[b]
            else:
                print(f"Warning: Sample {b} does not have enough image_pad tokens.")

        # 创建targets并设置masked positions
        targets = to_regress_tokens.input_ids.clone()

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        # 找到每个样本中<|im_start|>assistant的位置，并将其之前的所有位置设为-100
        im_start_assistant_id = 151644
        for b in range(batch_size):
            assistant_pos = (targets[b] == im_start_assistant_id).nonzero().squeeze()
            if assistant_pos.dim() == 0:
                assistant_pos = assistant_pos.unsqueeze(0)
            if len(assistant_pos) > 0:
                last_assistant_pos = assistant_pos[-1]
                targets[b, :last_assistant_pos + 2] = -100  # +2 to account for 'assistant\n'

        wrapped_img_embeds = to_regress_embeds
        wrapped_atts_img = to_regress_tokens.attention_mask
        return wrapped_img_embeds, wrapped_atts_img, targets

    def prompt_dy_wrap_infer(self, samples):
        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds = self.layer_norm(img_embeds)
        batch_size = img_embeds.shape[0]
        prompt = []
        for b in range(batch_size):
            dataset = samples['dataset'][b]
            gt = samples["input_text"][b]
            if dataset in ['merlin','amosmm','guizhou_abdomen','inhouse_duilie1']:
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.merlin_prompt}<|im_end|>\n<|im_start|>assistant\n')
            if dataset == 'atlas':
                prompt.append(f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>{("<|image_pad|>" * 32)}<|vision_end|>{self.atlas_prompt}<|im_end|>\n<|im_start|>assistant\n')
            if dataset in ['ct_rate','inspect','bimcv','ctrg','radchest','nlst','guizhou','rsna_pe_v1','rsna_pe_v2']:
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
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        self.tokenizer.padding_side = pad_size

        to_regress_embeds_writable = to_regress_embeds.clone()
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
        return wrapped_img_embeds, wrapped_atts_img

    def forward(self, samples):
        inputs_embeds, attention_mask, targets = self.prompt_dy_wrap_train(samples)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        # ipdb.set_trace()
        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        # param_grad_dic = {
        #     k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        # }
        state_dict = self.state_dict()
        # for k in list(state_dict.keys()):
        #     if k not in param_grad_dic.keys():
        #         del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
        del state_dict

    def save_checkpoint_merge(self):
        self.language_model = self.language_model.merge_and_unload()
        # ipdb.set_trace()
        state_dict = self.state_dict()
        # for k in list(state_dict.keys()):
        #     if k not in param_grad_dic.keys():
        #         del state_dict[k]
        save_obj = {
            "model": state_dict,
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "merged_dict.pth",
        )
        torch.save(save_obj, save_to)
        del state_dict
    
    def validation_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        to_regress_tokens = self.tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )
    
        img_embeds, atts_img = self.prompt_dy_wrap_infer(samples)

        inputs_embeds = img_embeds
        attention_mask = atts_img

        print(inputs_embeds.shape)
        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
        )
        hypo = self.qwen_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ref = self.qwen_processor.batch_decode(to_regress_tokens['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        print(hypo)
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('</s>')[0].strip()
        output_text = output_text.replace('<unk>', '')
        return output_text

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            # if val_score > self.val_score:
            self.save_checkpoint(eval_res)
            self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.tokenizer.padding_side = "right"
        to_regress_tokens = self.tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        img_embeds, atts_img = self.prompt_dy_wrap_infer(samples)

        inputs_embeds = img_embeds
        attention_mask = atts_img

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
        )

        hypo = self.qwen_processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ref = self.qwen_processor.batch_decode(to_regress_tokens['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        print(hypo)
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

